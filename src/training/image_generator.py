"""Image generation (diffusion decoder) for Terra multimodal model.

A small latent diffusion model conditioned on Terra's text embeddings.
The reverse of vision understanding: instead of image → embeddings, this
does embeddings → image.

How diffusion works:
1. Forward: take a clean image, add noise over T steps until it's pure static
2. Train: given a noisy image + timestep + text condition, predict the noise
3. Generate: start from pure noise, denoise step by step, conditioned on text

Architecture:
- Variational Autoencoder (VAE): compress images to a small latent space
- U-Net denoiser: predict noise in latent space, conditioned on Terra embeddings
- Text conditioning: Terra LLM hidden states → cross-attention in U-Net

This is the same core idea as Stable Diffusion, but much smaller (~30-80M params)
to fit on consumer hardware alongside the LLM.

Training pipeline:
1. Pre-train VAE on image reconstruction (independent of LLM)
2. Train U-Net denoiser conditioned on CLIP/SigLIP text embeddings
3. Swap conditioning to Terra's own embeddings and fine-tune
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImageGenConfig:
    """Image generation (diffusion) configuration."""

    # Image / latent space
    image_size: int = 256
    latent_channels: int = 4
    latent_size: int = 32  # image_size // 8 (VAE downsamples 8x)

    # VAE
    vae_hidden: int = 128
    vae_layers: int = 2

    # U-Net denoiser
    unet_channels: int = 128  # Base channel count
    unet_channel_mult: tuple[int, ...] = (1, 2, 4)  # Channel multipliers per resolution
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: tuple[int, ...] = (16, 8)  # Apply attention at these sizes
    unet_num_heads: int = 4

    # Conditioning from LLM
    context_dim: int = 896  # Terra's hidden_size

    # Diffusion process
    num_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012

    dropout: float = 0.0

    @classmethod
    def gen_tiny(cls, context_dim: int = 896) -> "ImageGenConfig":
        """~8M params - fast experiments, 64x64 images."""
        return cls(
            image_size=64,
            latent_size=8,
            vae_hidden=64,
            unet_channels=64,
            unet_channel_mult=(1, 2, 4),
            unet_num_res_blocks=1,
            unet_attention_resolutions=(8,),
            unet_num_heads=4,
            context_dim=context_dim,
        )

    @classmethod
    def gen_small(cls, context_dim: int = 896) -> "ImageGenConfig":
        """~30M params - decent quality, 256x256 images."""
        return cls(
            image_size=256,
            latent_size=32,
            vae_hidden=128,
            unet_channels=128,
            unet_channel_mult=(1, 2, 4),
            unet_num_res_blocks=2,
            unet_attention_resolutions=(16, 8),
            unet_num_heads=4,
            context_dim=context_dim,
        )

    @classmethod
    def gen_base(cls, context_dim: int = 896) -> "ImageGenConfig":
        """~80M params - good quality, 512x512 images."""
        return cls(
            image_size=512,
            latent_size=64,
            vae_hidden=128,
            vae_layers=3,
            unet_channels=192,
            unet_channel_mult=(1, 2, 3, 4),
            unet_num_res_blocks=2,
            unet_attention_resolutions=(32, 16, 8),
            unet_num_heads=8,
            context_dim=context_dim,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ImageGenConfig":
        valid = {}
        for k, v in d.items():
            if k in cls.__dataclass_fields__:
                if k in ("unet_channel_mult", "unet_attention_resolutions") and isinstance(v, list):
                    v = tuple(v)
                valid[k] = v
        return cls(**valid)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["unet_channel_mult"] = list(d["unet_channel_mult"])
        d["unet_attention_resolutions"] = list(d["unet_attention_resolutions"])
        return d

    def param_count_estimate(self) -> int:
        # VAE (encoder + decoder)
        vae = 2 * self.vae_hidden * self.vae_hidden * 9 * self.vae_layers * 4
        # U-Net
        base = self.unet_channels
        total_ch = sum(base * m for m in self.unet_channel_mult)
        unet = total_ch * total_ch * 9 * self.unet_num_res_blocks * 2  # rough
        # Cross-attention
        xattn = total_ch * self.context_dim * 4 * len(self.unet_attention_resolutions)
        return vae + unet + xattn


# ── Diffusion Noise Schedule ──

class DiffusionSchedule:
    """Cosine noise schedule for the diffusion process."""

    def __init__(self, config: ImageGenConfig):
        self.num_timesteps = config.num_timesteps
        # Linear beta schedule (can swap for cosine later)
        betas = torch.linspace(config.beta_start, config.beta_end, config.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register = {}
        self.register["betas"] = betas
        self.register["alphas_cumprod"] = alphas_cumprod
        self.register["sqrt_alphas_cumprod"] = torch.sqrt(alphas_cumprod)
        self.register["sqrt_one_minus_alphas_cumprod"] = torch.sqrt(1.0 - alphas_cumprod)
        self.register["sqrt_recip_alphas"] = torch.sqrt(1.0 / alphas)
        self.register["posterior_variance"] = betas * (1.0 - torch.cat([torch.tensor([0.0]), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)

    def _get(self, name: str, t: torch.Tensor) -> torch.Tensor:
        """Index schedule values by timestep."""
        vals = self.register[name].to(t.device)
        out = vals[t]
        # Reshape for broadcasting with (B, C, H, W)
        return out.view(-1, 1, 1, 1)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward process: add noise to x_start at timestep t.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._get("sqrt_alphas_cumprod", t)
        sqrt_one_minus_alpha = self._get("sqrt_one_minus_alphas_cumprod", t)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse process: one denoising step.

        Given predicted noise, compute x_{t-1} from x_t.
        """
        sqrt_recip_alpha = self._get("sqrt_recip_alphas", t)
        beta = self._get("betas", t)
        sqrt_one_minus_alpha = self._get("sqrt_one_minus_alphas_cumprod", t)

        # Predicted mean
        pred_mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha * model_output)

        if (t == 0).all():
            return pred_mean

        posterior_var = self._get("posterior_variance", t)
        noise = torch.randn_like(x_t)
        return pred_mean + torch.sqrt(posterior_var) * noise


# ── VAE (Variational Autoencoder) ──

class VAEResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class VAEEncoder(nn.Module):
    """Encode images to latent space. Downsamples 8x."""

    def __init__(self, config: ImageGenConfig):
        super().__init__()
        h = config.vae_hidden
        self.conv_in = nn.Conv2d(3, h, 3, padding=1)
        # 3 downsample stages: /2 /2 /2 = /8
        self.down1 = nn.Sequential(VAEResBlock(h), nn.Conv2d(h, h, 3, stride=2, padding=1))
        self.down2 = nn.Sequential(VAEResBlock(h), nn.Conv2d(h, h * 2, 3, stride=2, padding=1))
        self.down3 = nn.Sequential(VAEResBlock(h * 2), nn.Conv2d(h * 2, h * 2, 3, stride=2, padding=1))
        self.mid = VAEResBlock(h * 2)
        # Output: mean and logvar for reparameterization
        self.conv_out = nn.Conv2d(h * 2, config.latent_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar) of the latent distribution."""
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.conv_out(h)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class VAEDecoder(nn.Module):
    """Decode latent space back to images. Upsamples 8x."""

    def __init__(self, config: ImageGenConfig):
        super().__init__()
        h = config.vae_hidden
        self.conv_in = nn.Conv2d(config.latent_channels, h * 2, 3, padding=1)
        self.mid = VAEResBlock(h * 2)
        # 3 upsample stages
        self.up1 = nn.Sequential(VAEResBlock(h * 2), nn.ConvTranspose2d(h * 2, h * 2, 4, stride=2, padding=1))
        self.up2 = nn.Sequential(VAEResBlock(h * 2), nn.ConvTranspose2d(h * 2, h, 4, stride=2, padding=1))
        self.up3 = nn.Sequential(VAEResBlock(h), nn.ConvTranspose2d(h, h, 4, stride=2, padding=1))
        self.norm_out = nn.GroupNorm(32, h)
        self.conv_out = nn.Conv2d(h, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class TerraVAE(nn.Module):
    """VAE for compressing images to/from latent space."""

    def __init__(self, config: ImageGenConfig):
        super().__init__()
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent (with reparameterization trick)."""
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward: encode → sample → decode.

        Returns: (reconstructed, mean, logvar) for computing loss.
        """
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon, mean, logvar

    @staticmethod
    def loss(recon: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE loss = reconstruction + KL divergence."""
        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + 0.001 * kl_loss  # Small KL weight to avoid posterior collapse


# ── U-Net Denoiser ──

class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class CrossAttention(nn.Module):
    """Cross-attention: image features attend to text (LLM) conditioning."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.norm = nn.GroupNorm(_num_groups(query_dim), query_dim)
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image features
            context: (B, seq, context_dim) text embeddings from Terra LLM
        """
        B, C, H, W = x.shape
        residual = x

        x_flat = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(attn).permute(0, 2, 1).view(B, C, H, W)

        return residual + out


def _num_groups(channels: int, target: int = 32) -> int:
    """Pick a valid group count for GroupNorm."""
    for g in (target, 16, 8, 4, 1):
        if channels % g == 0:
            return g
    return 1


class UNetResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # Add time embedding
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNetBlock(nn.Module):
    """U-Net block: residual blocks + optional cross-attention."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, context_dim: int,
                 num_res_blocks: int = 2, has_attention: bool = False, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            ch_in = in_ch if i == 0 else out_ch
            self.res_blocks.append(UNetResBlock(ch_in, out_ch, time_dim, dropout))
            if has_attention:
                self.attn_blocks.append(CrossAttention(out_ch, context_dim, num_heads))
            else:
                self.attn_blocks.append(None)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x, t_emb)
            if attn is not None:
                x = attn(x, context)
        return x


class TerraUNet(nn.Module):
    """U-Net denoiser conditioned on timestep + Terra LLM text embeddings.

    Predicts noise ε given (noisy_latent, timestep, text_context).
    """

    def __init__(self, config: ImageGenConfig):
        super().__init__()
        ch = config.unet_channels
        mults = config.unet_channel_mult
        time_dim = ch * 4
        context_dim = config.context_dim

        # Time embedding
        self.time_embed = TimeEmbedding(ch)

        # Input projection
        self.conv_in = nn.Conv2d(config.latent_channels, ch, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        # Track channel sizes at each level for skip connections
        down_channels = []  # channels BEFORE each downsample
        in_ch = ch
        cur_size = config.latent_size

        for i, mult in enumerate(mults):
            out_ch = ch * mult
            has_attn = cur_size in config.unet_attention_resolutions
            self.down_blocks.append(UNetBlock(
                in_ch, out_ch, time_dim, context_dim,
                config.unet_num_res_blocks, has_attn, config.unet_num_heads, config.dropout,
            ))
            down_channels.append(out_ch)
            in_ch = out_ch
            if i < len(mults) - 1:
                self.downsamplers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
                cur_size //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle
        mid_ch = ch * mults[-1]
        self.mid_block = UNetBlock(
            mid_ch, mid_ch, time_dim, context_dim, 1, True, config.unet_num_heads, config.dropout,
        )

        # Upsampling path (mirror of down path)
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        prev_ch = mid_ch  # Output of bottleneck

        for i, mult in enumerate(reversed(mults)):
            out_ch = ch * mult
            skip_ch = down_channels.pop()  # Skip connection from corresponding down level
            has_attn = cur_size in config.unet_attention_resolutions
            self.up_blocks.append(UNetBlock(
                prev_ch + skip_ch, out_ch, time_dim, context_dim,
                config.unet_num_res_blocks, has_attn, config.unet_num_heads, config.dropout,
            ))
            prev_ch = out_ch
            if i < len(mults) - 1:
                self.upsamplers.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
                cur_size *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # Output: prev_ch is now ch * mults[0]
        out_norm_ch = ch * mults[0]
        self.norm_out = nn.GroupNorm(min(32, out_norm_ch), out_norm_ch)
        self.conv_out = nn.Conv2d(out_norm_ch, config.latent_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, latent_channels, H, W) noisy latent
            t: (B,) integer timesteps
            context: (B, seq, context_dim) text embeddings from Terra LLM
        Returns:
            (B, latent_channels, H, W) predicted noise
        """
        t_emb = self.time_embed(t)
        h = self.conv_in(x)

        # Down path (save skip connections)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamplers):
            h = block(h, t_emb, context)
            skips.append(h)
            h = down(h)

        # Middle
        h = self.mid_block(h, t_emb, context)

        # Up path (use skip connections in reverse)
        for block, up in zip(self.up_blocks, self.upsamplers):
            skip = skips.pop()
            # Ensure spatial dims match (handle off-by-one from strided convs)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb, context)
            h = up(h)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ── Full Image Generator ──

class TerraImageGenerator(nn.Module):
    """Complete text-to-image pipeline: Terra LLM embeddings → latent diffusion → image.

    Training:
    1. Pre-train VAE on image reconstruction (independent)
    2. Freeze VAE, train U-Net to denoise latents conditioned on text
    3. Fine-tune conditioning on Terra's own embeddings

    Inference:
    1. Get text embeddings from Terra LLM
    2. Start from random noise in latent space
    3. Denoise for N steps conditioned on text
    4. Decode latent → image with VAE decoder
    """

    def __init__(self, config: ImageGenConfig):
        super().__init__()
        self.config = config
        self.vae = TerraVAE(config)
        self.unet = TerraUNet(config)
        self.schedule = DiffusionSchedule(config)

    def training_step(
        self,
        images: torch.Tensor,
        context: torch.Tensor,
    ) -> dict:
        """One training step: encode image, add noise, predict noise.

        Args:
            images: (B, 3, H, W) real images
            context: (B, seq, context_dim) text embeddings from Terra LLM
        Returns:
            dict with 'loss', 'diffusion_loss', 'vae_loss'
        """
        B = images.shape[0]

        # Encode to latent space (detach VAE if frozen)
        with torch.no_grad():
            z = self.vae.encode(images)

        # Sample random timesteps
        t = torch.randint(0, self.config.num_timesteps, (B,), device=images.device)

        # Add noise
        noise = torch.randn_like(z)
        z_noisy = self.schedule.q_sample(z, t, noise)

        # Predict noise
        noise_pred = self.unet(z_noisy, t, context)

        # MSE loss on noise prediction
        diffusion_loss = F.mse_loss(noise_pred, noise)

        return {"loss": diffusion_loss, "diffusion_loss": diffusion_loss}

    def train_vae_step(self, images: torch.Tensor) -> dict:
        """Pre-train the VAE on image reconstruction.

        Args:
            images: (B, 3, H, W)
        Returns:
            dict with 'loss', 'recon_loss', 'kl_loss'
        """
        recon, mean, logvar = self.vae(images)
        loss = TerraVAE.loss(recon, images, mean, logvar)
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """Generate images from text embeddings.

        Args:
            context: (B, seq, context_dim) text embeddings from Terra LLM
            num_steps: denoising steps (fewer = faster, more = better quality)
            guidance_scale: classifier-free guidance scale (higher = more faithful to text)
        Returns:
            (B, 3, H, W) generated images in [-1, 1]
        """
        B = context.shape[0]
        device = context.device

        # Start from pure noise
        z = torch.randn(B, self.config.latent_channels, self.config.latent_size, self.config.latent_size, device=device)

        # Use evenly spaced timesteps for fewer steps
        step_size = self.config.num_timesteps // num_steps
        timesteps = list(range(self.config.num_timesteps - 1, -1, -step_size))

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Predict noise (with classifier-free guidance if scale > 1)
            if guidance_scale > 1.0:
                # Unconditional prediction (context = zeros)
                uncond_context = torch.zeros_like(context)
                noise_uncond = self.unet(z, t, uncond_context)
                noise_cond = self.unet(z, t, context)
                # Guided prediction
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.unet(z, t, context)

            z = self.schedule.p_sample(noise_pred, z, t)

        # Decode latent → image
        images = self.vae.decode(z)
        return images.clamp(-1, 1)

    def consistency_distillation_step(
        self,
        images: torch.Tensor,
        context: torch.Tensor,
    ) -> dict:
        """Consistency distillation: train the model to jump directly to x_0.

        Instead of predicting noise, learn to map ANY noisy image directly
        to the clean output in one step. This is what makes generation fast (1-4 steps).

        The idea: if the model can predict x_0 from x_t at ANY t, then at inference
        you only need 1 step instead of 50+.

        Training: sample two adjacent timesteps t and t-1, ensure the model's
        prediction is consistent (both should map to the same x_0).
        """
        B = images.shape[0]

        with torch.no_grad():
            z_0 = self.vae.encode(images)

        # Sample timestep pairs (t, t-1)
        t = torch.randint(1, self.config.num_timesteps, (B,), device=images.device)
        t_prev = t - 1

        noise = torch.randn_like(z_0)
        z_t = self.schedule.q_sample(z_0, t, noise)
        z_t_prev = self.schedule.q_sample(z_0, t_prev, noise)

        # Model should predict consistent outputs for both timesteps
        # (both should reconstruct the same clean image)
        pred_t = self.unet(z_t, t, context)
        with torch.no_grad():
            pred_t_prev = self.unet(z_t_prev, t_prev, context)

        # Consistency loss: predictions from adjacent timesteps should match
        loss = F.mse_loss(pred_t, pred_t_prev)

        return {"loss": loss, "consistency_loss": loss}

    @torch.no_grad()
    def generate_fast(
        self,
        context: torch.Tensor,
        num_steps: int = 4,
        guidance_scale: float = 5.0,
    ) -> torch.Tensor:
        """Fast generation using few-step denoising.

        After consistency distillation training, this produces good images
        in just 1-4 steps (vs 50+ for standard diffusion).
        Even without distillation, 4-8 steps gives decent results with DDIM-style sampling.
        """
        B = context.shape[0]
        device = context.device

        z = torch.randn(B, self.config.latent_channels, self.config.latent_size, self.config.latent_size, device=device)

        # DDIM-style: evenly spaced timesteps for fewer steps
        step_size = self.config.num_timesteps // num_steps
        timesteps = list(range(self.config.num_timesteps - 1, 0, -step_size))

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), max(0, t_val - step_size), device=device, dtype=torch.long)

            if guidance_scale > 1.0:
                uncond = self.unet(z, t, torch.zeros_like(context))
                cond = self.unet(z, t, context)
                noise_pred = uncond + guidance_scale * (cond - uncond)
            else:
                noise_pred = self.unet(z, t, context)

            # DDIM step (deterministic, no noise added)
            alpha_t = self.schedule._get("alphas_cumprod", t)
            alpha_prev = self.schedule._get("alphas_cumprod", t_prev)

            # Predict x_0
            x0_pred = (z - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            # Jump to t_prev
            z = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        images = self.vae.decode(z)
        return images.clamp(-1, 1)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        vae_params = sum(p.numel() for p in self.vae.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        return {
            "total": total,
            "total_millions": total / 1e6,
            "vae_millions": vae_params / 1e6,
            "unet_millions": unet_params / 1e6,
        }

    def save_pretrained(self, path: str):
        import json
        from pathlib import Path as P
        from safetensors.torch import save_file

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(save_dir / "image_generator.safetensors"))
        (save_dir / "image_gen_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TerraImageGenerator":
        import json
        from pathlib import Path as P
        from safetensors.torch import load_file

        config = ImageGenConfig.from_dict(json.loads((P(path) / "image_gen_config.json").read_text()))
        model = cls(config)
        state_dict = load_file(str(P(path) / "image_generator.safetensors"), device=device)
        model.load_state_dict(state_dict)
        return model
