"""GitHub issue tracker for autonomous task management."""

import json
import os
import subprocess
from dataclasses import dataclass


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    state: str
    priority: int  # computed from labels

    @property
    def is_actionable(self) -> bool:
        return self.state == "open" and "wontfix" not in self.labels


class IssueTracker:
    """Reads and manages GitHub issues for the evolution pipeline."""

    PRIORITY_LABELS = {
        "critical": 0,
        "high-priority": 1,
        "bug": 2,
        "enhancement": 3,
        "research": 4,
        "low-priority": 5,
    }

    def __init__(self):
        self.repo = os.environ.get("GITHUB_REPOSITORY", "")

    def get_open_issues(self) -> list[Issue]:
        """Fetch open issues using GitHub CLI."""
        try:
            result = subprocess.run(
                [
                    "gh", "issue", "list",
                    "--state", "open",
                    "--json", "number,title,body,labels,state",
                    "--limit", "50",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            raw = json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            return []

        issues = []
        for item in raw:
            label_names = [l["name"] for l in item.get("labels", [])]
            priority = min(
                (self.PRIORITY_LABELS.get(l, 99) for l in label_names),
                default=99,
            )
            issues.append(Issue(
                number=item["number"],
                title=item["title"],
                body=item.get("body", ""),
                labels=label_names,
                state=item.get("state", "open"),
                priority=priority,
            ))

        return sorted(issues, key=lambda i: i.priority)

    def get_actionable_issues(self) -> list[Issue]:
        """Get issues that the evolution pipeline can work on."""
        return [i for i in self.get_open_issues() if i.is_actionable]

    def comment_on_issue(self, number: int, body: str):
        """Post a comment on an issue with progress update."""
        try:
            subprocess.run(
                ["gh", "issue", "comment", str(number), "--body", body],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def create_issue(self, title: str, body: str, labels: list[str] | None = None):
        """Create a new issue for discovered improvements."""
        cmd = ["gh", "issue", "create", "--title", title, "--body", body]
        if labels:
            cmd.extend(["--label", ",".join(labels)])
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def categorize_issue(self, issue: Issue) -> str:
        """Determine what kind of work an issue requires."""
        labels = set(issue.labels)
        title_lower = issue.title.lower()
        body_lower = (issue.body or "").lower()

        if labels & {"bug", "fix"}:
            return "bugfix"
        if labels & {"training", "data"}:
            return "training"
        if labels & {"benchmark", "evaluation"}:
            return "evaluation"
        if labels & {"architecture", "model"}:
            return "architecture"
        if labels & {"research"}:
            return "research"
        if any(kw in title_lower or kw in body_lower for kw in ["benchmark", "score", "eval"]):
            return "evaluation"
        if any(kw in title_lower or kw in body_lower for kw in ["train", "data", "finetune"]):
            return "training"
        return "research"
