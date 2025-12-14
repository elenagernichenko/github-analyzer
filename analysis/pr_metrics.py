"""
Calculate social-technical PR metrics for catboost repo via GitHub MCP Server.
"""

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

from mcp_session import MCPDockerSession, extract_text_content

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / "github-mcp" / ".env"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OWNER = "catboost"
REPO = "catboost"
DAYS_BACK = 120
SAMPLE_SIZE = 30


def parse_github_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def hours_between(start: datetime, end: datetime) -> float:
    return round((end - start).total_seconds() / 3600, 2)


@dataclass
class PRRecord:
    number: int
    title: str
    state: str
    html_url: str
    author: str
    created_at: datetime
    closed_at: Optional[datetime]
    merged_at: Optional[datetime]
    issue_comments: List[Dict]
    review_comments: List[Dict]

    def to_dict(self, collected_at: datetime) -> Dict:
        lifetime_end = self.closed_at or collected_at
        lifetime_hours = hours_between(self.created_at, lifetime_end)

        first_response_at = self._first_response_timestamp()
        first_response_hours = (
            hours_between(self.created_at, first_response_at)
            if first_response_at
            else None
        )

        total_comments = len(self.issue_comments) + len(self.review_comments)
        participants = {self.author}
        for item in self.issue_comments + self.review_comments:
            user = (item.get("user") or {}).get("login")
            if user:
                participants.add(user)

        return {
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "html_url": self.html_url,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            "lifetime_hours": lifetime_hours,
            "first_response_at": first_response_at.isoformat()
            if first_response_at
            else None,
            "first_response_hours": first_response_hours,
            "issue_comment_count": len(self.issue_comments),
            "review_comment_count": len(self.review_comments),
            "total_comment_count": total_comments,
            "participants": sorted(participants),
        }

    def _first_response_timestamp(self) -> Optional[datetime]:
        timestamps: List[datetime] = []
        for item in self.issue_comments + self.review_comments:
            created_at = parse_github_datetime(item.get("created_at"))
            if created_at:
                timestamps.append(created_at)
        return min(timestamps) if timestamps else None


def fetch_pr_items(session: MCPDockerSession, start: date, sample_size: int) -> Sequence[Dict]:
    query = f"repo:{OWNER}/{REPO} comments:>0 created:>={start.isoformat()}"
    items: list[Dict] = []
    per_page = min(100, sample_size)
    pages = (sample_size + per_page - 1) // per_page

    for page in range(1, pages + 1):
        response = session.call_tool(
            "search_pull_requests",
            {
                "query": query,
                "sort": "created",
                "order": "desc",
                "perPage": per_page,
                "page": page,
            },
        )
        payload = extract_text_content(response)
        data = json.loads(payload)
        items.extend(data.get("items", []))
        if len(items) >= sample_size:
            break
        if not data.get("items"):
            break
    return items[:sample_size]


def fetch_pr_record(session: MCPDockerSession, number: int) -> PRRecord:
    base_args = {"owner": OWNER, "repo": REPO, "pullNumber": number}

    details = json.loads(
        extract_text_content(
            session.call_tool(
                "pull_request_read",
                {**base_args, "method": "get"},
            )
        )
    )
    issue_comments = json.loads(
        extract_text_content(
            session.call_tool(
                "pull_request_read",
                {**base_args, "method": "get_comments", "perPage": 100},
            )
        )
    )
    review_comments = json.loads(
        extract_text_content(
            session.call_tool(
                "pull_request_read",
                {
                    **base_args,
                    "method": "get_review_comments",
                    "perPage": 100,
                },
            )
        )
    )

    return PRRecord(
        number=details["number"],
        title=details.get("title", ""),
        state=details.get("state", ""),
        html_url=details.get("html_url", ""),
        author=(details.get("user") or {}).get("login", "unknown"),
        created_at=parse_github_datetime(details.get("created_at")),
        closed_at=parse_github_datetime(details.get("closed_at")),
        merged_at=parse_github_datetime(details.get("merged_at")),
        issue_comments=issue_comments,
        review_comments=review_comments,
    )


def aggregate_metrics(pr_dicts: List[Dict]) -> Dict[str, float]:
    lifetimes = [item["lifetime_hours"] for item in pr_dicts]
    discussion = [item["total_comment_count"] for item in pr_dicts]
    first_responses = [
        item["first_response_hours"]
        for item in pr_dicts
        if item["first_response_hours"] is not None
    ]

    metrics = {
        "pr_count": len(pr_dicts),
        "avg_lifetime_hours": round(mean(lifetimes), 2) if lifetimes else 0.0,
        "avg_lifetime_days": round((mean(lifetimes) / 24), 2) if lifetimes else 0.0,
        "avg_comments_per_pr": round(mean(discussion), 2) if discussion else 0.0,
        "avg_first_response_hours": round(mean(first_responses), 2)
        if first_responses
        else None,
    }
    if metrics["avg_first_response_hours"] is not None:
        metrics["avg_first_response_hours"] = metrics["avg_first_response_hours"]
        metrics["avg_first_response_hours_days"] = round(
            metrics["avg_first_response_hours"] / 24, 2
        )
    return metrics


def build_summary(pr_dicts: List[Dict], metrics: Dict[str, float]) -> str:
    top_discussions = sorted(
        pr_dicts,
        key=lambda item: item["total_comment_count"],
        reverse=True,
    )[:3]

    lines = [
        f"Период анализа: последние {DAYS_BACK} дней ({metrics['pr_count']} PR).",
        f"Средняя длительность жизни PR: {metrics['avg_lifetime_days']} суток.",
        f"Средняя скорость первой реакции: "
        f"{metrics.get('avg_first_response_hours', 'n/a')} часов.",
        f"Среднее число комментариев на PR: {metrics['avg_comments_per_pr']}.",
        "",
        "PR с наибольшими обсуждениями:",
    ]
    for pr in top_discussions:
        lines.append(
            f"- #{pr['number']} ({pr['total_comment_count']} комментариев): "
            f"{pr['title']} – {pr['html_url']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PR samples via MCP")
    parser.add_argument("--days-back", type=int, default=DAYS_BACK, help="Сколько дней назад смотреть PR")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Сколько PR запросить")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь для сохранения JSON (по умолчанию data/pr_samples.json)",
    )
    args = parser.parse_args()

    start_day = datetime.now(timezone.utc).date() - timedelta(days=args.days_back)
    collected_at = datetime.now(timezone.utc)
    pr_dicts: List[Dict] = []

    with MCPDockerSession(env_file=ENV_FILE) as session:
        items = fetch_pr_items(session, start_day, sample_size=args.sample_size)[: args.sample_size]
        for item in items:
            record = fetch_pr_record(session, item["number"])
            pr_dicts.append(record.to_dict(collected_at))

    payload = {
        "owner": OWNER,
        "repo": REPO,
        "collected_at": collected_at.isoformat(),
        "period_start": start_day.isoformat(),
        "period_end": collected_at.date().isoformat(),
        "sample_size": len(pr_dicts),
        "prs": pr_dicts,
    }
    raw_path = args.output or DATA_DIR / "pr_samples.json"
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    metrics = aggregate_metrics(pr_dicts)
    summary_text = build_summary(pr_dicts, metrics)
    summary_path = PROJECT_ROOT / "analysis" / "pr_metrics_summary.md"
    summary_path.write_text(summary_text)


if __name__ == "__main__":
    main()


