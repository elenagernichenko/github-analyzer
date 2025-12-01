"""Calculate PR metrics: first comment time and role distribution."""

from datetime import datetime, timezone
from statistics import mean
from typing import Dict, List

from role_classifier import classify_all_participants


def parse_datetime(value: str) -> datetime:
    """Parse GitHub datetime string."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def calculate_first_comment_time(pr_data: List[Dict]) -> float:
    """Calculate average time to first comment in hours."""
    times = []
    for pr in pr_data:
        try:
            created_at = parse_datetime(pr["created_at"])
            comments = pr.get("issue_comments", []) + pr.get("review_comments", [])
            if comments:
                comment_times = [
                    parse_datetime(c.get("created_at", ""))
                    for c in comments
                    if isinstance(c, dict) and c.get("created_at")
                ]
                if comment_times:
                    first_comment = min(comment_times)
                    hours = (first_comment - created_at).total_seconds() / 3600
                    times.append(hours)
        except (KeyError, ValueError):
            continue
    return round(mean(times), 2) if times else 0.0


def calculate_role_distribution(pr_data: List[Dict]) -> Dict[str, int]:
    """Calculate role distribution across all participants."""
    result = classify_all_participants(pr_data)
    return result["distribution"]


def calculate_metrics(pr_data: List[Dict]) -> Dict[str, any]:
    """Calculate both metrics."""
    first_comment_avg = calculate_first_comment_time(pr_data)
    role_dist = calculate_role_distribution(pr_data)
    return {
        "avg_first_comment_hours": first_comment_avg,
        "role_distribution": role_dist,
        "total_categories": len(role_dist),
    }

