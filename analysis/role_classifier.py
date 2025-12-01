"""Role classification based on participant activity patterns."""

from typing import Dict, List, Set

ROLE_PATTERNS = {
    "Пассивного потребления": [
        "Наблюдатель (Lurker)",
        "Пассивный пользователь (Passive user)",
        "Редкий контрибьютор (Rare contributor)",
    ],
    "Инициации обратной связи": [
        "Сообщающий ошибки (Bug reporter)",
        "Координатор (Coordinator)",
    ],
    "Периферийного участия": [
        "Периферийный разработчик (Peripheral developer)",
        "Номад-кодер (Nomad Coder)",
        "Независимый (Independent)",
    ],
    "Активного соисполнительства": [
        "Исправитель багов (Bug fixer)",
        "Активный разработчик (Active developer)",
        "Исправитель проблем (Issue fixer)",
        "Воин кода (Code Warrior)",
    ],
    "Кураторства и управления": [
        "Распорядитель проекта (Project steward)",
        "Координатор (Coordinator)",
        "Контролер прогресса (Progress controller)",
    ],
    "Лидерства и наставничества": [
        "Лидер проекта (Project leader)",
        "Основной участник (Core member)",
        "Основной разработчик (Core developer)",
    ],
    "Социального влияния": [
        "Рок-звезда проекта (Project Rockstar)",
    ],
}


def classify_participant(
    username: str,
    prs_authored: int,
    prs_reviewed: int,
    comments_count: int,
    total_prs: int,
) -> str:
    """Classify participant role based on activity metrics."""
    participation_rate = (prs_authored + prs_reviewed) / max(total_prs, 1)
    comment_rate = comments_count / max(total_prs, 1)

    if participation_rate < 0.05 and comment_rate < 0.1:
        return "Пассивного потребления"
    elif prs_authored == 0 and comments_count > 0:
        return "Инициации обратной связи"
    elif participation_rate < 0.15:
        return "Периферийного участия"
    elif participation_rate < 0.4:
        return "Активного соисполнительства"
    elif prs_reviewed > prs_authored * 2:
        return "Кураторства и управления"
    elif participation_rate > 0.4 and prs_authored > 5:
        return "Лидерства и наставничества"
    elif participation_rate > 0.6:
        return "Социального влияния"
    else:
        return "Активного соисполнительства"


def classify_all_participants(pr_data: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Classify all participants and return distribution by category."""
    participants: Dict[str, Dict[str, int]] = {}
    total_prs = len(pr_data)

    for pr in pr_data:
        author = pr.get("author", "")
        if author:
            if author not in participants:
                participants[author] = {"authored": 0, "reviewed": 0, "comments": 0}
            participants[author]["authored"] += 1

        for comment in pr.get("issue_comments", []) + pr.get("review_comments", []):
            user = comment.get("user", {}).get("login", "") if isinstance(comment, dict) else ""
            if user and user != author:
                if user not in participants:
                    participants[user] = {"authored": 0, "reviewed": 0, "comments": 0}
                participants[user]["reviewed"] += 1
                participants[user]["comments"] += 1

    distribution: Dict[str, int] = {}
    for username, stats in participants.items():
        category = classify_participant(
            username,
            stats["authored"],
            stats["reviewed"],
            stats["comments"],
            total_prs,
        )
        distribution[category] = distribution.get(category, 0) + 1

    return {"distribution": distribution, "total_participants": len(participants)}

