
from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import requests

from mcp_session import MCPDockerSession, extract_text_content

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_ENV = PROJECT_ROOT / "github-mcp" / ".env"
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "pr_metrics_summary.md"
PR_JSON_PATH = PROJECT_ROOT / "data" / "pr_samples.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "llm_commentary.txt"
OWNER = "catboost"
REPO = "catboost"
MODEL = "mistralai/mistral-tiny"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _ensure_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if name == "OPENROUTER_API_KEY" and value:
                os.environ[name] = value
                return value
    raise SystemExit("OPENROUTER_API_KEY is missing; export it or add to .env")


def _load_summary() -> str:
    if SUMMARY_PATH.exists():
        return SUMMARY_PATH.read_text().strip()
    return "Нет свежего файла pr_metrics_summary.md; запусти analysis/pr_metrics.py."


def _load_metrics_digest() -> str:
    if not PR_JSON_PATH.exists():
        return "Нет файла data/pr_samples.json; запусти analysis/pr_metrics.py."
    data = json.loads(PR_JSON_PATH.read_text())
    prs = data.get("prs", [])
    if not prs:
        return "В data/pr_samples.json нет PR записей."

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    lifetimes_days = avg([item.get("lifetime_hours", 0) for item in prs]) / 24
    first_resp = avg(
        [item["first_response_hours"] for item in prs if item.get("first_response_hours")]
    )
    comments = avg([item.get("total_comment_count", 0) for item in prs])

    lines = [
        f"Период: {data.get('period_start')} – {data.get('period_end')} (PR: {len(prs)})",
        f"Средняя длительность PR: {lifetimes_days:.1f} суток",
        f"Среднее время первой реакции: {first_resp:.1f} часов" if first_resp else "Нет данных по первой реакции",
        f"Среднее число комментариев на PR: {comments:.2f}",
        "PR с наибольшими обсуждениями:",
    ]

    top_prs = sorted(prs, key=lambda item: item.get("total_comment_count", 0), reverse=True)[:3]
    for pr in top_prs:
        lines.append(
            f"#{pr.get('number')} {pr.get('title')} — {pr.get('total_comment_count')} комментариев, состояние {pr.get('state')}"
        )
    return "\n".join(lines)


def _fetch_pr_snapshot(limit: int = 5) -> List[str]:
    since = (datetime.now(timezone.utc) - timedelta(days=120)).date().isoformat()
    query = f"repo:{OWNER}/{REPO} comments:>0 created:>={since}"
    args = {
        "query": query,
        "sort": "updated",
        "order": "desc",
        "perPage": limit,
    }
    with MCPDockerSession(env_file=MCP_ENV) as session:
        response = session.call_tool("search_pull_requests", args)
    payload = json.loads(extract_text_content(response))
    items = payload.get("items", [])[:limit]
    snapshot: List[str] = []
    for item in items:
        snapshot.append(
            textwrap.shorten(
                f"#{item.get('number')} {item.get('title')} (state={item.get('state')}, comments={item.get('comments')}, updated={item.get('updated_at')})",
                width=200,
                placeholder="…",
            )
        )
    return snapshot


def _call_llm(summary: str, digest: str, snapshot: List[str], key: str) -> str:
    snapshot_block = "\n    - ".join(snapshot) if snapshot else "данных нет"
    prompt = textwrap.dedent(
        f"""
        У тебя есть агрегированные метрики по PR репозитория {OWNER}/{REPO} и список самых обсуждаемых PR, полученный напрямую через MCP.
        Сформируй блок \"Наблюдения\" (3 пункта) и блок \"Рекомендации\" (2 пункта). Каждый пункт начинай с \"-\".
        В наблюдениях привязывайся к числам (примеры: 47 суток, 2.75 комментария) и, где возможно, упоминай конкретные PR по их номерам.
        В рекомендациях предлагай конкретные шаги по улучшению code review (ускорение реакции, перераспределение нагрузки и т.п.), не повторяй мысли и не предлагай удалять комментарии.
        Ответ на русском.

        Метрики (pr_metrics_summary.md):
        {summary}

        Числовой дайджест (data/pr_samples.json):
        {digest}

        Актуальные PR из MCP:
        - {snapshot_block}
        """
    ).strip()
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ты синьор инженер GitHub, анализируешь pull request метрики и даёшь практичные советы."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(
        OPENROUTER_URL,
        json=body,
        headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/elenagernichenko/github-analyzer",
            "X-Title": "github-analyzer",
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    key = _ensure_openrouter_key()
    summary = _load_summary()
    digest = _load_metrics_digest()
    snapshot = _fetch_pr_snapshot()
    commentary = _call_llm(summary, digest, snapshot, key)
    OUTPUT_PATH.write_text(commentary)
    print(f"Saved commentary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
