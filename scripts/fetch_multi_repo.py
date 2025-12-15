#!/usr/bin/env python3
"""
Сбор PR с комментариями из нескольких репозиториев для дообучения.
Фильтрация: минимум 2 комментария от людей (не ботов).
Цель: 7000+ комментариев для обучения на 7 паттернов.
"""
import os, json, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
BOT_PATTERNS = ["bot", "copilot", "dependabot", "renovate", "codecov", "github-actions"]

session = requests.Session()
session.headers.update({"Accept": "application/vnd.github+json", "Authorization": f"Bearer {GITHUB_TOKEN}"})

def is_bot(username: str) -> bool:
    return any(b in username.lower() for b in BOT_PATTERNS)

def fetch_pr_list(owner: str, repo: str, limit: int = 2000) -> list:
    prs, page = [], 1
    while len(prs) < limit:
        r = session.get(f"https://api.github.com/repos/{owner}/{repo}/pulls",
                        params={"state": "all", "per_page": 100, "page": page}, timeout=30)
        if r.status_code != 200: break
        batch = r.json()
        if not batch: break
        prs.extend(batch)
        print(f"[{owner}/{repo}] page {page}: {len(prs)} PRs", flush=True)
        page += 1
    return prs[:limit]

def fetch_pr_comments(pr: dict, owner: str, repo: str) -> dict | None:
    num = pr["number"]
    try:
        issue_r = session.get(f"https://api.github.com/repos/{owner}/{repo}/issues/{num}/comments", 
                              params={"per_page": 100}, timeout=20)
        review_r = session.get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}/comments",
                               params={"per_page": 100}, timeout=20)
        issue_comments = issue_r.json() if issue_r.status_code == 200 else []
        review_comments = review_r.json() if review_r.status_code == 200 else []
    except: return None

    author = (pr.get("user") or {}).get("login", "unknown")
    comments = []
    for c in (issue_comments or []) + (review_comments or []):
        user = (c.get("user") or {}).get("login", "")
        body = (c.get("body") or "").strip()
        if user and body and not is_bot(user):
            comments.append({"user": user, "body": body[:1000]})

    # Фильтр: минимум 2 комментария от людей
    if len(comments) < 2: return None

    return {
        "repo": f"{owner}/{repo}", "number": num, "author": author,
        "created_at": pr.get("created_at"), "comments": comments
    }

def collect_repo(owner: str, repo: str, target_comments: int, workers: int = 10) -> list:
    print(f"\n[*] Collecting from {owner}/{repo}, target: {target_comments} comments", flush=True)
    prs_raw = fetch_pr_list(owner, repo, limit=3000)
    
    results, total_comments = [], 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_pr_comments, pr, owner, repo): pr["number"] for pr in prs_raw}
        for i, future in enumerate(as_completed(futures)):
            if total_comments >= target_comments: break
            rec = future.result()
            if rec:
                results.append(rec)
                total_comments += len(rec["comments"])
            if (i + 1) % 100 == 0:
                print(f"  [{owner}/{repo}] {i+1}/{len(prs_raw)} PRs, {total_comments} comments", flush=True)
    
    print(f"[*] {owner}/{repo}: {len(results)} PRs, {total_comments} comments", flush=True)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pr_samples_multi.json")
    parser.add_argument("--target", type=int, default=7000, help="Target total comments")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    repos = [("microsoft", "vscode"), ("chaoss", "augur")]
    per_repo = args.target // len(repos)

    start = time.time()
    all_prs = []
    for owner, repo in repos:
        all_prs.extend(collect_repo(owner, repo, per_repo, args.workers))

    total_comments = sum(len(pr["comments"]) for pr in all_prs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"repos": [f"{o}/{r}" for o, r in repos], "prs": all_prs}, ensure_ascii=False, indent=2))

    print(f"\n[DONE] {len(all_prs)} PRs, {total_comments} comments in {time.time()-start:.1f}s")
    print(f"Saved to {output}")

if __name__ == "__main__":
    main()

