#!/usr/bin/env python3
"""
Fast PR fetcher for GitHub repos using REST API with parallel requests.
Optimized for ~1000 PRs in 5-7 minutes.
"""
import os
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

session = requests.Session()
session.headers.update({
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
})


def fetch_pr_list(owner: str, repo: str, per_page: int = 100, limit: int = 1000) -> list:
    """Fetch PRs (paginated) up to limit."""
    prs = []
    page = 1
    while len(prs) < limit:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {"state": "all", "per_page": per_page, "page": page, "sort": "created", "direction": "desc"}
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 403:
            print(f"[!] Rate limited at page {page}. Collected {len(prs)} PRs so far.", flush=True)
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        prs.extend(batch)
        print(f"[PR list] page {page}: fetched {len(batch)}, total {len(prs)}", flush=True)
        page += 1
    return prs[:limit]


def fetch_pr_details(pr: dict, owner: str, repo: str) -> dict:
    """Fetch comments for a single PR and build record with comment texts."""
    num = pr["number"]
    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{num}/comments"
    review_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}/comments"

    try:
        issue_r = session.get(issue_url, params={"per_page": 100}, timeout=20)
        review_r = session.get(review_url, params={"per_page": 100}, timeout=20)
        issue_comments = issue_r.json() if issue_r.status_code == 200 else []
        review_comments = review_r.json() if review_r.status_code == 200 else []
    except Exception as e:
        print(f"[!] Error fetching comments for PR #{num}: {e}", flush=True)
        issue_comments, review_comments = [], []

    participants = set()
    author = (pr.get("user") or {}).get("login", "unknown")
    participants.add(author)
    
    # Собираем комментарии с текстами
    comments = []
    for c in issue_comments or []:
        u = (c.get("user") or {}).get("login")
        body = c.get("body", "")
        if u:
            participants.add(u)
            if body and len(body.strip()) > 0:
                comments.append({"user": u, "body": body.strip()[:1000], "type": "issue"})
    for c in review_comments or []:
        u = (c.get("user") or {}).get("login")
        body = c.get("body", "")
        if u:
            participants.add(u)
            if body and len(body.strip()) > 0:
                comments.append({"user": u, "body": body.strip()[:1000], "type": "review"})

    return {
        "number": num,
        "title": pr.get("title", ""),
        "state": pr.get("state", ""),
        "html_url": pr.get("html_url", ""),
        "author": author,
        "created_at": pr.get("created_at"),
        "closed_at": pr.get("closed_at"),
        "merged_at": pr.get("merged_at"),
        "first_response_hours": None,
        "issue_comment_count": len(issue_comments or []),
        "review_comment_count": len(review_comments or []),
        "total_comment_count": len(issue_comments or []) + len(review_comments or []),
        "participants": sorted(participants),
        "comments": comments,  # новое поле с текстами комментариев
    }


def main():
    parser = argparse.ArgumentParser(description="Fast GitHub PR fetcher")
    parser.add_argument("--owner", default="microsoft")
    parser.add_argument("--repo", default="vscode")
    parser.add_argument("--output", default="data/pr_samples_vscode.json")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers for comments")
    args = parser.parse_args()

    start = time.time()
    print(f"[*] Fetching PRs from {args.owner}/{args.repo}...", flush=True)

    prs_raw = fetch_pr_list(args.owner, args.repo)
    print(f"[*] Total PRs to process: {len(prs_raw)}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_pr_details, pr, args.owner, args.repo): pr["number"] for pr in prs_raw}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                rec = future.result()
                results.append(rec)
            except Exception as e:
                print(f"[!] Failed PR: {e}", flush=True)
            if done % 50 == 0 or done == len(prs_raw):
                elapsed = time.time() - start
                print(f"[progress] {done}/{len(prs_raw)} PRs processed ({elapsed:.1f}s)", flush=True)

    # Sort by number descending
    results.sort(key=lambda x: x["number"], reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"owner": args.owner, "repo": args.repo, "prs": results}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    elapsed = time.time() - start
    print(f"[*] Done! Collected {len(results)} PRs in {elapsed:.1f}s", flush=True)
    print(f"[*] Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()

