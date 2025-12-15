#!/usr/bin/env python3
"""
Fetch ~7000 PRs with comments from microsoft/vscode via GitHub REST API.
Requires env GITHUB_TOKEN.
Saves to data/pr_samples_vscode_7000.json.
"""
import os
import json
import time
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    print("[!] Set GITHUB_TOKEN env var", flush=True)

session = requests.Session()
session.headers.update({
    "Accept": "application/vnd.github+json",
    "User-Agent": "github-analyzer-fetch/1.0",
    "X-GitHub-Api-Version": "2022-11-28",
    "Authorization": f"Bearer {TOKEN}" if TOKEN else "",
})


def check_rate_limit():
    try:
        r = session.get("https://api.github.com/rate_limit", timeout=10)
        data = r.json()
        core = data.get("resources", {}).get("core", {})
        print(
            f"[rate_limit] remaining={core.get('remaining')} reset={core.get('reset')} status={r.status_code}",
            flush=True,
        )
    except Exception as e:
        print(f"[rate_limit] unable to fetch: {e}", flush=True)


def fetch_pr_list(limit: int) -> list:
    prs, page = [], 1
    check_rate_limit()
    while len(prs) < limit:
        url = "https://api.github.com/repos/microsoft/vscode/pulls"
        params = {"state": "all", "per_page": 100, "page": page, "sort": "created", "direction": "desc"}
        for attempt in range(3):
            r = session.get(url, params=params, timeout=30)
            if r.status_code >= 500:
                wait = 2 * (attempt + 1)
                print(f"[warn] {r.status_code} at page {page}, attempt {attempt+1}/3, wait {wait}s", flush=True)
                time.sleep(wait)
                continue
            break

        if r.status_code == 403:
            print(
                f"[!] 403 at page {page}, collected {len(prs)}; body={r.text[:200]}",
                flush=True,
            )
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        prs.extend(batch)
        if len(prs) >= limit:
            prs = prs[:limit]
            break
        page += 1
        print(f"[list] page {page-1}: total {len(prs)}", flush=True)
    return prs


def fetch_pr_details(pr: dict) -> dict:
    num = pr["number"]
    issue_url = f"https://api.github.com/repos/microsoft/vscode/issues/{num}/comments"
    review_url = f"https://api.github.com/repos/microsoft/vscode/pulls/{num}/comments"
    try:
        issue_r = session.get(issue_url, params={"per_page": 100}, timeout=20)
        review_r = session.get(review_url, params={"per_page": 100}, timeout=20)
        issue_comments = issue_r.json() if issue_r.status_code == 200 else []
        review_comments = review_r.json() if review_r.status_code == 200 else []
    except Exception as e:
        print(f"[!] PR #{num} error: {e}", flush=True)
        issue_comments, review_comments = [], []

    comments = []
    participants = set()
    author = (pr.get("user") or {}).get("login", "unknown")
    participants.add(author)
    for c in issue_comments + review_comments:
        u = (c.get("user") or {}).get("login", "")
        body = (c.get("body") or "").strip()
        if not u:
            continue
        participants.add(u)
        if body:
            comments.append({"user": u, "body": body[:1000]})

    return {
        "repo": "microsoft/vscode",
        "number": num,
        "author": author,
        "created_at": pr.get("created_at"),
        "comments": comments,
        "participants": sorted(participants),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=7000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/pr_samples_vscode_7000.json"))
    args = parser.parse_args()

    start = time.time()
    print(f"[*] Fetching list (target {args.limit} PRs)...", flush=True)
    prs_raw = fetch_pr_list(args.limit)
    print(f"[*] Got {len(prs_raw)} PR headers, fetching comments...", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_pr_details, pr): pr["number"] for pr in prs_raw}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if i % 100 == 0:
                print(f"[progress] {i}/{len(prs_raw)}", flush=True)

    results.sort(key=lambda x: x["number"], reverse=True)
    total_comments = sum(len(pr.get("comments", [])) for pr in results)

    payload = {"owner": "microsoft", "repo": "vscode", "prs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"[*] Saved {len(results)} PRs, {total_comments} comments -> {args.output}")
    print(f"Elapsed: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
