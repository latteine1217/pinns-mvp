#!/usr/bin/env python3
"""
GitHub 搜尋與下載（clone）工具

用途：
  - 透過 GitHub CLI (`gh`) 搜尋 repositories
  - 以 `gh repo clone` 下載（支援 shallow clone）

需求：
  - 安裝 GitHub CLI：`gh`
  - 先登入：`gh auth login`

範例：
  # 搜尋 repositories（顯示前 10 筆）
  python scripts/tools/github_search_and_clone.py "physics informed neural networks" -L 10

  # 直接 clone 第 1 筆到 ./external
  python scripts/tools/github_search_and_clone.py "pinns turbulence" -L 10 --clone 1 --dest external --depth 1

  # 使用 GitHub 搜尋語法（qualifiers）
  python scripts/tools/github_search_and_clone.py "PINN language:python stars:>500" -L 20
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional


@dataclass(frozen=True)
class RepoResult:
    full_name: str
    url: Optional[str]
    description: Optional[str]
    stargazers_count: Optional[int]
    updated_at: Optional[str]


def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _require_gh() -> None:
    try:
        _run(["gh", "--version"])
    except FileNotFoundError as e:
        raise SystemExit("❌ `gh` not found. Install GitHub CLI first: https://cli.github.com/") from e


def _ensure_auth() -> None:
    proc = _run(["gh", "auth", "status"], check=False)
    if proc.returncode != 0:
        raise SystemExit(
            "❌ `gh` is not authenticated.\n"
            "Run: gh auth login\n"
            f"\nDetails:\n{proc.stderr.strip()}"
        )


def search_repos(query: str, limit: int, sort: str, order: str) -> list[RepoResult]:
    cmd = [
        "gh",
        "search",
        "repos",
        query,
        "--limit",
        str(limit),
        "--sort",
        sort,
        "--order",
        order,
        "--json",
        "fullName,url,description,stargazersCount,updatedAt",
    ]
    proc = _run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"❌ GitHub search failed:\n{proc.stderr.strip()}")

    try:
        data: List[dict[str, Any]] = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(f"❌ Failed to parse `gh search repos` JSON output: {e}") from e

    results: List[RepoResult] = []
    for item in data:
        results.append(
            RepoResult(
                full_name=item.get("fullName") or "",
                url=item.get("url"),
                description=item.get("description"),
                stargazers_count=item.get("stargazersCount"),
                updated_at=item.get("updatedAt"),
            )
        )

    results = [r for r in results if r.full_name]
    return results


def print_results(results: list[RepoResult]) -> None:
    if not results:
        print("No results.")
        return

    for idx, r in enumerate(results, start=1):
        stars = f"{r.stargazers_count:,}" if isinstance(r.stargazers_count, int) else "?"
        updated = r.updated_at or "?"
        desc = (r.description or "").strip().replace("\n", " ")
        if len(desc) > 120:
            desc = desc[:117] + "..."
        url = r.url or ""
        print(f"[{idx:>2}] {r.full_name}  ★ {stars}  updated {updated}")
        if desc:
            print(f"     {desc}")
        if url:
            print(f"     {url}")


def clone_repo(
    repo_full_name: str,
    *,
    dest_dir: Path,
    depth: Optional[int],
    branch: Optional[str],
    single_branch: bool,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    repo_name = repo_full_name.split("/", 1)[-1]
    target = dest_dir / repo_name

    cmd: List[str] = ["gh", "repo", "clone", repo_full_name, str(target)]
    git_args: List[str] = []
    if depth is not None:
        git_args += ["--depth", str(depth)]
    if branch:
        git_args += ["--branch", branch]
    if single_branch:
        git_args += ["--single-branch"]
    if git_args:
        cmd += ["--", *git_args]

    proc = _run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"❌ Clone failed:\n{proc.stderr.strip()}")

    print(f"✅ Cloned to: {target}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("query", help="GitHub search query (supports qualifiers)")
    p.add_argument("-L", "--limit", type=int, default=30, help="Max results (default: 30)")
    p.add_argument("--sort", default="best-match", choices=["best-match", "stars", "forks", "help-wanted-issues", "updated"])
    p.add_argument("--order", default="desc", choices=["asc", "desc"])
    p.add_argument("--clone", type=int, default=None, help="Clone N-th result (1-based)")
    p.add_argument("--dest", default=".", help="Destination directory (default: current directory)")
    p.add_argument("--depth", type=int, default=1, help="Git clone depth (default: 1). Use 0 for full history.")
    p.add_argument("--branch", default=None, help="Clone a specific branch")
    p.add_argument("--single-branch", action="store_true", help="Clone only the history leading to the tip of the specified branch")
    p.add_argument("--no-auth-check", action="store_true", help="Skip `gh auth status` check")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    _require_gh()
    if not args.no_auth_check:
        _ensure_auth()

    depth = None if args.depth == 0 else int(args.depth)
    dest_dir = Path(os.path.expanduser(args.dest)).resolve()

    results = search_repos(args.query, args.limit, args.sort, args.order)
    print_results(results)

    if args.clone is None:
        return 0

    if not results:
        raise SystemExit("❌ No results to clone.")

    idx = args.clone
    if idx < 1 or idx > len(results):
        raise SystemExit(f"❌ --clone must be between 1 and {len(results)} (got {idx})")

    repo = results[idx - 1]
    clone_repo(
        repo.full_name,
        dest_dir=dest_dir,
        depth=depth,
        branch=args.branch,
        single_branch=args.single_branch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
