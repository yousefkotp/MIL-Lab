#!/usr/bin/env python3
"""
delete_pt.py — safely find & delete *.pt files under a target directory

Usage examples:
  python delete_pt.py                                  # dry run (no deletion)
  python delete_pt.py --force                          # actually delete
  python delete_pt.py --path /other/dir                # change target (dry run)
  python delete_pt.py --path /other/dir --force
  python delete_pt.py --older-than 7                   # only files older than 7 days (dry run)
  python delete_pt.py --older-than 7 --force
  python delete_pt.py --keep-pattern 'best|final'      # keep files whose name matches regex
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

DEFAULT_TARGET = Path("/home/mila/k/kotpy/scratch/MIL-Lab/results")
SAFE_PREFIX = Path("/home/mila/k/kotpy/scratch/MIL-Lab/results")  # edit if needed


def human_size(nbytes: int) -> str:
    units = ["B", "K", "M", "G", "T", "P"]
    size = float(nbytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.1f}{u}"
        size /= 1024.0
    return f"{nbytes}B"


def iter_pt_files(root: Path, older_than_days: int | None, keep_re: re.Pattern | None,
                  follow_symlinks: bool = False) -> Iterator[Path]:
    now = time.time()
    cutoff = None if older_than_days is None else now - older_than_days * 24 * 3600

    # Walk
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        d = Path(dirpath)
        for name in filenames:
            if not name.endswith(".pt"):
                continue
            p = d / name

            # Skip symlinked files unless following
            if p.is_symlink() and not follow_symlinks:
                continue

            # Age filter
            if cutoff is not None:
                try:
                    if p.stat().st_mtime > cutoff:
                        continue
                except FileNotFoundError:
                    continue

            # Keep pattern (do not delete if it matches)
            if keep_re is not None and keep_re.search(name):
                continue

            yield p


def scan_targets(root: Path, older_than_days: int | None, keep_pattern: str | None,
                 follow_symlinks: bool) -> Tuple[List[Path], int]:
    keep_re = re.compile(keep_pattern) if keep_pattern else None
    files = list(iter_pt_files(root, older_than_days, keep_re, follow_symlinks))
    total_bytes = 0
    for p in files:
        try:
            total_bytes += p.stat().st_size
        except FileNotFoundError:
            pass
    return files, total_bytes


def ensure_safe_prefix(target: Path, safe_prefix: Path) -> None:
    try:
        target_resolved = target.resolve()
        safe_resolved = safe_prefix.resolve()
    except Exception:
        target_resolved = target
        safe_resolved = safe_prefix
    if safe_resolved not in target_resolved.parents and target_resolved != safe_resolved:
        print("ERROR: Refusing to operate outside safe prefix.", file=sys.stderr)
        print(f"  Target: {target_resolved}", file=sys.stderr)
        print(f"  Prefix: {safe_resolved}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Safely delete *.pt files under a directory (dry-run by default).")
    ap.add_argument("--path", type=Path, default=DEFAULT_TARGET, help="Target directory to scan.")
    ap.add_argument("--force", action="store_true", help="Actually delete files (default is dry-run).")
    ap.add_argument("--older-than", type=int, default=None,
                    help="Only delete files older than N days.")
    ap.add_argument("--keep-pattern", type=str, default=None,
                    help="Regex for filenames to KEEP (e.g., 'best|final').")
    ap.add_argument("--follow-symlinks", action="store_true",
                    help="Follow symlinks while walking (off by default).")
    ap.add_argument("--show", type=int, default=20, help="Show first N matches in dry-run (default 20).")
    args = ap.parse_args()

    target = args.path
    if not target.exists() or not target.is_dir():
        print(f"ERROR: Target directory does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    ensure_safe_prefix(target, SAFE_PREFIX)

    files, total_bytes = scan_targets(
        target, args.older_than, args.keep_pattern, args.follow_symlinks
    )

    if not files:
        print(f"No matching *.pt files found under: {target}")
        if args.older_than is not None:
            print(f"(Filtered to files older than {args.older_than} day(s))")
        if args.keep_pattern:
            print(f"(Keeping files matching regex: {args.keep_pattern})")
        sys.exit(0)

    print(f"Scanning: {target}")
    if args.older_than is not None:
        print(f"  Filter: older than {args.older_than} day(s)")
    if args.keep_pattern:
        print(f"  Keep regex: {args.keep_pattern}")
    print(f"Found {len(files)} file(s), total ~ {human_size(total_bytes)}")

    if not args.force:
        print("\nDRY RUN (no files will be deleted). Showing first "
              f"{min(args.show, len(files))} matches:")
        for p in files[: args.show]:
            print(p)
        if len(files) > args.show:
            print(f"... (and {len(files) - args.show} more)")
        print("\nTo actually delete, re-run with --force")
        sys.exit(0)

    # Deletion phase
    errors = 0
    for p in files:
        try:
            p.unlink()
            print(f"deleted: {p}")
        except Exception as e:
            errors += 1
            print(f"FAILED to delete: {p}  ({e})", file=sys.stderr)

    print(f"\nDone. Deleted {len(files) - errors} file(s)"
          f"{' with errors' if errors else ''}. Freed ~ {human_size(total_bytes)} (approx).")
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
