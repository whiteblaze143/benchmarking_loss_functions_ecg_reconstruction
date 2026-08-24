#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_WORKTREE = PROJECT_ROOT.parent / "benchmarking_loss_functions_ecg_reconstruction_site"
BOOK_OUTPUT_DIR = PROJECT_ROOT / "book" / "_book"

def main():
    print("=== GitHub Pages Deployment Script ===")
    
    # 1. Check if site worktree exists
    if not SITE_WORKTREE.exists():
        print(f"Error: Site worktree at {SITE_WORKTREE} does not exist.")
        print("Run: git worktree add ../benchmarking_loss_functions_ecg_reconstruction_site gh-pages")
        sys.exit(1)

    # 2. Render Quarto book if needed
    if "--skip-render" not in sys.argv:
        print("Rendering Quarto book...")
        res = subprocess.run(["quarto", "render", "book/"], cwd=PROJECT_ROOT)
        if res.returncode != 0:
            print("Error: Quarto render failed.")
            sys.exit(1)
            
    if not BOOK_OUTPUT_DIR.exists():
        print(f"Error: Output directory {BOOK_OUTPUT_DIR} does not exist.")
        sys.exit(1)

    # 3. Copy rendered files to gh-pages worktree
    print(f"Syncing rendered files from {BOOK_OUTPUT_DIR} to {SITE_WORKTREE}...")
    for item in SITE_WORKTREE.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for item in BOOK_OUTPUT_DIR.iterdir():
        target = SITE_WORKTREE / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy(item, target)

    # Add .nojekyll to ensure GitHub Pages serves underscore directories
    (SITE_WORKTREE / ".nojekyll").touch()

    print("Checking git status in site worktree...")
    subprocess.run(["git", "status", "-s"], cwd=SITE_WORKTREE)

    if "--push" in sys.argv:
        print("Committing and pushing to gh-pages...")
        subprocess.run(["git", "add", "."], cwd=SITE_WORKTREE)
        subprocess.run(["git", "commit", "-m", "chore: update GitHub Pages site build"], cwd=SITE_WORKTREE)
        subprocess.run(["git", "push", "origin", "gh-pages"], cwd=SITE_WORKTREE)
        print("Deployment complete!")
    else:
        print("\nSync complete! To commit and push, run:")
        print("python3 scripts/deploy_gh_pages.py --push")

if __name__ == "__main__":
    main()
