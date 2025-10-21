#!/usr/bin/env python3
import sys
from pathlib import Path

EXTS = ['.py', '.txt', '.md']
root = Path('.')

def is_utf8(p: Path) -> bool:
    try:
        p.read_text(encoding='utf-8')
        return True
    except UnicodeDecodeError:
        return False

def fix_file(p: Path):
    # backup
    bak = p.with_suffix(p.suffix + '.bak')
    if not bak.exists():
        p.replace(bak)
    else:
        # fallback read bak
        pass
    # read bak as cp1252 if utf-8 fails
    try:
        text = bak.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = bak.read_text(encoding='cp1252', errors='replace')
    p.write_text(text, encoding='utf-8')
    print(f"Fixed: {p} (backup: {bak})")

def main():
    changed = False
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS:
            if not is_utf8(p):
                print(f"Non-UTF8 file: {p}")
                fix_file(p)
                changed = True
    if not changed:
        print("No non-UTF8 files detected.")

if __name__ == '__main__':
    main()