#!/usr/bin/env python3
"""Старт на локалното HR приложение: python старт.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
QT_LIBS = ROOT / ".qt-libs"
cursor = QT_LIBS / "libxcb-cursor.so.0"
if cursor.is_file():
    extra = str(QT_LIBS)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if extra not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = extra + os.pathsep + current
        os.execv(sys.executable, [sys.executable, *sys.argv])

for path in (str(REPO), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from desktop import main

if __name__ == "__main__":
    main()
