#!/usr/bin/env python3
"""
Daily EQC dashboard entrypoint

This wraps the EQC-only console dashboard so you can run:

    python3 daily_dashboard.py [--eqc Combined_EQC.csv] [--date dd-mm-yyyy] [--projects ...]

It prints cumulative, this-month, and today counts per project using:
- Cumulative mapping: Pre=Pre+During+Post+Other; During=During+Post+Other; Post=Post+Other
- Daily (today) override: Pre=Pre, During=During, Post=Post+Other

"""
from __future__ import annotations

import sys

import build_dashboard as _dashboard


def main(argv: list[str] | None = None) -> None:
    _dashboard.main(argv or [])


if __name__ == "__main__":
    main(sys.argv[1:])
