#!/usr/bin/env bash
set -euo pipefail

# Install pandas-ta directly from GitHub to avoid missing/broken PyPI pin resolution.
pip install "pandas-ta @ git+https://github.com/twopirllc/pandas-ta.git"

# Install vectorbt without dependency resolution so pip won't fail on pandas-ta metadata constraints.
pip install vectorbt==0.26.2 --no-deps

# Install the rest of the project dependencies.
pip install -r requirements.txt
