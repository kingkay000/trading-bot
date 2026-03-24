#!/usr/bin/env bash
set -euo pipefail

# Install pandas-ta from PyPI to avoid outbound git clone restrictions in Render builds.
pip install "pandas-ta==0.4.71b0"

# Install vectorbt without dependency resolution so pip won't fail on pandas-ta metadata constraints.
pip install vectorbt==0.26.2 --no-deps

# Install the rest of the project dependencies.
pip install -r requirements.txt
