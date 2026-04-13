#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a local test environment using the interpreter pinned in .python-version.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_VER="$(cat "${ROOT_DIR}/.python-version")"
PY_BIN="${HOME}/.pyenv/versions/${PY_VER}/bin/python"

echo "[setup] repo=${ROOT_DIR}"
echo "[setup] python_version=${PY_VER}"

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[error] Python ${PY_VER} not found at ${PY_BIN}"
  echo "        Install it via: pyenv install ${PY_VER}"
  exit 1
fi

echo "[setup] using ${PY_BIN}"
"${PY_BIN}" -m pip --version

echo "[setup] upgrading pip/setuptools/wheel"
if ! "${PY_BIN}" -m pip install --upgrade pip setuptools wheel; then
  echo "[warn] base tooling upgrade failed (likely restricted network/proxy)."
fi

echo "[setup] installing project requirements"
if ! "${PY_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"; then
  echo "[warn] pip install failed (often due to restricted network/proxy)."
  echo "[warn] If running in restricted environment, provide internal wheel index or wheelhouse."
  exit 2
fi

echo "[setup] done"
