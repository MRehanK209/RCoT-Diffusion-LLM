#!/usr/bin/env bash
# Verifies MacTeX CLI, then runs `make` in the thesis project root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(/usr/libexec/path_helper 2>/dev/null)" || true
export PATH="/Library/TeX/texbin:${PATH}"

if command -v latexmk >/dev/null 2>&1 && [[ -x "$(command -v latexmk)" ]]; then
  echo "TeX OK: $(command -v latexmk)"
  cd "$ROOT"
  make
  echo "Build finished. PDF: $ROOT/thesis.pdf"
  exit 0
fi

echo "TeX CLI not found (expected /Library/TeX/texbin after MacTeX install)."
PKG="$(ls /opt/homebrew/Caskroom/mactex/*/mactex-*.pkg 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "${PKG}" && -f "${PKG}" ]]; then
  echo "Opening MacTeX installer (finish all steps, then run this script again):"
  echo "  ${PKG}"
  open "${PKG}"
  echo ""
  echo "Or install from Terminal (admin password):"
  echo "  sudo installer -pkg \"${PKG}\" -target /"
  exit 1
fi

echo "No MacTeX .pkg found. Run: brew install --cask mactex"
exit 1
