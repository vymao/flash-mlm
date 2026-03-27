#!/usr/bin/env bash
set -euo pipefail

poetry run pytest src/flash_mlm/host/test_host_utils.py src/flash_mlm/host/test_host.py

if [ $# -ge 1 ]; then
    poetry version "$1"
else
    poetry version patch
fi

VERSION=$(poetry version -s)

git add -A
git commit -m "v${VERSION}: release"
git tag "v${VERSION}"
git push origin main --tags
