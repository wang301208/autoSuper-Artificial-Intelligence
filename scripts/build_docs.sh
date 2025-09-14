#!/usr/bin/env bash
# Build project documentation ensuring config tables are up-to-date
set -euo pipefail

python "$(dirname "$0")/generate_user_config_docs.py"
mkdocs build -f docs/mkdocs.yml "$@"
