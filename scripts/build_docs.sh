#!/usr/bin/env bash
# Build project documentation ensuring config tables are up-to-date
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"

if [[ ! -d "$DOCS_DIR" ]]; then
  echo "Documentation directory not present; skipping docs build."
  exit 0
fi

python "$(dirname "$0")/generate_user_config_docs.py" --output "$DOCS_DIR/configuration.md" "$DOCS_DIR/AutoGPT/configuration/user_configurable.md"
mkdocs build -f "$DOCS_DIR/mkdocs.yml" "$@"
