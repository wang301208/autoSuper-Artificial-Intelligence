#!/usr/bin/env bash
# Build project documentation ensuring config tables are up-to-date
set -euo pipefail

python "$(dirname "$0")/generate_user_config_docs.py" --output "docs/configuration.md" "docs/AutoGPT/configuration/user_configurable.md"
mkdocs build -f docs/mkdocs.yml "$@"
