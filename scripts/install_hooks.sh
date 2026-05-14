#!/bin/sh
# Install local git hooks from scripts/hooks/ into .git/hooks/.
#
# Git hooks are not tracked by git itself; this script copies the canonical
# tracked versions from scripts/hooks/ into the local repo's .git/hooks/.
# Run once after cloning.
#
# Usage: bash scripts/install_hooks.sh
#
# Cf. memory/DECISIONS.md ADR-006.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/scripts/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_SRC" ]; then
    echo "ERROR: $HOOKS_SRC does not exist. Are you running from the repo root?"
    exit 1
fi

for hook in pre-commit post-commit; do
    src="$HOOKS_SRC/$hook"
    dst="$HOOKS_DST/$hook"
    if [ ! -f "$src" ]; then
        echo "  skip: $hook (no source-of-truth in $HOOKS_SRC)"
        continue
    fi
    cp "$src" "$dst"
    chmod +x "$dst"
    echo "  installed: $dst"
done

echo "Done. Local hooks installed from $HOOKS_SRC."
