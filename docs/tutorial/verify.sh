#!/bin/sh
# verify.sh — execute the tutorial chapters as notebooks, top to bottom.
#
# Each converted chapter ends in a machine-verification cell that re-derives
# the chapter's printed claims and fails loudly on drift, so "the notebooks
# ran" means "the chapters are true". Chapters are jupytext-paired MyST
# markdown: plain readable markdown in git, notebooks derived on demand —
# no Jupyter needed to *read* them, only to run this.
#
#   ./verify.sh                 # all converted chapters
#   ./verify.sh ch01_first_cell.md ...
#
# Needs: pip install jupytext nbclient ipykernel  (plus the hhg9 package)
set -e
cd "$(dirname "$0")"

chapters="$*"
if [ -z "$chapters" ]; then
    # Only chapters carrying jupytext frontmatter are executable.
    chapters=$(grep -l '^jupytext:' ch*.md)
fi

dir="`pwd`/jup"

for md in $chapters; do
    nb="${dir}/$(basename "$md" .md).ipynb"

#    nb="${TMPDIR:-/tmp}/$(basename "$md" .md).ipynb"
    jupytext --quiet --to ipynb "$md" -o "$nb"
    python3 - "$nb" <<'EOF'
import sys
import nbformat
from nbclient import NotebookClient

nb = nbformat.read(sys.argv[1], as_version=4)
NotebookClient(nb, timeout=300).execute()
print(f"{sys.argv[1].rsplit('/', 1)[-1]}: executed OK")
EOF
done
