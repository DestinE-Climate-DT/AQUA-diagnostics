#!/usr/bin/env bash
# Instantiate the skeleton diagnostic under a new name.
#
# Usage: instantiate.sh <snake_case_name> [ClassName] [repo_root]
#   ClassName defaults to the CamelCase of the name (ocean_drift -> OceanDrift).
#   repo_root defaults to the git top-level of the current directory.
#
# Copies templates/ into the repository, renaming files and replacing the tokens
# dummy -> <name>, Dummy -> <ClassName>, DUMMY -> <NAME>. Refuses to overwrite anything.
set -euo pipefail

name="${1:?usage: instantiate.sh <snake_case_name> [ClassName] [repo_root]}"
if ! [[ "$name" =~ ^[a-z][a-z0-9]*(_[a-z0-9]+)*$ ]]; then
    echo "error: '$name' is not snake_case" >&2; exit 1
fi
if [[ "$name" == "dummy" ]]; then
    echo "error: 'dummy' is the stub used by the config tests, pick another name" >&2; exit 1
fi
cls="${2:-$(echo "$name" | awk -F_ '{for (i=1;i<=NF;i++) printf toupper(substr($i,1,1)) substr($i,2)}')}"
upper="$(echo "$name" | tr '[:lower:]' '[:upper:]')"
root="${3:-$(git rev-parse --show-toplevel)}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/../templates" && pwd)"

# Collect the targets first, so that nothing is written if one already exists
targets=()
while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    targets+=("$rel")
    dest="$root/${rel//dummy/$name}"
    if [[ -e "$dest" ]]; then echo "error: $dest already exists" >&2; exit 1; fi
done < <(find "$src" -type f -print0)

for rel in "${targets[@]}"; do
    dest="$root/${rel//dummy/$name}"
    mkdir -p "$(dirname "$dest")"
    sed -e "s/Dummy/$cls/g" -e "s/DUMMY/$upper/g" -e "s/dummy/$name/g" "$src/$rel" > "$dest"
    echo "created ${dest#"$root"/}"
done
