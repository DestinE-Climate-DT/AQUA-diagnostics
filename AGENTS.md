# AQUA-diagnostics Agent Instructions

Before starting work, retrieve and read both files in full, in this order:

1. Shared rules, directly from GitHub:
   [AQUA/AGENTS.common.md](https://raw.githubusercontent.com/DestinE-Climate-DT/AQUA/main/AGENTS.common.md).
2. Local instructions: [AGENTS.repo.md](AGENTS.repo.md).

AQUA's shared rules take precedence for common matters. Resolve repository-relative
paths against this AQUA-diagnostics repository. Do not load AQUA's `AGENTS.repo.md`.
No AQUA checkout is required; do not keep a local copy of `AGENTS.common.md` here.
If the shared rules cannot be retrieved or either file cannot be read in full,
report the problem and stop before making changes.

Never push directly to `main`. Merge a pull request only with the explicit human
authorization for that specific merge described in the shared rules.
