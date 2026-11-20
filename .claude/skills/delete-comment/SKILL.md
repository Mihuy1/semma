---
name: delete-comment
description: Audit, categorize, and eliminate code comments (Tautology, Excuse, Stale, Decision Record), file issues for excuses, extract decision records into docs/adr/, and ratchet down .comment-baseline.json.
---

# /delete-comment

Eliminate prose code comments systematically without losing architectural knowledge or sweeping hacks under the rug.

## Workflow

### 1. Discover remaining comments
Run the comment linter to list all current prose comments (tool directives such as `#!`, `# noqa`, `# type: ignore`, `# pyright:`, `# fmt:`, `# pragma:`, `# shellcheck` are exempt):

```bash
uv run python scripts/lint_comments.py --allow-stale-baseline
```

To inspect exact comment blocks across a file or the whole repo, inspect `.comment-baseline.json` or scan with `scripts/lint_comments.py` (`scan_repo`).

### 2. Categorize every comment in context
Read the surrounding code for each comment before categorizing it into **exact one** of:

- **Tautology**: Repeats what is already obvious from the code (section banners, step labels, restating variable/test behavior).
- **Excuse**: Justifies a hacky, fragile, or incomplete implementation, workaround, timing `sleep`, private-attribute access, or tech debt that should be fixed properly.
- **Stale**: References symbols, libraries, behavior, or constraints that no longer match the current codebase.
- **Decision Record**: Documents a non-obvious architectural, protocol, algorithmic, concurrency, or platform decision ("why" the code is shaped this way) that cannot be inferred from the code alone.

If auditing a large batch of files, shard by directory and delegate categorization in parallel, then verify every `Excuse`, `Stale`, and borderline `Decision Record` against the actual source code.

### 3. Apply the action for each category
- **Tautology / Stale**: Delete the comment.
- **Excuse**: Delete the comment **and** file a GitHub issue describing the hacky implementation and the proper fix.
- **Decision Record**: Consolidate the decision into an Architecture Decision Record under `docs/adr/` (update an existing subsystem ADR or create the next numbered `docs/adr/NNNN-<topic>.md` and list it in `docs/adr/README.md`), then delete the inline comment.

### 4. Ratchet the baseline and verify
1. Update the ratchet ceiling so deleted comments can never creep back in:
   ```bash
   uv run python scripts/lint_comments.py --update-baseline
   ```
2. Verify the linter passes cleanly:
   ```bash
   uv run python scripts/lint_comments.py
   ```
3. Verify syntax and tests:
   ```bash
   QT_QPA_PLATFORM=offscreen dbus-run-session -- uv run --no-sync pytest
   ```
