# Legacy v0 page generators

These scripts wrote the first four sample pages in the v0 format, where an
LLM placed every item by hand (content and coordinates together). The
pipeline now uses content-only `content.json` v1 plus
`scripts/layout_engine.py`, and `scripts/render_inkml.py` no longer reads v0
layouts. The pages' content was ported to v1 in `artifacts/page_000N/`.

They are kept for reference. Their truth-table recomputation and circuit
simulation are generalized in `scripts/validate_page.py`. Page 4's RK4
simulation of the RC derivation is not: v1 content checks the algebraic lines
numerically (`check`), and the integral and differential lines go unchecked. Run from this directory, they would write to
`scripts/artifacts/` rather than the repo's `artifacts/`.
