# Docs

Executed code lives in scripts under `docs/snippets/`, included by `pymdownx.snippets`
(`--8<-- "docs/snippets/<path>.py:name"`, delimited by `# --8<-- [start:name]`). Use
`source="material-block"`, and `html="1"` for figures. One script per session, each
runnable standalone from the repo root. Figures and tables go through
`docs/snippets/_render.py` (house style, percentile windows, scale bars, the
colour-to-CSS-variable swap); each script imports it in a hidden `plot_helpers` /
`table_helpers` section.

Five traps, none visible from the sources:

- markdown-exec does not redirect `sys.stdout`. It injects its own `print` into the
  globals of the block it executes, so only a `print` **written in the block** is
  captured; one inside an imported module resolves to the builtin, prints to the build's
  terminal, and leaves the block rendering as empty — with the build still exiting 0.
  Hence `print(figure_html(fig))` rather than a `print_figure(fig)` that prints, and
  hence the per-block duplication this replaced.

- Build with `--clean --strict` (what `build_docs` does). Plain `zensical build` exits 0
  and reports "No issues found" even when a code block raised, and serves cached HTML.
- Each page gets a fresh markdown-exec session, so a page cannot use a variable bound on
  another. Hence the silent `reopen_*` sections atop getting-started pages 2 and 3.
- Print tables with the `table_html` helper and `html="1"`, never `.to_markdown()` —
  block-level Markdown is not run over markdown-exec output, so a pipe table stays literal
  `|---|`. The helper also strips pandas' `class`/`border` attributes, which every theme
  table rule is gated against (`table:not([class])`).
- Every `.md` under `docs/` becomes a page, this file included (`/CLAUDE/`). Zensical does
  not read `exclude_docs`, and neither an underscore prefix nor burying the file under
  `snippets/` prevents it; a post-build `rm` would not help either, because `mike deploy`
  builds its own copy of the site rather than publishing `site/`. The page is absent from
  `sitemap.xml` but present in the site search index. Accepted deliberately: the split
  between this file and the root `CLAUDE.md` is worth more than the stray page.
