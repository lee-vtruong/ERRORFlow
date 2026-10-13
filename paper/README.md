# ERRORFlow paper draft

`errorflow_v3.tex` is an LNCS manuscript draft generated from the frozen V3
results through WP4. It intentionally leaves authors, affiliations, funding,
and references as explicit TODOs rather than inventing them.

The user-provided LNCS package remains under `Template-Paper/` and is not
tracked or modified by this draft. To compile locally from `paper/`, make the
class and bibliography style visible through `TEXINPUTS`, or copy the draft
into the template directory temporarily:

```bash
cd paper
TEXINPUTS=../Template-Paper: pdflatex errorflow_v3.tex
```

The bibliography will remain unresolved until a verified `references.bib` is
added. External result PDFs are generated on the server under
`outputs/v3/figures/external/`; matched-budget PDFs are under
`outputs/v3/figures/matched_budget/`.

Before submission:

1. replace author and affiliation placeholders;
2. add and verify primary-source references;
3. copy selected vector PDFs into the submission bundle and add figure calls;
4. reconcile page limits with the target venue;
5. run LaTeX compilation and inspect the rendered PDF page by page.

## Draft build checkpoint

On 2026-10-13, the manuscript was compiled with the user-provided LNCS 2026
class into `output/pdf/errorflow_v3_draft.pdf`.

- build status: successful, two `pdflatex` passes;
- rendered length: 5 pages;
- LaTeX log: no overfull/underfull box or unresolved-reference warnings;
- visual QA: all 5 rendered pages inspected; no clipping, overlap, or table
  overflow observed;
- scope: research draft only, because verified references, author metadata,
  and selected result figures remain pending.
