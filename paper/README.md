# ERRORFlow paper draft

`errorflow_v3.tex` is the full LNCS research manuscript generated from the
frozen V3 results through WP5. Author, affiliation, and funding fields remain
anonymous placeholders rather than invented metadata.

The user-provided LNCS package remains under `Template-Paper/` and is not
tracked or modified by this draft. To compile locally from `paper/`, make the
class and bibliography style visible through `TEXINPUTS`, or copy the draft
into the template directory temporarily:

```bash
cd paper
TEXINPUTS=../Template-Paper: pdflatex errorflow_v3.tex
```

The repository includes a verified bibliography, a frozen numeric snapshot,
vector-figure generator, claim ledger, and reproducibility guide. Raw V3
prediction artifacts remain server-side because `outputs/` is ignored.

Before submission:

1. replace author and affiliation placeholders;
2. reconcile page limits with the target venue;
3. export a checksum-addressed server artifact archive;
4. pin model/tokenizer revisions and the Python environment.

## Draft build checkpoint

On 2026-10-13, the manuscript was compiled with the user-provided LNCS 2026
class into `output/pdf/errorflow_v3_draft.pdf`.

- build status: successful, two `pdflatex` passes;
- rendered length: 5 pages;
- LaTeX log: no overfull/underfull box or unresolved-reference warnings;
- visual QA: all 5 rendered pages inspected; no clipping, overlap, or table
  overflow observed;
- scope: full anonymous research draft; author metadata and external artifact
  release remain pending.
