# V2 extension manuscript (working copy)

Working draft of the V2 journal extension. It started as an unmodified copy of the arXiv v1 source at tag `arxiv-v1`.

**Edit the manuscript here, not in the parent `paper/` folder.** `paper/main.tex`, `paper/figs/`, `paper/main.pdf` and `paper/arxiv_submission_v1.zip` are the frozen v1 record.

Structure and publication strategy, including when a separate paper would be better: [`MANUSCRIPT_PLAN.md`](MANUSCRIPT_PLAN.md).

Build from this folder (paths are relative):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The V1 PDF and arXiv zip are deliberately not copied here. V1 numbers in this draft carry the open stop-token question described in [`../../V2/benchmark_smoke_report.md`](../../V2/benchmark_smoke_report.md); resolve it before reusing them in V2 claims.
