# Build and verification notes

Prepared on 5 November 2026; certificate added on 6 November 2026.

- The supplied PDF was compiled with portable Tectonic 0.17.0 using its XeTeX engine and the BibTeX fallback of biblatex. It contains 39 pages.
- The Overleaf entry point is `main.tex`, configured for XeLaTeX and Biber, as requested by the supplied format. The local-only wrapper selects BibTeX because the portable compiler includes that backend; the wrapper is excluded from the upload ZIP.
- The Overleaf/Biber build itself was not executed in an Overleaf account. The PDF preview verifies the report content and LaTeX layout locally. Small bibliography or pagination differences can arise from the newer packages on Overleaf.
- Every included chapter and image exists. Cross-reference labels are unique; all citations have bibliography records. The final local TeX pass has no undefined references, missing-character messages or overfull boxes.
- All PDF pages were rendered and visually inspected. The architecture diagram, results table and bibliography also received enlarged inspection.
- Local compiler notices about legacy package comment encoding and per-chapter auxiliary BibTeX files do not affect the rendered document. The main bibliography completed successfully.
- The report's experimental measurements come from existing saved results. No training was run and no control, training or model files were modified while preparing the report.

The class is a supplied compatibility class, not an official institutional template. Required signed or approved front matter must come from the department.

Certificate update: inserted the user-supplied certificate wording before the abstract, added its contents entry and reused the title-page metadata. Signature and date remain blank. Recompiled and visually checked the certificate and updated contents page.
