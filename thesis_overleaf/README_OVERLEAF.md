# Overleaf setup

This is a complete report draft, not only a template. It uses your supplied chapter organization and author-year bibliography style. The title page contains Tridibesh Sarkar, 23AE30023, Aerospace Engineering, Dual Degree, IIT Kharagpur, and Prof. Priyam Chakraborty.

## Upload and compile

1. Download `BTP_Report_Overleaf.zip` from the delivered files.
2. In Overleaf select **New Project > Upload Project** and upload the ZIP.
3. Open **Menu / Settings** and select `main.tex` as the main document.
4. Select **XeLaTeX** as the compiler. Overleaf runs Biber for the bibliography automatically.
5. Press **Recompile**. If references initially show question marks, use **Recompile from scratch** and allow the bibliography passes to finish.

All ten external figure files are already included in `figures/` and linked in the chapters. No local Windows paths or external image URLs are used by the LaTeX document. The architecture diagram is generated directly by TikZ.

## Files to edit

| File | What it controls |
|---|---|
| `metadata.tex` | Working title, name, roll number, degree, supervisor, institute and date |
| `main.tex` | Packages, bibliography style and chapter order |
| `template.cls` | Page layout and title page |
| `intro/abstract.tex` | Abstract |
| `intro/certificate.tex` | Certificate before the abstract, with blank signature and date |
| `intro/introduction.tex` | Motivation, objectives and contribution |
| `background/background.tex` | PPO, barrier functions, prediction and related work |
| `design/design.tex` | Actual observation, action, reward and filter equations |
| `impl/implementation.tex` | Training configuration, curriculum, scenarios and metrics |
| `eval/evaluation.tex` | Results, model comparison, failures and limitations |
| `conclude/conclude.tex` | Conclusions and future work |
| `appendix/appendix.tex` | Reproduction details, versions, checkpoint hashes and supplementary figures |
| `thesis.bib` | References |

## Template compatibility

The pasted preamble referred to `\documentclass{template}`, but the original `template.cls` was not provided. The included class is a newly supplied compatibility class based on the standard `report` class. It supports the metadata commands used in your pasted format. **It is not an official IIT Kharagpur thesis class.** Confirm departmental margins, title wording, front-matter requirements and declaration/certificate pages before submission.

The certificate uses the wording you supplied and leaves the supervisor signature and date blank. Add any other mandatory institutional front matter using the department's approved wording. The supplied title is a working title and can be changed in `metadata.tex`.

If you later obtain the original class, preserve this working project first. Its title-page or front-matter commands may differ; replacing a class file blindly can change or break those pages.

## Results used

- Training run: `20261105T022305_489044Z`, seed 42, eight environments, requested ceiling two million aggregate transitions.
- Training stopped at 670,000 transitions after the configured repeated validation successes.
- **Main results and all three scenario trajectory figures:** selected/validated checkpoint at **630,000** transitions. `best_candidate.zip` and `validated_model.zip` are identical for this run.
- **Separate comparison table and additional-goal chart:** final checkpoint at **670,000** transitions.
- Both reached 59/60 additional goals. Do not call this universal 98.3% reliability.
- Pure RL entered the figure-8 safety buffer but did not cross the collision threshold. The filtered policies preserved the observed buffer and reached the goal.

Do not replace individual figures with plots from the older 340k, 507,904-step or 150,528-step models without updating the captions, tables and conclusions together.

## Figures

See `FIGURE_GUIDE.md` for every figure filename, the chapter where it is used, and its source. `figure_sources.json` retains the source paths and copied-image hashes. The main results tables are generated from saved JSON measurements, not manually invented values.

## Before submission

- Confirm the working title and degree/report wording with your supervisor.
- Read the report and verify that you can explain the equations, implementation adaptations and limitations.
- Confirm any departmental declaration, certificate, acknowledgement and AI-assistance disclosure requirements and supply accurate text if required.
- Retain the experiment archive and code that support the reported results.
- Check the final Overleaf PDF after editing metadata or figures, especially the table of contents, references and page breaks.

The report describes a paper-inspired implementation and evaluation. It does not claim a new safety theorem, an exact reproduction of all paper experiments, global convergence, superiority over PID, or hardware validation.
