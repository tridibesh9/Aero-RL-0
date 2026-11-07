# Figure guide

Every figure is already linked in the report. Upload the whole ZIP to Overleaf; do not upload just `main.tex`.

| File in `figures/` | Used in | Meaning |
|---|---|---|
| `learning_progress.png` | Evaluation | Training validation curves through 670k; stage goals change during training |
| `goal_generalization.png` | Evaluation | 59/60 success comparison for selected 630k and final 670k |
| `selected_comparison.png` | Evaluation | Selected model: arrival times and minimum centre distances |
| `selected_goal_errors.png` | Evaluation | Selected model: goal error versus time in all three cases |
| `case1_trajectory.png` | Evaluation | Selected model: linear obstacle, 3D trajectories |
| `case2_trajectory.png` | Evaluation | Selected model: accelerating obstacle, 3D trajectories |
| `case3_trajectory.png` | Evaluation | Selected model: figure-eight obstacle, 3D trajectories |
| `case3_separation.png` | Evaluation | Selected model: figure-eight distance histories; strongest buffer evidence |
| `case1_separation.png` | Appendix | Selected model: linear obstacle distance histories |
| `case2_separation.png` | Appendix | Selected model: accelerating obstacle distance histories |

The architecture diagram in the design chapter is drawn in TikZ and needs no image upload.

Example of the existing LaTeX link:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=.72\textwidth]{case3_trajectory.png}
  \caption{Case 3 trajectories for the selected 630,000-step checkpoint.}
  \label{fig:case3}
\end{figure}
```

`main.tex` sets `\graphicspath{{figures/}}`, so the chapter uses the filename directly.

## Provenance

- `case1_trajectory.png`: `results\evaluations\20261105T081629_168420Z\best_candidate\scenarios\case1\3d_traj.png`. validated_model.zip at 630000.
- `case1_separation.png`: `results\evaluations\20261105T081629_168420Z\best_candidate\scenarios\case1\min_dist.png`. validated_model.zip at 630000.
- `case2_trajectory.png`: `results\evaluations\20261105T081629_168420Z\best_candidate\scenarios\case2\3d_traj.png`. validated_model.zip at 630000.
- `case2_separation.png`: `results\evaluations\20261105T081629_168420Z\best_candidate\scenarios\case2\min_dist.png`. validated_model.zip at 630000.
- `case3_trajectory.png`: `results\evaluations\20261105T081629_168420Z\best_candidate\scenarios\case3\3d_traj.png`. validated_model.zip at 630000.
- `case3_separation.png`: `results\evaluations\20261105T081629_168420Z\best_candidate\scenarios\case3\min_dist.png`. validated_model.zip at 630000.
- `learning_progress.png`: `results\evaluations\20261105T081629_168420Z\learning_progress.png`. training history to 670000.
- `selected_comparison.png`: `results\evaluations\20261105T081629_168420Z`. 630000 unless explicitly labelled final 670000.
- `selected_goal_errors.png`: `results\evaluations\20261105T081629_168420Z`. 630000 unless explicitly labelled final 670000.
- `goal_generalization.png`: `results\evaluations\20261105T081629_168420Z`. 630000 unless explicitly labelled final 670000.
