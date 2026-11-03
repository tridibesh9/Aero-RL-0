# BTP thesis roadmap

Prepared: 31 October 2026

**Submission deadline: 6 November 2026. Format: LaTeX. First professor meeting: 2 or 3 November. Second meeting: before submission; target 4 or 5 November, subject to availability.**

The deadline changes the priorities: begin writing immediately, evaluate the existing system, and establish what the evidence supports. The noise extension below is optional future work unless the professor specifically requires it and the baseline and complete draft are already ready. Aim to have the final PDF ready on 5 November. Confirm the exact submission time and any signatures, printing, or binding requirements now.

The professor must confirm whether an implementation and evaluation study meets the expected BTP contribution. Continue preparing the background, method description, and baseline evidence while that decision is pending.

## 1. Agree on what the thesis will answer

Suggested working title:

**Simulation-Based Evaluation of UAV Navigation Using PPO, IMM Prediction and Adaptive Control Barrier Functions**

Main question:

> Does the implemented prediction-and-filtering architecture improve obstacle avoidance while preserving goal-reaching performance across different obstacle motions?

Optional extension, only after baseline validation and a complete thesis draft:

> How sensitive is that performance to noise in obstacle-position measurements?

Keep one drone and one obstacle initially. Treat additional drones, real hardware, and new learning algorithms as separate scope decisions with the professor.

**Deliverable:** A short approved problem statement, objectives, scope, contribution, and experiment list.

## 2. Learn the system in a useful order

Study the concepts below just deeply enough to explain their inputs, outputs, purpose, and limitations before attempting every derivation.

| Order | Concept | Question you should be able to answer | Repository entry point |
|---|---|---|---|
| 1 | Environment, observations, actions, rewards | What does the agent observe, command, and get rewarded for? | `envs/UAVNavEnv.py` |
| 2 | PPO | What is learned during training and what happens during inference? | `train_ppo.py` |
| 3 | Motion models and Kalman filtering | How can repeated position measurements estimate velocity and future position? | `controllers/imm_predictor.py` |
| 4 | IMM | Why combine constant-velocity, constant-acceleration, and turning models? | `controllers/imm_predictor.py` |
| 5 | CBF, adaptive coefficient, and optimization | How does the filter modify a proposed command, and under what assumptions? | `controllers/acbf_filter.py`, `utils/math_utils.py` |
| 6 | Full experiment | How are controllers compared, and what is actually measured? | `controllers/hierarchical_ctrl.py`, `evaluate.py`, `utils/logger.py` |

Build a glossary for PPO, PID, IMM, CV, CA, CT, CBF, ACBF, CLF, state, action, reward, prediction horizon, and constraint slack.

Read the base paper in this order: abstract and conclusion, architecture figure, experiments, then method equations. Annotate each equation with the corresponding implementation if one exists.

**Deliverable:** A one-page architecture explanation and a small literature table. For each relevant paper, record its problem, method, evaluation, limitations, and relation to your work. A starting target is 5-8 closely relevant papers, subject to departmental expectations; this is not a fixed thesis requirement.

Base paper: https://www.sciltp.com/journals/am/articles/2607004775

## 3. Establish a trustworthy baseline

Preserve the current model and saved outputs before running experiments. The existing evaluation script writes into `results/eval/` and can replace files there. The training script also uses fixed output locations.

Before interpreting a full benchmark:

1. Record the code revision, package versions, model file, configuration, random seed, and scenario definition for each experiment.
2. Assess the acceleration-versus-PID-waypoint mismatch and the optimizer-versus-execution limit mismatch. Prioritize bounded corrections that can be checked within the deadline. A change in action meaning may require retraining; it should not be silently applied to an existing trained policy. Record any unresolved mismatch as a limitation and restrict claims to the evaluated implementation.
3. Define collision, desired clearance, barrier satisfaction, goal success, and timeout separately. The current barrier zero occurs at about 0.707 m for `ds=0.5`, while the plotted threshold is 0.5 m and the logger collision threshold is 0.4 m.
4. Review relative-motion sign conventions in the adaptive coefficient and document the chosen interpretation.
5. Improve logging to include the terminal state, termination reason, actual executed command, solver status, and any constraint relaxation.
6. Make random resets reproducible across episodes. The current reset implementation creates a fresh generator from `seed`, including when no seed is supplied.
7. Execute all three controllers on all three cases and record the outputs in separate run directories.

The current ACBF implementation permits constraint slack and can return the original command on solver failure. Describe measured safety performance and failure conditions; do not transfer the paper's formal guarantee to this implementation without establishing its assumptions.

**Deliverable:** A reproducible baseline report with nine controller-scenario combinations, clear definitions, and traceable artifacts.

## 4. Design the experiments before collecting the final results

Suggested measurements:

| Measurement | Why it matters |
|---|---|
| Goal-reaching success rate | Avoidance is not enough if the drone never completes the mission. |
| Collision rate | Measures failure using a documented geometric or simulator-contact definition. |
| Desired-clearance violation rate | Distinguishes entering the safety buffer from physical contact. |
| Minimum centre distance and minimum surface clearance | Removes ambiguity about what a reported distance means. |
| Completion time among successful episodes | Measures efficiency; report timeouts separately. |
| Path length | Quantifies detours. |
| Filter intervention magnitude | Measures how much the filter changes the reference command. |
| Solver failure and constraint-relaxation statistics | Shows whether the safety optimization is being satisfied. |
| Median and 95th-percentile control computation time | Assesses whether computation fits the configured 50 ms control period on the test machine. |

Use identical scenario definitions and paired random seeds when comparing controllers. Repeating an entirely deterministic episode with the same initial conditions produces the same trajectory, not evidence of robustness. Introduce documented variation, such as initial positions, obstacle speed, or measurement-noise realizations.

Under the 6 November deadline, first complete and inspect the nine controller-scenario combinations. Then run a small, explicitly defined set of varied trials if time permits; record the sample count and do not imply that a small sample establishes robust safety. Single deterministic runs should be reported as individual case studies, not population success-rate estimates. When comparing learned policies, evaluation seeds measure environment variation; multiple independent training seeds are needed to assess training variability. A one-model evaluation must state that limitation.

Report counts as well as percentages. Zero collisions in a finite experiment is an observed result, not a proof of zero collision probability.

**Deliverable:** An experiment matrix, metric definitions, and a fixed evaluation procedure agreed before the final runs.

## 5. Optional extension after the core submission is ready

Candidate extension: **obstacle-position measurement noise**. Given the confirmed deadline, keep this in future work unless the professor requires an extension and time remains after baseline evaluation and the full draft. Do not make it a prerequisite for writing or completing the baseline study.

1. Retain exact simulated obstacle position as ground truth for distance and collision measurements.
2. Give the estimator a noisy position observation, with a documented noise model and seed.
3. Use a clean-observation baseline and two explicitly chosen noise levels. Choose levels with the professor or justify them as a sensitivity study; do not claim realism without sensor evidence.
4. Keep the PPO model, scenarios, and controller settings fixed for the first comparison. Document estimator-noise assumptions.
5. Compare all three controller configurations across all three obstacle cases.
6. Optionally compare IMM with a constant-velocity predictor to isolate the value of the predictor, if time permits.

This yields a focused study of when prediction and filtering help, when they degrade, and why. If the professor requires an algorithmic contribution, agree on the specific method change before committing to this as the entire thesis.

**Deliverable:** Robustness plots and a supported answer to the approved research question.

## 6. Write while the experiments are being prepared

The confirmed submission format is **LaTeX**. Obtain the department's existing template from the professor, department, or a recent student now. Until it arrives, draft chapter content in separate `.tex` files so it can be inserted into the required template. Keep citations in a `.bib` file and experiment figures in a dedicated figures directory. Compile from the first day to catch missing references and layout problems early.

A provisional chapter structure is:

1. **Introduction:** problem, motivation, research questions, objectives, scope, and contributions.
2. **Background and related work:** UAV navigation, PPO, state estimation, IMM, CBFs, and the limitations motivating the study.
3. **System and methodology:** environment, observation/action definitions, reward, controller architecture, assumptions, and equations.
4. **Implementation and experimental design:** repository structure, corrections, configuration, baselines, scenarios, seeds, and metric definitions.
5. **Results and discussion:** baseline comparisons, extension experiments, variability, failure cases, and explanations supported by evidence.
6. **Conclusion and future work:** answers to the research questions, limitations, and justified next steps.

Then references and appendices as required. Front matter, certificates, declarations, page counts, and citation style are department-specific.

Start the introduction, architecture explanation, and experiment procedure now. Write final contributions, conclusions, and abstract after the evidence is settled. Clearly distinguish methods taken from the paper, inherited implementation, your corrections, and your own experiments.

## 7. Deadline schedule: 31 October to 6 November 2026

| Date | Priority | Deliverable by the end of the day |
|---|---|---|
| 31 October | Obtain the LaTeX template; set up the thesis outline; preserve current artifacts; write the problem, objectives, and system explanation | Compiling thesis outline with initial prose, architecture diagram, agreed list of validation tasks |
| 1 November | Check measurement and command-path issues; run the baseline comparisons; draft the methodology and six meeting slides | Recorded runs or a precise failure report, initial plots, methods draft, meeting slides |
| 2 November | Prefer the first professor meeting if available; confirm contribution and required fixes; continue experiments and writing | Scope decision if the meeting occurs, initial results discussion, references |
| 3 November | Hold the first meeting if not held on 2 November; finish required core experiments and the complete first draft | Readable end-to-end thesis draft, including limitations and preliminary conclusion |
| 4 November | Verify every numerical claim and figure; target the second professor review | Review PDF with traceable final results and a short list of decisions needed |
| 5 November | Incorporate feedback; hold the second meeting if necessary; check formatting, citations, front matter, and submission requirements | Final PDF, LaTeX source backup, required signatures or other paperwork |
| 6 November | Submit before the exact institutional cutoff; keep time for upload or printing problems | Submitted thesis and submission acknowledgement |

If the first meeting is on 3 November, continue the writing and baseline work on 2 November. The meeting should refine the draft and scope, not mark the start of writing. Stop optional experiments when they threaten completing or checking the thesis.

## 8. Minimum evidence package for the submission

- A clearly attributed description of the source method and the actual implemented architecture.
- Three controllers compared across the three existing scenarios, with run configuration and model identity recorded; any incomplete experiment explicitly identified.
- Correctly defined goal completion, collisions, clearance, and episode termination, with terminal states included in measurement.
- Readable trajectory and separation plots and a comparison table derived from the documented runs.
- A discussion of both successful and unsuccessful outcomes, implementation mismatches, and the limits of the evidence.
- A complete LaTeX thesis in the department's format, with valid references and required front matter.

This is a proposed submission scope for the professor to assess, not a guarantee that it satisfies departmental requirements. Poor controller performance is a result to explain honestly; it should not be hidden or replaced with paper values.

## What to do first

1. Read `PROFESSOR_MEETING_BRIEF.md` and practise its opening explanation aloud.
2. Turn its outline into six simple slides if a presentation is expected.
3. Obtain the LaTeX template and confirm the exact submission cutoff, simulation-only scope, and contribution expected by the professor.
4. Start writing and baseline validation immediately. Launch a new long training run only if a necessary correction requires it and there is enough time to evaluate and document the new model.
