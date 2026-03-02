# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Experiment

```bash
conda activate value_capture_psychopy
cd experiment

# Local testing (windowed, fast ITIs, feedback always shown)
python main.py 99 1 1 --settings debug --force_overwrite

# Practice session (session=1, eye-tracker lab)
python main.py <subject> 1 <run>

# fMRI scanning (session≥2, waits for MRI 't' trigger)
python main.py <subject> 2 <run>
```

Key CLI flags: `--settings` (default: `default`), `--use_eyetracker` (session 1 only), `--force_overwrite`.

## Environment Setup

Two conda environments:
- `value_capture_psychopy` — runs the task (PsychoPy + exptools2)
- `value_capture_analysis` — Jupyter notebooks in `notebooks/`

```bash
bash create_env/setup.sh          # both
bash create_env/setup.sh psychopy # task only
bash create_env/setup.sh analysis # analysis only
```

**Do not install `psychtoolbox`** — it crashes on macOS. Audio uses `pygame.mixer`.

exptools2 must be installed manually if not found at `../../abstract_values/libs/exptools2`:
```bash
conda activate value_capture_psychopy
pip install -e /path/to/exptools2
```

## Architecture

The experiment is built on [exptools2](https://github.com/VU-Cog-Sci/exptools2) (PsychoPy wrapper).

```
experiment/
├── main.py       – CLI entry point; constructs ValueCaptureSession and runs it
├── session.py    – ValueCaptureSession: trial generation, bar schedule, color mapping
├── trial.py      – Trial classes: SingletonTrial, SingletonTrial_training, instruction/utility trials
├── stimuli.py    – TargetStimulusArray, PRFBarStimulus, FixationStimulus, BackgroundCircle
├── utils.py      – Output path helpers
├── instructions.yml – End-of-run/break text templates
└── settings/
    ├── default.yml  – 7T scanner (Spinoza Centre; fullscreen, scanner keys y/b)
    └── debug.yml    – Laptop testing (windowed, fast ITIs, keys a/s, feedback_probability=1.0)
```

**Data flow:** `main.py` → `ValueCaptureSession.__init__` (stimuli created) → `create_trials()` (trial list built) → `run()` (iterates trials). Trial logs land in `experiment/logs/sub-XX/ses-Y/run-Z/` (BIDS-like).

## Critical Design Constraints

- `n_trials` must be divisible by **4** (equal counts: 3 value ranks + absent) and by **20** (2 × `n_bar_positions`; default 10). Minimum valid value: 60.
- ITI lists (`iti1`, `iti2`) must evenly divide `n_trials`.
- `session=1` → `SingletonTrial_training` (feedback always, eye-movement beeps); `session≥2` → `SingletonTrial` (feedback on ~33% of trials).
- Instructions are shown only when `run == 1`.

## Value–Color Counterbalancing

`value_condition = subject % 2` controls the rank→color mapping:
- **condition 0**: rank 0 = `FULL_GREEN` (#00ab78), rank 1 = `MID_ORANGE` (#999253), rank 2 = `FULL_ORANGE` (#d56f2c)
- **condition 1**: rank 0 = `FULL_ORANGE`, rank 1 = `MID_ORANGE`, rank 2 = `FULL_GREEN`

The target and all non-distractor items are always `GREY` (#8f8f8f). Points multipliers: `[1, 10, 100]` for ranks 0–2; absent trials earn 1×.

## Trial Phase Structure

Each `SingletonTrial` has 6 phases (indices used in `draw()` and `get_events()`):

| Index | Name | Duration |
|---|---|---|
| 0 | trial_start | 0.5 s |
| 1 | pre_target | 0.5 s |
| 2 | target | 1.75 s — search array + PRF bar drawn here; response collected |
| 3 | iti1 | 1.0/1.5/2.0 s |
| 4 | feedback | 1.0 s — points text or blank |
| 5 | iti2 | 1.0/1.5/2.0 s |

**PRF bar** (`stimuli.PRFBarStimulus`) is drawn **only during phase 2**. It sits at a fixed position per trial (8 Hz flicker), with positions drawn from a balanced schedule covering all 10 horizontal × 10 vertical positions equally per run.

## Feedback Logic

`show_feedback` is pre-generated in `create_trials()` as a shuffled boolean array (exact count = `round(n_trials × feedback_probability)`), then passed explicitly to each `SingletonTrial`. In `draw()` at phase 4, the trial checks `self.parameters['show_feedback']` to decide whether to render text. `SingletonTrial_training` always shows feedback regardless.
