# Value Capture — fMRI Experiment

A PsychoPy/exptools2 experiment for studying value-driven attentional capture in the MRI scanner. Participants search for a uniquely-oriented bar among distractors and report whether it contains a dot. On most trials a colour singleton distractor signals the reward value of that trial, allowing us to study how reward history biases spatial attention.

A flickering checkerboard bar (PRF stimulus) is presented simultaneously with the search array, enabling population receptive field (PRF) mapping from the same fMRI data.

## Task Design

Each trial proceeds through the following phases:

| Phase | Duration | Description |
|---|---|---|
| Trial start | 0.5 s | Fixation only |
| Pre-target | 0.5 s | Fixation only (cue period) |
| Target | 1.75 s | Search array + PRF bar visible; response window |
| ITI 1 | 1.0 / 1.5 / 2.0 s | Fixation |
| Feedback | 1.0 s | Points earned (33% of scanning trials; 100% in practice) |
| ITI 2 | 1.0 / 1.5 / 2.0 s | Fixation |

### Search display

Eight bars are arranged in a circle at 4° eccentricity. One bar (the **target**) is oriented uniquely; the rest share the opposite orientation. Participants press one of two keys to report whether the target contains a small dot.

On 75% of trials one bar has a distinctive colour (the **distractor**). Its colour signals the points multiplier for that trial:

| Colour | Multiplier |
|---|---|
| Full green `#00ab78` | × 1 (low) |
| Olive/khaki `#999253` | × 10 (medium) |
| Full orange `#d56f2c` | × 100 (high) |

The green–orange assignment is counterbalanced across subjects by parity (`subject % 2`). The three colours are equally spaced in CIELAB (ΔE ≈ 46.5 to each endpoint).

On 25% of trials no distractor is present (absent condition); those trials earn 1× points.

### Points

```
base_points = round((1 − RT / 1.75) × 100)   # 0–100, clipped at 0
earned_points = base_points × multiplier
```

Feedback messages:
- **+N** — correct response, coloured in the distractor colour
- **Incorrect!** — wrong key pressed
- **Too late!** — no response within 1.75 s

### PRF bar

A flickering (8 Hz) checkerboard bar sits at a fixed position during the target phase. Positions are drawn from a balanced schedule: 10 horizontal × 10 vertical = 20 positions, each visited 3 times per 60-trial run. Bar and aperture are circular, centred at fixation.

### Trial counts (40 trials per run)

| Condition | N |
|---|---|
| Low value (× 1) | 10 |
| Medium value (× 10) | 10 |
| High value (× 100) | 10 |
| Distractor absent | 10 |

### Run duration

A single run lasts exactly **5 min 30 s (330 s)**. The duration is deterministic: ITIs are tiled rather than independently sampled.

| Component | Duration |
|---|---|
| Leading fixation baseline | 20 s |
| Trials 1–13 (6.75 s avg each) | 87.75 s |
| Rest pause | 10 s |
| Trials 14–26 (6.75 s avg each) | 94.5 s |
| Rest pause | 10 s |
| Trials 27–40 (6.75 s avg each) | 87.75 s |
| Trailing fixation baseline | 20 s |
| **Total** | **330 s (5 min 30 s)** |

Per-trial breakdown: 0.5 (trial start) + 0.5 (pre-target) + 1.75 (target) + 1.5 (ITI 1, mean) + 1.0 (feedback) + 1.5 (ITI 2, mean) = 6.75 s. ITIs are right-skewed: `[0.5, 0.75, 1.0, 1.0, 1.25, 1.5, 2.0, 4.0]` s (mean = 1.5 s, median = 1.125 s).

A full scanning session of 10 runs is **3300 s (55 min)** of stimulus time, not counting inter-run setup.

## Sessions

| Session | Description |
|---|---|
| 1 | Practice in the eye-tracker lab. Starts on key press. All trials use `SingletonTrial_training` which beeps on excessive eye movements. Feedback shown on every trial. |
| 2+ | fMRI scanning. Waits for the MRI `t` trigger (press twice: once to show a blank, once to start trials). Feedback shown on ~33% of trials. |

Instructions are shown only on run 1 of session 1 (practice). Scanning sessions (session ≥ 2) never show instructions.

## Repository Layout

```
value_capture/
├── experiment/
│   ├── main.py          # entry point
│   ├── session.py       # ValueCaptureSession
│   ├── trial.py         # trial classes
│   ├── stimuli.py       # stimulus classes (bars, fixation, PRF bar)
│   ├── utils.py         # output path helpers
│   ├── instructions.yml # end-of-run / break text
│   ├── beep.wav         # eye-movement alert sound
│   ├── settings/
│   │   ├── default.yml  # 7T scanner (Spinoza Centre)
│   │   └── debug.yml    # laptop testing (windowed, short ITIs)
│   └── logs/            # output (gitignored)
├── create_env/
│   ├── environment_psychopy.yml  # conda env for running the task
│   ├── environment_analysis.yml  # conda env for data analysis
│   └── setup.sh                  # creates both environments
├── notebooks/           # analysis notebooks
└── value_levels_2.osexp # original online OpenSesame experiment
```

## Installation

### Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [exptools2](https://github.com/VU-Cog-Sci/exptools2) (installed separately; see below)

### Create environments

```bash
# Both environments at once
bash create_env/setup.sh

# Or individually
bash create_env/setup.sh psychopy   # experiment only
bash create_env/setup.sh analysis   # analysis only
```

If exptools2 is not found automatically, install it manually:

```bash
conda activate value_capture_psychopy
pip install -e /path/to/exptools2
```

> **Note:** Do not install `psychtoolbox` — it crashes on macOS. Audio is handled via `pygame.mixer` instead.

## Running the Experiment

```bash
conda activate value_capture_psychopy
cd experiment
python main.py <subject> <session> <run> [--settings SETTINGS] [--use_eyetracker] [--force_overwrite]
```

| Argument | Description |
|---|---|
| `subject` | Participant number (integer) |
| `session` | 1 = practice, 2+ = scanning |
| `run` | Run number within session |
| `--settings` | Settings file name without `.yml` (default: `default`) |
| `--use_eyetracker` | Enable Eyelink eye-tracker (session 1 only) |
| `--force_overwrite` | Skip the existing-output-directory check (useful for testing) |

### Examples

```bash
# Practice session, run 1, default settings
python main.py 1 1 1

# MRI session 2, run 3
python main.py 1 2 3

# Local testing with debug settings
python main.py 99 1 1 --settings debug --force_overwrite
```

### Running a full scanning session (Windows / MRI scanner)

Use the PowerShell script to run all runs of a session back-to-back:

```powershell
cd experiment
.\run_session.ps1
```

The script will:
1. Activate the `value_capture_psychopy` conda environment automatically
2. Prompt for subject ID, session number, and number of runs (8 or 10)
3. Optionally run the **DeepMREye calibration** (from `experiment/deepmreye_calibration/`) before the first task run
4. Step through each run with a confirmation prompt — type `q` to abort early
5. Offer a second DeepMREye calibration run at the end of the session

#### First-time setup (submodule)

The DeepMREye calibration lives in `experiment/deepmreye_calibration/` as a git submodule. After cloning the repository, initialise it with:

```bash
git submodule update --init
```

### Output

Logs are written to `experiment/logs/sub-XX/ses-Y/run-Z/` in BIDS-like naming:

```
sub-01_ses-2_task-val_cap_run-1_events.tsv
sub-01_ses-2_task-val_cap_run-1_parameters.yml
```

## Settings

Key parameters in `settings/default.yml` (scanner) and `settings/debug.yml` (laptop):

| Key | Default | Debug | Description |
|---|---|---|---|
| `design.n_trials` | 40 | 40 | Trials per run (must be divisible by 4 and by 20) |
| `design.feedback_probability` | 0.5 | 1.0 | Fraction of trials showing feedback |
| `durations.target` | 1.75 | 1.75 | Response window (s) |
| `durations.iti1/iti2` | [0.5…4.0] | [0.5] | ITI durations tiled (s); right-skewed, mean = 1.5 s |
| `experiment.keys` | `["y","b"]` | `["a","s"]` | [no-dot, dot] response keys |
| `experiment.eccentricity_stimulus` | 4.0 | 4.0 | Search array eccentricity (°) |

## Data Analysis

```bash
conda activate value_capture_analysis
jupyter lab
```

Analysis notebooks live in `notebooks/`.
