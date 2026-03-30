import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
from nilearn import image


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _slice_timing(n_slices=81, mb=3, tr=1.6):
    """Ascending SMS slice timing: excitation k fires slices k, k+N/MB, k+2*N/MB."""
    n_exc = n_slices // mb
    dt = tr / n_exc
    return [round((i % n_exc) * dt, 9) for i in range(n_slices)]


SBREF_JSON = {
    "Manufacturer": "Philips",
    "ManufacturersModelName": "Ingenia",
    "MagneticFieldStrength": 7,
    "ReceiveCoilName": ["SENSE-Head-7T-Ant", "SENSE-Head-7T-Post"],
    "FlipAngle": 66,
    "EchoTime": 0.024,
    "RepetitionTime": 1.6,
    "SliceThickness": 1.7,
    "VoxelSize": [1.68, 1.68, 1.7],
    "FieldOfView": [215, 215, 137.7],
    "MultibandAccelerationFactor": 1,
    "ParallelReductionFactorInPlane": 2.4,
    "FatSuppressionTechnique": "SPIR",
    "MRAcquisitionType": "2D",
    "PulseSequenceType": "EPI",
    "PartialFourier": 0.796610177,
    "PhaseEncodingDirection": "j",
    "TotalReadoutTime": 0.033138,  # 53 / 1599.4 Hz (SBref BW, MB=1)
}

BOLD_JSON = {
    "Manufacturer": "Philips",
    "ManufacturersModelName": "Ingenia",
    "MagneticFieldStrength": 7,
    "ReceiveCoilName": ["SENSE-Head-7T-Ant", "SENSE-Head-7T-Post"],
    "FlipAngle": 66,
    "EchoTime": 0.024,
    "RepetitionTime": 1.6,
    "SliceThickness": 1.7,
    "VoxelSize": [1.68, 1.68, 1.7],
    "FieldOfView": [215, 215, 137.7],
    "MultibandAccelerationFactor": 3,
    "ParallelReductionFactorInPlane": 2.4,
    "FatSuppressionTechnique": "SPIR",
    "MRAcquisitionType": "2D",
    "PulseSequenceType": "EPI",
    "PartialFourier": 0.796610177,
    "PhaseEncodingDirection": "j",
    "TotalReadoutTime": 0.027578312,
    "SliceTiming": _slice_timing(),
}

FMAP_JSON = {
    "EchoTimeDifference": 0.001,
    "EchoTime1": 0.0045,
    "EchoTime2": 0.0055,
}


def main(subject, session, bids_dir, behavior_only=False):
    bids_dir = Path(bids_dir)

    target_dir = bids_dir / f"sub-{subject:02d}" / f"ses-{session}"
    target_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _log(f"Starting sub-{subject:02d} ses-{session}")

    if not behavior_only:
        source_dir = bids_dir / "sourcedata" / "mri" / f"sub-{subject:02d}_ses-{session}"
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        _convert_anat(subject, session, source_dir, target_dir)
        _convert_func(subject, session, source_dir, target_dir)
        _convert_fmap(subject, session, source_dir, target_dir)

    _convert_behavior(subject, session, bids_dir, target_dir)

    _log(f"Done: sub-{subject:02d} ses-{session}  (total {time.time()-t0:.0f}s)")


def _convert_anat(subject, session, source_dir, target_dir):
    _log("--- Anatomical ---")
    anat_dir = target_dir / "anat"
    anat_dir.mkdir(exist_ok=True)

    sub = f"sub-{subject:02d}"
    ses = f"ses-{session}"

    if session == 1:
        candidates = list(source_dir.glob("*acq-MP2_085mm*_1.PAR"))
        assert len(candidates) in [0, 1]
        if candidates:
            _log(f"  Loading MP2RAGE: {candidates[0].name}")
            t = time.time()
            nii = nib.Nifti1Image(nib.load(candidates[0]).dataobj,
                                  nib.load(candidates[0]).affine)
            image.index_img(nii, 0).to_filename(
                anat_dir / f"{sub}_{ses}_inv-1_MP2RAGE.nii.gz")
            image.index_img(nii, 1).to_filename(
                anat_dir / f"{sub}_{ses}_inv-2_MP2RAGE.nii.gz")
            _log(f"  Saved inv-1 and inv-2 MP2RAGE ({time.time()-t:.0f}s)")

        candidates = list(source_dir.glob("*acq-MP2_085mm*_3.PAR"))
        assert len(candidates) in [0, 1]
        if candidates:
            _log(f"  Loading T1w UNI: {candidates[0].name}")
            t = time.time()
            raw = nib.load(candidates[0])
            nii = nib.Nifti1Image(raw.dataobj, raw.affine)
            if nii.ndim == 4:
                nii = image.index_img(nii, 0)
            nii = image.math_img("np.where(img == img.max(), 0, img)", img=nii)
            nii = image.math_img("img - img.min()", img=nii)
            nii.to_filename(anat_dir / f"{sub}_{ses}_T1w.nii.gz")
            _log(f"  Saved T1w ({time.time()-t:.0f}s)")

    elif session == 2:
        candidates = list(source_dir.glob("*acq-T2w*.PAR"))
        assert len(candidates) in [0, 1]
        if candidates:
            _log(f"  Loading T2w: {candidates[0].name}")
            t = time.time()
            raw = nib.load(candidates[0])
            nii = nib.Nifti1Image(raw.dataobj, raw.affine)
            if nii.ndim == 4:
                nii = image.math_img("np.where(img == img.max(), 0, img)", img=nii)
            nii.to_filename(anat_dir / f"{sub}_{ses}_T2w.nii.gz")
            _log(f"  Saved T2w ({time.time()-t:.0f}s)")


def _convert_func(subject, session, source_dir, target_dir):
    _log("--- Functional ---")
    func_dir = target_dir / "func"
    func_dir.mkdir(exist_ok=True)

    sub = f"sub-{subject:02d}"
    ses = f"ses-{session}"

    for task in ["valuecapture", "deepmreye"]:
        candidates = sorted(source_dir.glob(f"*task-{task}_run-*_bold_*.PAR"))
        n_runs = len(candidates)
        _log(f"  Found {n_runs} task-{task} bold runs")

        for i, par in enumerate(candidates, 1):
            run = int(re.search(r"run-(\d+)", par.name).group(1))
            size_mb = par.stat().st_size / 1e6
            _log(f"  [{i}/{n_runs}] task-{task} run-{run}  ({size_mb:.0f} MB)  loading...")
            t = time.time()
            try:
                raw = image.load_img(par)
                nii = nib.Nifti1Image(raw.dataobj, raw.affine)
                n = nii.shape[-1]
                mag = image.index_img(nii, slice(0, n // 2))
                phase = image.index_img(nii, slice(n // 2, n))

                base = func_dir / f"{sub}_{ses}_task-{task}_run-{run}"
                _log(f"  [{i}/{n_runs}] writing mag...")
                mag.to_filename(f"{base}_part-mag_bold.nii.gz")
                _log(f"  [{i}/{n_runs}] writing phase...")
                phase.to_filename(f"{base}_part-phase_bold.nii.gz")

                with open(f"{base}_part-mag_bold.json", "w") as f:
                    json.dump(BOLD_JSON, f, indent=2)
                with open(f"{base}_part-phase_bold.json", "w") as f:
                    json.dump(BOLD_JSON, f, indent=2)

                _log(f"  [{i}/{n_runs}] done ({time.time()-t:.0f}s)")

            except Exception as e:
                _log(f"  ERROR {par.name}: {e}")

    sbref_candidates = sorted(source_dir.glob("*_sbref_*.PAR"))
    _log(f"  Found {len(sbref_candidates)} sbref candidates")

    for par in sbref_candidates:
        run = int(re.search(r"run-(\d+)", par.name).group(1))
        _log(f"  sbref run-{run}  loading...")
        t = time.time()
        raw = image.load_img(par)
        nii = nib.Nifti1Image(raw.dataobj, raw.affine)
        if nii.ndim == 4:
            nii = image.index_img(nii, 0)
        base = func_dir / f"{sub}_{ses}_run-{run}_sbref"
        nii.to_filename(f"{base}.nii.gz")
        with open(f"{base}.json", "w") as f:
            json.dump(SBREF_JSON, f, indent=2)
        _log(f"  sbref run-{run} done ({time.time()-t:.0f}s)")


def _convert_fmap(subject, session, source_dir, target_dir):
    _log("--- Field maps ---")
    fmap_dir = target_dir / "fmap"
    fmap_dir.mkdir(exist_ok=True)

    sub = f"sub-{subject:02d}"
    ses = f"ses-{session}"

    candidates = sorted(source_dir.glob("*_phasediff_*.PAR"))
    _log(f"  Found {len(candidates)} fmap candidates")
    n = len(candidates)

    # Build IntendedFor: split 8 valuecapture runs across n fmaps, plus deepmreye for first fmap
    def intended_for(ix):
        runs_per_fmap = 8 // n  # integer division
        remainder = 8 % n
        # Each fmap covers runs_per_fmap runs, last fmap gets the remainder too
        start = 1 + ix * runs_per_fmap
        end = start + runs_per_fmap + (remainder if ix == n - 1 else 0)
        intended = [
            f"ses-{session}/func/{sub}_{ses}_task-valuecapture_run-{r}_rec-NORDIC_bold.nii.gz"
            for r in range(start, end)
        ]
        if ix == 0:
            intended.append(
                f"ses-{session}/func/{sub}_{ses}_task-deepmreye_run-1_rec-NORDIC_bold.nii.gz"
            )
        return intended

    for ix, par in enumerate(candidates):
        raw = image.load_img(par)
        nii = nib.Nifti1Image(raw.dataobj, raw.affine)
        mag = image.index_img(nii, 0)
        phasediff = image.index_img(nii, 1)

        fmap_json = dict(FMAP_JSON)
        fmap_json["IntendedFor"] = intended_for(ix)

        base = fmap_dir / f"{sub}_{ses}_run-{ix + 1}"
        mag.to_filename(f"{base}_magnitude1.nii.gz")
        phasediff.to_filename(f"{base}_phasediff.nii.gz")

        with open(f"{base}_magnitude1.json", "w") as f:
            json.dump(fmap_json, f, indent=2)
        with open(f"{base}_phasediff.json", "w") as f:
            json.dump(fmap_json, f, indent=2)


def _convert_behavior(subject, session, bids_dir, target_dir):
    _log("--- Behavior ---")
    func_dir = target_dir / "func"
    func_dir.mkdir(exist_ok=True)
    behavior_src = bids_dir / "sourcedata" / "behavior" / f"sub-{subject:02d}" / f"ses-{session}"

    if not behavior_src.exists():
        _log(f"  No behavior sourcedata at {behavior_src}, skipping.")
        return

    for run_dir in sorted(behavior_src.iterdir()):
        if not run_dir.is_dir():
            continue
        run_match = re.search(r"run-(\d+)", run_dir.name)
        if not run_match:
            continue
        run = int(run_match.group(1))

        for events_file in run_dir.glob("*_events.tsv"):
            dest = func_dir / f"sub-{subject:02d}_ses-{session}_task-valuecapture_run-{run}_events.tsv"
            shutil.copy2(events_file, dest)
            _log(f"  {events_file.name} → {dest.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw MRI data to BIDS format.")
    parser.add_argument("subject", type=int, help="Subject number (e.g. 1)")
    parser.add_argument("session", type=int, help="Session number (1 or 2)")
    parser.add_argument("--bids_dir", default="/data/ds-valuecapture",
                        help="Path to BIDS directory")
    parser.add_argument("--behavior_only", action="store_true",
                        help="Only convert behavior (skip MRI/fmap conversion)")
    args = parser.parse_args()
    main(args.subject, args.session, args.bids_dir, behavior_only=args.behavior_only)
