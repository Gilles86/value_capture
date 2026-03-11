import argparse
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
from nilearn import image
from tqdm import tqdm


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
    "MRAcquisitionType": "3D",
    "PulseSequenceType": "EPI",
    "PartialFourier": 0.796610177,
    "PhaseEncodingDirection": "j",
    "TotalReadoutTime": 0.027578312,
}

FMAP_JSON = {
    "EchoTimeDifference": 0.001,
    "EchoTime1": 0.0045,
    "EchoTime2": 0.0055,
}


def main(subject, session, bids_dir):
    bids_dir = Path(bids_dir)
    source_dir = bids_dir / "sourcedata" / "mri" / f"sub-{subject:02d}_ses-{session}"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir = bids_dir / f"sub-{subject:02d}" / f"ses-{session}"
    target_dir.mkdir(parents=True, exist_ok=True)

    _convert_anat(subject, session, source_dir, target_dir)
    _convert_func(subject, session, source_dir, target_dir)
    _convert_fmap(subject, session, source_dir, target_dir)
    _convert_behavior(subject, session, bids_dir, target_dir)

    print(f"\nDone: sub-{subject:02d} ses-{session}")


def _convert_anat(subject, session, source_dir, target_dir):
    print("\n--- Anatomical ---")
    anat_dir = target_dir / "anat"
    anat_dir.mkdir(exist_ok=True)

    sub = f"sub-{subject:02d}"
    ses = f"ses-{session}"

    if session == 1:
        # _1.PAR: 4 vols — inv-1 and inv-2 (magnitude and phase each)
        candidates = list(source_dir.glob("*acq-MP2_085mm*_1.PAR"))
        print(f"Found {len(candidates)} MP2RAGE (_1) candidates")
        assert len(candidates) in [0, 1]
        if candidates:
            nii = nib.Nifti1Image(nib.load(candidates[0]).dataobj,
                                  nib.load(candidates[0]).affine)
            image.index_img(nii, 0).to_filename(
                anat_dir / f"{sub}_{ses}_inv-1_MP2RAGE.nii.gz")
            image.index_img(nii, 1).to_filename(
                anat_dir / f"{sub}_{ses}_inv-2_MP2RAGE.nii.gz")
            print("Saved inv-1 and inv-2 MP2RAGE")

        # _3.PAR: UNI T1 map
        candidates = list(source_dir.glob("*acq-MP2_085mm*_3.PAR"))
        print(f"Found {len(candidates)} T1w (UNI) candidates")
        assert len(candidates) in [0, 1]
        if candidates:
            raw = nib.load(candidates[0])
            nii = nib.Nifti1Image(raw.dataobj, raw.affine)
            if nii.ndim == 4:
                nii = image.index_img(nii, 0)
            nii = image.math_img("np.where(img == img.max(), 0, img)", img=nii)
            nii = image.math_img("img - img.min()", img=nii)
            nii.to_filename(anat_dir / f"{sub}_{ses}_T1w.nii.gz")
            print("Saved T1w")

    elif session == 2:
        candidates = list(source_dir.glob("*acq-T2w*.PAR"))
        print(f"Found {len(candidates)} T2w candidates")
        assert len(candidates) in [0, 1]
        if candidates:
            raw = nib.load(candidates[0])
            nii = nib.Nifti1Image(raw.dataobj, raw.affine)
            if nii.ndim == 4:
                nii = image.math_img("np.where(img == img.max(), 0, img)", img=nii)
            nii.to_filename(anat_dir / f"{sub}_{ses}_T2w.nii.gz")
            print("Saved T2w")


def _convert_func(subject, session, source_dir, target_dir):
    print("\n--- Functional ---")
    func_dir = target_dir / "func"
    func_dir.mkdir(exist_ok=True)

    sub = f"sub-{subject:02d}"
    ses = f"ses-{session}"

    for task in ["valuecapture", "deepmreye"]:
        candidates = sorted(source_dir.glob(f"*task-{task}_run-*_bold_*.PAR"))
        print(f"Found {len(candidates)} task-{task} bold runs")

        for par in tqdm(candidates, desc=f"task-{task}"):
            try:
                raw = image.load_img(par)
                nii = nib.Nifti1Image(raw.dataobj, raw.affine)
                n = nii.shape[-1]
                mag = image.index_img(nii, slice(0, n // 2))
                phase = image.index_img(nii, slice(n // 2, n))

                run = int(re.search(r"run-(\d+)", par.name).group(1))
                base = func_dir / f"{sub}_{ses}_task-{task}_run-{run}"

                mag.to_filename(f"{base}_part-mag_bold.nii.gz")
                phase.to_filename(f"{base}_part-phase_bold.nii.gz")

                with open(f"{base}_part-mag_bold.json", "w") as f:
                    json.dump(BOLD_JSON, f, indent=2)
                with open(f"{base}_part-phase_bold.json", "w") as f:
                    json.dump(BOLD_JSON, f, indent=2)

            except Exception as e:
                print(f"  ERROR {par.name}: {e}")

    # Sbref — one per scanner block (run-1, run-2, run-3)
    sbref_candidates = sorted(source_dir.glob("*_sbref_*.PAR"))
    print(f"Found {len(sbref_candidates)} sbref candidates")

    for par in sbref_candidates:
        run = int(re.search(r"run-(\d+)", par.name).group(1))
        raw = image.load_img(par)
        nii = nib.Nifti1Image(raw.dataobj, raw.affine)
        # Take first volume (magnitude)
        if nii.ndim == 4:
            nii = image.index_img(nii, 0)
        base = func_dir / f"{sub}_{ses}_run-{run}_sbref"
        nii.to_filename(f"{base}.nii.gz")
        with open(f"{base}.json", "w") as f:
            json.dump(BOLD_JSON, f, indent=2)


def _convert_fmap(subject, session, source_dir, target_dir):
    print("\n--- Field maps ---")
    fmap_dir = target_dir / "fmap"
    fmap_dir.mkdir(exist_ok=True)

    sub = f"sub-{subject:02d}"
    ses = f"ses-{session}"

    candidates = sorted(source_dir.glob("*_phasediff_*.PAR"))
    print(f"Found {len(candidates)} fmap candidates")
    n = len(candidates)

    # Build IntendedFor: split 8 valuecapture runs across n fmaps, plus deepmreye for first fmap
    def intended_for(ix):
        runs_per_fmap = 8 // n  # integer division
        remainder = 8 % n
        # Each fmap covers runs_per_fmap runs, last fmap gets the remainder too
        start = 1 + ix * runs_per_fmap
        end = start + runs_per_fmap + (remainder if ix == n - 1 else 0)
        intended = [
            f"ses-{session}/func/{sub}_{ses}_task-valuecapture_run-{r}_part-mag_bold.nii.gz"
            for r in range(start, end)
        ]
        if ix == 0:
            intended.append(
                f"ses-{session}/func/{sub}_{ses}_task-deepmreye_run-1_part-mag_bold.nii.gz"
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
    print("\n--- Behavior ---")
    func_dir = target_dir / "func"
    behavior_src = bids_dir / "sourcedata" / "behavior" / f"sub-{subject:02d}" / f"ses-{session}"

    if not behavior_src.exists():
        print(f"No behavior sourcedata at {behavior_src}, skipping.")
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
            print(f"  {events_file.name} → {dest.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw MRI data to BIDS format.")
    parser.add_argument("subject", type=int, help="Subject number (e.g. 1)")
    parser.add_argument("session", type=int, help="Session number (1 or 2)")
    parser.add_argument("--bids_dir", default="/data/ds-valuecapture",
                        help="Path to BIDS directory")
    args = parser.parse_args()
    main(args.subject, args.session, args.bids_dir)
