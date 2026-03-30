#!/usr/bin/env python3
"""
Unwrap GRE phasediff fieldmaps with ROMEO and save as pre-computed Hz fieldmaps.

For each sub-XX/ses-Y/fmap/..._phasediff.nii.gz in the BIDS dataset:
  1. Run ROMEO to spatially unwrap the phase
  2. Convert unwrapped radians → Hz using EchoTimeDifference from the JSON
  3. Save as ..._fieldmap.nii.gz with a BIDS-compliant JSON sidecar

fmriprep will then use the _fieldmap (Hz) files directly, bypassing its own
(PRELUDE-based) phase unwrapping entirely.

Usage:
    python romeo_unwrap_fieldmaps.py [--bids-dir DIR] [--romeo PATH] [--dry-run]
"""
import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


ROMEO = Path.home() / "bin" / "mritools" / "bin" / "romeo"


def unwrap_one(phasediff: Path, magnitude: Path, romeo: Path, tmpdir: Path) -> Path:
    out = tmpdir / "unwrapped.nii"
    cmd = [
        str(romeo),
        "-p", str(phasediff),
        "-m", str(magnitude),
        "-o", str(out),
        # No --no-phase-rescale: phasediff is in Philips raw units (±500),
        # ROMEO will rescale to [-π, π] before unwrapping
        "-k", "robustmask",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ROMEO failed:\n{result.stderr}")
    # ROMEO may write .nii or .nii.gz
    if not out.exists():
        out = tmpdir / "unwrapped.nii.gz"
    return out


def process_phasediff(phasediff: Path, romeo: Path, dry_run: bool):
    fmap_dir = phasediff.parent
    stem = phasediff.name.replace("_phasediff.nii.gz", "")

    magnitude = fmap_dir / f"{stem}_magnitude1.nii.gz"
    json_in   = fmap_dir / f"{stem}_phasediff.json"
    nii_out   = fmap_dir / f"{stem}_fieldmap.nii.gz"
    json_out  = fmap_dir / f"{stem}_fieldmap.json"

    if not magnitude.exists():
        print(f"  [SKIP] magnitude not found: {magnitude}")
        return
    if not json_in.exists():
        print(f"  [SKIP] JSON not found: {json_in}")
        return

    meta = json.loads(json_in.read_text())
    delta_te = meta["EchoTimeDifference"]  # seconds
    intended_for = meta.get("IntendedFor", [])

    print(f"  phasediff : {phasediff.name}")
    print(f"  ΔTE       : {delta_te*1000:.2f} ms")

    if dry_run:
        print(f"  [DRY-RUN] would write {nii_out.name} + {json_out.name}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        unwrapped_path = unwrap_one(phasediff, magnitude, romeo, Path(tmpdir))

        img = nib.load(unwrapped_path)
        data_rad = img.get_fdata(dtype=np.float32)

        # radians → Hz, negated to match PRELUDE/fmriprep sign convention
        data_hz = -data_rad / (2.0 * math.pi * delta_te)

        out_img = nib.Nifti1Image(data_hz, img.affine, img.header)
        out_img.header.set_data_dtype(np.float32)
        nib.save(out_img, str(nii_out))

    # JSON sidecar: Units required by BIDS for _fieldmap suffix
    json_out.write_text(json.dumps({
        "Units": "Hz",
        "IntendedFor": intended_for,
    }, indent=2))

    print(f"  → {nii_out.name}  ({data_hz.min():.1f} … {data_hz.max():.1f} Hz)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-dir", default="/data/ds-valuecapture",
                        help="BIDS root directory")
    parser.add_argument("--romeo", default=str(ROMEO),
                        help="Path to romeo binary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without running ROMEO")
    args = parser.parse_args()

    romeo = Path(args.romeo)
    if not romeo.exists():
        raise FileNotFoundError(f"ROMEO binary not found: {romeo}")

    bids = Path(args.bids_dir)
    phasediffs = sorted(bids.glob("sub-*/ses-*/fmap/*_phasediff.nii.gz"))

    if not phasediffs:
        print("No phasediff files found.")
        return

    print(f"Found {len(phasediffs)} phasediff files\n")
    for pd in phasediffs:
        print(f"[{pd.parent.parent.parent.name}/{pd.parent.parent.name}] {pd.stem}")
        process_phasediff(pd, romeo, dry_run=args.dry_run)
        print()


if __name__ == "__main__":
    main()
