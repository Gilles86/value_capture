"""Model-free gain readout: decode each held-out trial's visual input from its GLMsingle betas (after retsupp's decoding).

The encoding model has PRFs only (Gaussian PRF, bar + 8 search items, the paradigm of fit_prf.py) and NO gain terms.
A target or distractor gain then shows up as decoded intensity at that item (and on the bar near it) above or below
the true, identical input, without fitting a gain-field model.

Cross-validation by session: PRFs from ``prf_glmsingle_items_train-ses<other>`` (fit_prf.py --train-sessions), the
noise model (ResidualFitter: omega, dof) from the training session's betas, voxel selection on the training fit's R².
The decoded session never entered any of these (GLMsingle itself was fitted jointly over all runs, a mild leak).

Decoders (closed-form MAP for the linear Gaussian PRF model, one image per trial, no temporal coupling):
  outline: only the pixels the true stimulus covers (bar + 8 items) are free, the rest fixed at 0, so the
           decoder estimates how strongly each part of the known input drove the voxels.
  free:    every pixel free.

Readout per trial and ROI (table): decoded mean over each item's footprint and over the bar inside each quadrant,
labelled by role: target, distractor (high / low value), the neutral diagonal items, cardinal items.

    python value_capture/decode/decode_trials.py 04 --test-session 2 --rois V1 V2 V3
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nilearn import masking  # noqa: E402

from value_capture.prf import fit_prf as fp  # noqa: E402
from value_capture.utils.data import Subject  # noqa: E402

ROIS = {'V1': (1,), 'V2': (2,), 'V3': (3,), 'hV4': (4,), 'VO': (5, 6), 'LO': (7, 8), 'TO': (9, 10),
        'V3AB': (11, 12)}
DIAGONALS = (1, 3, 5, 7)


def item_masks():
    """(8, n_pixels) footprint of each item at either orientation, on the fitting grid."""
    g = fp.make_grid_coordinates()
    out = []
    for i in range(8):
        both = (fp.make_items_image(i, 0.).ravel() > 0) | (fp.make_items_image(i, 90.).ravel() > 0)
        cx, cy = fp.ITEM_XY[i]
        out.append(both & (np.hypot(g.x - cx, g.y - cy).to_numpy() < 1.0))   # drop the other items
    return np.array(out)


def quadrant_of(i):
    """Quadrant (sx, sy) facing diagonal item i."""
    x, y = fp.ITEM_XY[i]
    return np.sign(x), np.sign(y)


def roles(target, distractor):
    """Role of each of the 8 items in this trial."""
    r = {}
    for i in range(8):
        if i == target:
            r[i] = 'target'
        elif not np.isnan(distractor) and i == int(distractor):
            r[i] = 'distractor'
        elif i in DIAGONALS:
            r[i] = 'neutral_diag'
        else:
            r[i] = 'cardinal'
    return r


def readout(decoded, paradigm, trials, imasks, grid):
    gx, gy = grid.x.to_numpy(), grid.y.to_numpy()
    inside = np.hypot(gx, gy) <= fp.GEOM['radius']
    rows = []
    for k, (_, t) in enumerate(trials.iterrows()):
        dec, true = decoded[k], paradigm[k]
        bar = (true > 0) & inside
        rl = roles(int(t.target_location), t.distractor_location)
        row = dict(trial_index=k, session=t.session, run=t.run, trial_nr=t.trial_nr, value_rank=t.value_rank,
                   distractor_present=bool(t.distractor_present), target_location=t.target_location,
                   distractor_location=t.distractor_location, bar_position=t.bar_position,
                   bar_orientation=t.bar_orientation, bar_mean=float(dec[bar].mean()) if bar.any() else np.nan)
        acc = {}
        for i in range(8):
            acc.setdefault(f'item_{rl[i]}', []).append(dec[imasks[i]].mean())
        for key, v in acc.items():
            row[key] = float(np.mean(v))
        # bar inside the quadrant facing each diagonal item
        for i in DIAGONALS:
            sx, sy = quadrant_of(i)
            q = bar & (np.sign(gx) == sx) & (np.sign(gy) == sy)
            if q.any():
                key = f'bar_{rl[i]}'
                acc_bar = row.get(key + '_sum', 0.) + dec[q].sum()
                row[key + '_sum'] = acc_bar
                row[key + '_n'] = row.get(key + '_n', 0) + int(q.sum())
        for key in [k_ for k_ in list(row) if k_.endswith('_sum')]:
            row[key[:-4]] = row.pop(key) / row.pop(key[:-4] + '_n')
        rows.append(row)
    return pd.DataFrame(rows)


def main(subject, test_session, rois, bids_folder, max_voxels, min_r2, decoders, l2_norm, learning_rate,
         max_iter, min_iter, out_root, gm_threshold=.1, resid_max_iter=2000):
    from braincoder.models import GaussianPRF2D
    from braincoder.optimize import ResidualFitter

    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sub.get_sessions()
    train_session = [s for s in sessions if s != test_session][0]
    prf_dir = (Path(bids_folder) / 'derivatives' / f'prf_glmsingle_items_train-ses{train_session}'
               / f'sub-{subject}' / 'ses-all' / 'func')
    pars_all = pd.read_csv(prf_dir / f'sub-{subject}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv',
                           sep='\t', index_col=0)

    betas = sub.get_single_trial_estimates(sessions, glmsingle_deriv='glmsingle')
    trials = fp.add_item_info(sub, sub.get_trial_metadata(sessions, glmsingle_deriv='glmsingle').reset_index(drop=True))
    raw = pd.concat([sub.get_onsets(int(s), int(r)).query("event_type == 'target'")
                     [['trial_nr', 'distractor_location', 'distractor_present']].assign(session=s, run=r)
                     for (s, r), _ in trials.groupby(['session', 'run'])])
    trials = trials.drop(columns=['distractor_present'], errors='ignore').merge(
        raw.drop_duplicates(['session', 'run', 'trial_nr']), on=['session', 'run', 'trial_nr'], how='left')
    mask_img = fp._get_gm_mask(sub, sessions[0], bids_folder, gm_threshold)
    Y = masking.apply_mask(betas, mask_img).astype(np.float32)
    assert Y.shape[1] == len(pars_all), (Y.shape, len(pars_all))
    varea = masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=sessions[0]), mask_img)

    grid = fp.make_grid_coordinates()
    paradigm = fp.build_paradigm(trials, 'bar_items')
    is_train = (trials.session == train_session).to_numpy()
    imasks = item_masks()
    out_root = Path(out_root) / f'sub-{subject}'
    out_root.mkdir(parents=True, exist_ok=True)

    for roi in rois:
        sel = np.flatnonzero(np.isin(np.round(varea), ROIS[roi]) & (pars_all.r2.to_numpy() >= min_r2)
                             & np.isfinite(pars_all.r2.to_numpy()))
        if len(sel) > max_voxels:
            sel = sel[np.argsort(-pars_all.r2.to_numpy()[sel])[:max_voxels]]
        if len(sel) < 20:
            print(f'{roi}: {len(sel)} voxels, skipped', flush=True)
            continue
        print(f'{roi}: {len(sel)} voxels (train R² >= {min_r2})', flush=True)
        pars = pars_all.iloc[sel][['x', 'y', 'sd', 'baseline', 'amplitude']].reset_index(drop=True).astype(np.float32)

        def frame(a):
            return pd.DataFrame(a, index=pd.Index(np.arange(len(a)), name='frame'))

        tr_data, tr_par = frame(Y[is_train][:, sel]), frame(paradigm[is_train])
        te_data, te_par = frame(Y[~is_train][:, sel]), frame(paradigm[~is_train])
        t0 = time.time()
        model = GaussianPRF2D(grid_coordinates=grid, paradigm=tr_par, data=tr_data, parameters=pars)
        omega, dof = ResidualFitter(model=model, data=tr_data, paradigm=tr_par, parameters=pars).fit(
            max_n_iterations=resid_max_iter, progressbar=False, method='gauss')
        print(f'  noise model on {is_train.sum()} training trials: {time.time() - t0:.0f} s', flush=True)
        model = GaussianPRF2D(grid_coordinates=grid, paradigm=te_par, data=te_data, parameters=pars)
        # The Gaussian PRF model is linear in the stimulus: y = b + W s (W = amplitude · RF, voxels × pixels). With
        # Gaussian noise (covariance Ω, the ResidualFitter estimate; its t dof is ignored) and an L2 prior, the MAP
        # image is closed form: s = (Wᵀ Ω⁻¹ W + 2λ I)⁻¹ Wᵀ Ω⁻¹ (y − b). No StimulusFitter (needs TensorFlow + TFP,
        # not in this JAX env) and no gradient descent to tune.
        P_te = paradigm[~is_train]
        px = np.flatnonzero(P_te.max(0) > 0)                       # pixels any test trial uses
        eye = pd.DataFrame(np.eye(paradigm.shape[1], dtype=np.float32)[px], index=pd.Index(range(len(px)), name='frame'))
        zero = pd.DataFrame(np.zeros((1, paradigm.shape[1]), np.float32), index=pd.Index([0], name='frame'))
        b = np.asarray(model.predict(paradigm=zero, parameters=pars))[0]
        W = (np.asarray(model.predict(paradigm=eye, parameters=pars)) - b).T.astype(np.float64)   # voxels × px
        Oi = np.linalg.inv(np.asarray(omega, np.float64))
        WtOi = W.T @ Oi
        R = te_data.to_numpy(np.float64) - b
        for dec in decoders:
            t0 = time.time()
            decoded = np.zeros(P_te.shape, np.float32)
            if dec == 'free':
                A = np.linalg.solve(WtOi @ W + 2 * l2_norm * np.eye(len(px)), WtOi)
                decoded[:, px] = (A @ R.T).T
            else:                                                   # outline: only the true input's pixels free
                for k in range(len(R)):
                    m = P_te[k, px] > 0
                    Wm = W[:, m]
                    decoded[k, px[m]] = np.linalg.solve(Wm.T @ Oi @ Wm + 2 * l2_norm * np.eye(m.sum()),
                                                        Wm.T @ Oi @ R[k])
            ro = readout(decoded, P_te, trials[~is_train].reset_index(drop=True), imasks, grid)
            true_r = [np.corrcoef(decoded[k], paradigm[~is_train][k])[0, 1] for k in range(len(decoded))]
            ro = ro.assign(subject=subject, roi=roi, decoder=dec, n_voxels=len(sel), r_true=true_r,
                           train_session=train_session)
            stem = f'ses-{test_session}_roi-{roi}_dec-{dec}'
            ro.to_csv(out_root / f'readout_{stem}.tsv', sep='\t', index=False)
            np.savez_compressed(out_root / f'decoded_{stem}.npz', decoded=decoded.astype(np.float16),
                                voxel_index=sel)
            print(f'  {dec}: {time.time() - t0:.0f} s, median r(decoded, true) = {np.nanmedian(true_r):.3f}',
                  flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--test-session', type=int, required=True)
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'hV4', 'LO', 'V3AB'])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--max-voxels', type=int, default=2000)
    ap.add_argument('--min-r2', type=float, default=.1, help='R² of the training-session PRF fit')
    ap.add_argument('--decoders', nargs='+', default=['outline', 'free'])
    ap.add_argument('--l2-norm', type=float, default=1.0)
    ap.add_argument('--learning-rate', type=float, default=.5)
    ap.add_argument('--max-iter', type=int, default=1000)
    ap.add_argument('--min-iter', type=int, default=200)
    ap.add_argument('--out-root', default='/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/decoded_trials')
    a = ap.parse_args()
    main(a.subject, a.test_session, a.rois, a.bids_folder, a.max_voxels, a.min_r2, a.decoders, a.l2_norm,
         a.learning_rate, a.max_iter, a.min_iter, a.out_root)
