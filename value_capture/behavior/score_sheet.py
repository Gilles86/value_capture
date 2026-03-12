"""Print a score sheet of total points earned per participant (scanning runs only)."""

import pandas as pd
from value_capture.utils.data import get_all_subject_ids, Subject


def make_score_sheet(bids_folder='/data/ds-valuecapture'):
    rows = []

    for subject_id in get_all_subject_ids():
        sub = Subject(subject_id, bids_folder=bids_folder)

        for session in sub.get_sessions():
            df = sub.get_behavioral_data(session=session, exclude_outlier_rts=False)

            total = df['earned_points'].sum()
            n_trials = len(df)
            n_responded = df['rt'].notna().sum()
            accuracy = df['correct'].mean()

            rows.append({
                'subject': subject_id,
                'session': session,
                'total_points': int(total),
                'n_trials': n_trials,
                'n_responded': n_responded,
                'accuracy': round(accuracy, 2),
            })

    score_sheet = pd.DataFrame(rows).set_index(['subject', 'session'])
    return score_sheet


if __name__ == '__main__':
    sheet = make_score_sheet()
    print(sheet.to_string())
