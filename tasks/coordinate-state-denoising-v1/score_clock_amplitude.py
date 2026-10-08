"""Stratify completed clock responses using existing TRAIN-frozen motion bins.

No coordinates are generated or modified. Labels use the real 100ps future
only for evaluation, shared between both clocks and every compared model.
"""
import argparse
import json
from pathlib import Path

from molvid.evaluation.ca_detail import system_estimate
from molvid.runtime import atomic_write_json


def clock_rows(root):
    summary = json.loads((root / 'summary.json').read_text())
    if summary['status'] != 'completed' or summary['conditions'] != 96:
        raise ValueError('Complete 96-condition clock panel required')
    rows = json.loads((root / 'rows.json').read_text())
    indexed = {(r['system'], r['anchor']): r['metrics'] for r in rows}
    if len(rows) != 96 or len(indexed) != 96:
        raise ValueError('Missing or duplicated clock conditions')
    return indexed


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('candidate', 'reference', 'amplitude', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    candidate, reference = clock_rows(args.candidate), clock_rows(args.reference)
    amplitude = json.loads(args.amplitude.read_text())
    protocol = json.loads(args.protocol.read_text())
    if amplitude['status'] != 'completed':
        raise ValueError('Completed amplitude labels required')
    entries = {e['sample_id']: e for e in protocol['records']}
    labels = {}
    for row in amplitude['rows']:
        entry = entries[row['sample_id']]
        if entry['system'] != row['system'] or entry['lag_ps'] != 100:
            raise ValueError('Labels must use the shared 100ps evaluation future')
        key = (entry['system'], entry['anchor'])
        if key in labels:
            raise ValueError('Duplicated amplitude condition')
        labels[key] = row['bin']
    if candidate.keys() != reference.keys() or labels.keys() != candidate.keys():
        raise ValueError('Identical complete condition sets required')
    for key in candidate:
        for name, value in candidate[key].items():
            if name.startswith(('MD_', 'zero_response_')) and reference[key].get(name) != value:
                raise ValueError('Candidate and reference must use identical MD clock targets')
    estimates = {}
    for label in ('low', 'medium', 'high'):
        keys = [key for key in candidate if labels[key] == label]
        metrics = {}
        for name in next(iter(candidate.values())):
            if any(candidate[key].get(name) is None or reference[key].get(name) is None for key in keys):
                raise ValueError('Do not silently drop missing metric observations')
            metrics[name] = {
                'candidate': system_estimate([(k[0], candidate[k][name]) for k in keys]),
                'reference': system_estimate([(k[0], reference[k][name]) for k in keys]),
                'candidate_minus_reference': system_estimate([(k[0], candidate[k][name] - reference[k][name]) for k in keys]),
            } if keys else {}
        estimates[label] = dict(conditions=len(keys), systems=len({k[0] for k in keys}), metrics=metrics)
    atomic_write_json(args.output, dict(
        status='completed', model_calls=0, optimizer_updates=0,
        candidate=str(args.candidate), reference=str(args.reference),
        amplitude_labels=str(args.amplitude), protocol=str(args.protocol),
        TRAIN_bin_boundaries_A=amplitude['TRAIN_bin_boundaries_A'], estimates=estimates,
        convention='Same100ps true-motion label for both clocks and all models; no target-derived conditions or sample exclusion',
        limitations=['Exploratory marginal system bootstrap within bins; not independent confirmation',
                     'Systems can contribute different anchors to different bins; bins are not independent cohorts',
                     'Conditioning evaluation on100ps future amplitude can select MD clock differences; not a causal amplitude effect',
                     'Two draws and one MD future/condition; matched query-index noise is not common-physical-time coupling',
                     'Response amplitude alone is not correct dynamics; inspect errors against MD and zero-response'],
        promotion=False))


if __name__ == '__main__':
    main()
