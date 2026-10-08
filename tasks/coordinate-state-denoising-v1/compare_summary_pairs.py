"""Pair existing complete estimates by system; never subtract CI endpoints."""
import argparse
import json
from pathlib import Path

from molvid.evaluation.ca_detail import system_estimate
from molvid.runtime import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stratum', choices=['all','old16','new48'])
    args = parser.parse_args()
    left, right = [json.loads(path.read_text()) for path in (args.candidate, args.reference)]
    if any(value.get('status') != 'completed' for value in (left, right)):
        raise ValueError('Only completed panels can be compared')
    for key in ('systems', 'conditions', 'forecasts', 'future_frames'):
        if left.get(key) != right.get(key):
            raise ValueError(f'Unmatched panel: {key}')
    left_estimates = left['strata'][args.stratum] if args.stratum else left['estimates']
    right_estimates = right['strata'][args.stratum] if args.stratum else right['estimates']
    if not left_estimates or set(left_estimates) != set(right_estimates):
        raise ValueError('Metric definitions differ')
    paired = {}
    for metric in left_estimates:
        a = left_estimates[metric]['per_system']
        b = right_estimates[metric]['per_system']
        if set(a) != set(b) or not a:
            raise ValueError(f'Unmatched systems: {metric}')
        paired[metric] = system_estimate([(system, a[system] - b[system]) for system in sorted(a)])
    atomic_write_json(args.output, dict(status='completed', model_calls=0,
        candidate=str(args.candidate), reference=str(args.reference),
        difference='candidate minus reference; sign is not universally better/worse',
        estimates=paired, stratum=args.stratum, limitations=['Reused development systems; paired system bootstrap',
            'Different released-model training and native input contracts remain confounded',
            'No equilibrium or long-time kinetics claim']))


if __name__ == '__main__':
    main()
