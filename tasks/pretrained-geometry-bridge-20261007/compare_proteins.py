"""Paired scientific metrics from completed cached forecasts; no model calls."""
import argparse
import json
from pathlib import Path
from molvid.runtime import atomic_write_json
from tools.compare_expanded_native import paired_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('candidate', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    def read(root):
        summary = json.loads((root/'summary.json').read_text())
        if summary['status'] != 'completed' or summary['systems'] != 48:
            raise ValueError('Completed full48-system forecasts required')
        return json.loads((root/'rows.json').read_text())
    left, right = read(args.candidate), read(args.reference)
    # Historical PVB outputs contain one variant; do not pool projection arms.
    if len({r['variant'] for r in left}) != 1 or len({r['variant'] for r in right}) != 1:
        raise ValueError('One predeclared variant per comparison')
    result = paired_rows(left, right, {r['system'] for r in left})
    result.update(status='completed', candidate=str(args.candidate), reference=str(args.reference),
        model_calls=0, independent_test=False, promotion=False,
        limitations=['Marginal system bootstrap, not multiplicity corrected',
            'Released PVB H1 versus candidate H4 and different training exposure',
            'Mean amplitude ratio is not universally lower-is-better'])
    atomic_write_json(args.output, result)
    print(json.dumps({k: {a:b for a,b in v.items() if a != 'per_system'}
                      for k,v in result['delta_left_minus_right'].items()}))
