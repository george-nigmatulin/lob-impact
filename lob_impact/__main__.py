"""List the tools: python -m lob_impact."""
import importlib

TOOLS = ('flow_balance', 'drift', 'scorecard', 'trajectory', 'master_curve', 'triangle',
         'points', 'delta', 'decay_shape', 'vol', 'synthetic', 'paths')


def main():
    print('lob_impact tools. Run python -m lob_impact.<tool> --help, or read the full')
    print('documentation of a tool with python -m pydoc lob_impact.<tool>.\n')
    for name in TOOLS:
        doc = (importlib.import_module('lob_impact.' + name).__doc__ or '').strip()
        print('  %-13s %s' % (name, doc.split('\n')[0]))


if __name__ == '__main__':
    main()
