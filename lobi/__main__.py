"""List the tools: python -m lobi."""
import importlib

TOOLS = ('flow_balance', 'drift', 'scorecard', 'trajectory', 'master_curve', 'triangle',
         'points', 'delta', 'decay_shape', 'vol', 'synthetic', 'paths')


def main():
    print('LOB-Impact Benchmark tools. Run python -m lobi.<tool> --help, or read the full')
    print('documentation of a tool with python -m pydoc lobi.<tool>.\n')
    for name in TOOLS:
        doc = (importlib.import_module('lobi.' + name).__doc__ or '').strip()
        print('  %-13s %s' % (name, doc.split('\n')[0]))


if __name__ == '__main__':
    main()
