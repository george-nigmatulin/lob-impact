"""The lob_impact command: ``lob_impact`` lists the tools, ``lob_impact <tool> [args]`` runs one.

The same as ``python -m lob_impact`` and ``python -m lob_impact.<tool> [args]``.
"""
import importlib
import sys

TOOLS = ('flow_balance', 'drift', 'scorecard', 'trajectory', 'master_curve', 'triangle',
         'points', 'delta', 'decay_shape', 'vol', 'synthetic', 'paths')


def list_tools():
    print('lob_impact tools. Run lob_impact <tool> --help (or python -m lob_impact.<tool> --help),')
    print('or read the full documentation of a tool with python -m pydoc lob_impact.<tool>.\n')
    for name in TOOLS:
        doc = (importlib.import_module('lob_impact.' + name).__doc__ or '').strip()
        print('  %-13s %s' % (name, doc.split('\n')[0]))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ('-h', '--help'):
        list_tools()
        return 0
    tool, rest = argv[0].replace('-', '_'), argv[1:]
    if tool not in TOOLS:
        print(f'lob_impact: no tool {argv[0]!r}; one of {", ".join(TOOLS)}', file=sys.stderr)
        return 2
    return importlib.import_module('lob_impact.' + tool).main(rest) or 0


if __name__ == '__main__':
    sys.exit(main())
