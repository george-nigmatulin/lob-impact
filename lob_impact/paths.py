"""Resolve the data roots the benchmark reads and writes.

Every location comes from here. Set the environment variables, or copy
``config/paths.example.yaml`` to ``config/paths.yaml`` and edit it; the environment
wins over the file, the file over the defaults. A relative path is taken relative to
the current directory when it comes from an environment variable, and relative to
LOB_IMPACT_REPO when it comes from the YAML file or a default.

    LOB_IMPACT_REPO      base of the defaults: the source checkout when the package runs from
                   one (pip install -e .), otherwise the current directory
    LOB_IMPACT_DATA      parent of the two rollout trees, if you keep them together
    LOB_IMPACT_CONTROLS  rollouts without a metaorder: free runs and the blind arm
    LOB_IMPACT_GRID      metaorder-injection rollouts (the visible arm)
    LOB_IMPACT_RESULTS   caches and tables the tools write

The names before version 0.2 (LOBI_*, and lobi_* in the YAML file) are still read.
Every tool also takes explicit path flags, which override these roots. Nothing here
touches the network, and the tools write only under LOB_IMPACT_RESULTS or an explicit
output flag.
"""
import os

try:                                    # optional: only needed for config/paths.yaml
    import yaml
except ImportError:                     # pragma: no cover
    yaml = None

_PKG = os.path.dirname(os.path.abspath(__file__))
_CHECKOUT = os.path.dirname(_PKG)
# a source checkout carries pyproject.toml next to the package; an installed copy does not
_DEFAULT_REPO = _CHECKOUT if os.path.exists(os.path.join(_CHECKOUT, 'pyproject.toml')) \
    else os.getcwd()


def _from_yaml(repo):
    path = os.path.join(repo, 'config', 'paths.yaml')
    if not os.path.exists(path):
        return {}
    if yaml is None:
        raise ImportError('config/paths.yaml exists but PyYAML is not installed '
                          '(pip install PyYAML), or set the LOB_IMPACT_* environment variables')
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve(value, repo):
    value = os.path.expanduser(str(value))
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(repo, value))


def _env(key):
    """An environment variable, or its name before 0.2 (LOBI_*, when the package was lobi)."""
    return os.environ.get(key) or os.environ.get(key.replace('LOB_IMPACT_', 'LOBI_'))


LOB_IMPACT_REPO = _resolve(_env('LOB_IMPACT_REPO') or _DEFAULT_REPO, os.getcwd())
_YAML = _from_yaml(LOB_IMPACT_REPO)


def _root(key, default):
    """Environment first, then config/paths.yaml, then the default."""
    if _env(key):
        return _resolve(_env(key), os.getcwd())
    from_yaml = _YAML.get(key.lower()) or _YAML.get(key.lower().replace('lob_impact_', 'lobi_'))
    return _resolve(from_yaml or default, LOB_IMPACT_REPO)


LOB_IMPACT_DATA = _root('LOB_IMPACT_DATA', 'data')
LOB_IMPACT_CONTROLS = _root('LOB_IMPACT_CONTROLS', os.path.join(LOB_IMPACT_DATA, 'controls'))
LOB_IMPACT_GRID = _root('LOB_IMPACT_GRID', os.path.join(LOB_IMPACT_DATA, 'grid'))
LOB_IMPACT_RESULTS = _root('LOB_IMPACT_RESULTS', 'results')

ROOTS = ('LOB_IMPACT_REPO', 'LOB_IMPACT_DATA', 'LOB_IMPACT_CONTROLS', 'LOB_IMPACT_GRID', 'LOB_IMPACT_RESULTS')


def describe():
    """Print the resolved roots: the first thing to check when a file is not found."""
    for key in ROOTS:
        print('%-14s %s' % (key, globals()[key]))


def main(argv=None):
    import argparse
    argparse.ArgumentParser(prog='python -m lob_impact.paths',
                            description='Print the resolved data roots; the module docstring '
                                        'explains where each comes from.').parse_args(argv)
    describe()


if __name__ == '__main__':
    main()
