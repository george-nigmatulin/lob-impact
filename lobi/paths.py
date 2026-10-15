"""Resolve the data roots the benchmark reads and writes.

Every location comes from here. Set the environment variables, or copy
``config/paths.example.yaml`` to ``config/paths.yaml`` and edit it; the environment
wins over the file, the file over the defaults. A relative path is taken relative to
the current directory when it comes from an environment variable, and relative to
LOBI_REPO when it comes from the YAML file or a default.

    LOBI_REPO      base of the defaults: the source checkout when the package runs from
                   one (pip install -e .), otherwise the current directory
    LOBI_DATA      parent of the two rollout trees, if you keep them together
    LOBI_CONTROLS  rollouts without a metaorder: free runs and the blind arm
    LOBI_GRID      metaorder-injection rollouts (the visible arm)
    LOBI_RESULTS   caches and tables the tools write

Every tool also takes explicit path flags, which override these roots. Nothing here
touches the network, and the tools write only under LOBI_RESULTS or an explicit
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
                          '(pip install PyYAML), or set the LOBI_* environment variables')
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve(value, repo):
    value = os.path.expanduser(str(value))
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(repo, value))


LOBI_REPO = _resolve(os.environ.get('LOBI_REPO') or _DEFAULT_REPO, os.getcwd())
_YAML = _from_yaml(LOBI_REPO)


def _root(key, default):
    """Environment first, then config/paths.yaml, then the default."""
    if os.environ.get(key):
        return _resolve(os.environ[key], os.getcwd())
    return _resolve(_YAML.get(key.lower()) or default, LOBI_REPO)


LOBI_DATA = _root('LOBI_DATA', 'data')
LOBI_CONTROLS = _root('LOBI_CONTROLS', os.path.join(LOBI_DATA, 'controls'))
LOBI_GRID = _root('LOBI_GRID', os.path.join(LOBI_DATA, 'grid'))
LOBI_RESULTS = _root('LOBI_RESULTS', 'results')

ROOTS = ('LOBI_REPO', 'LOBI_DATA', 'LOBI_CONTROLS', 'LOBI_GRID', 'LOBI_RESULTS')


def describe():
    """Print the resolved roots: the first thing to check when a file is not found."""
    for key in ROOTS:
        print('%-14s %s' % (key, globals()[key]))


def main(argv=None):
    import argparse
    argparse.ArgumentParser(prog='python -m lobi.paths',
                            description='Print the resolved data roots; the module docstring '
                                        'explains where each comes from.').parse_args(argv)
    describe()


if __name__ == '__main__':
    main()
