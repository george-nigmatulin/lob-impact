"""lob_impact.paths: the data roots from the environment, including the names before 0.2."""
import importlib
import os


def roots(env, monkeypatch):
    for key in list(os.environ):
        if key.startswith(('LOB_IMPACT_', 'LOBI_')):
            monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import lob_impact.paths as paths
    return importlib.reload(paths)


def test_the_environment_sets_the_roots(monkeypatch, tmp_path):
    p = roots({'LOB_IMPACT_RESULTS': str(tmp_path / 'r')}, monkeypatch)
    assert p.LOB_IMPACT_RESULTS == str(tmp_path / 'r')


def test_the_names_before_0_2_are_still_read(monkeypatch, tmp_path):
    p = roots({'LOBI_CONTROLS': str(tmp_path / 'c')}, monkeypatch)
    assert p.LOB_IMPACT_CONTROLS == str(tmp_path / 'c')


def test_the_new_name_wins_over_the_old(monkeypatch, tmp_path):
    p = roots({'LOBI_GRID': str(tmp_path / 'old'), 'LOB_IMPACT_GRID': str(tmp_path / 'new')}, monkeypatch)
    assert p.LOB_IMPACT_GRID == str(tmp_path / 'new')
