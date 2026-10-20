"""The lob-impact command (lob_impact.__main__): lists the tools, runs one by name."""
from lob_impact.__main__ import TOOLS, main


def test_without_a_tool_it_lists_them(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in TOOLS)


def test_a_tool_runs_by_name_with_hyphens_or_underscores(capsys):
    assert main(['paths']) == 0
    assert 'LOB_IMPACT_RESULTS' in capsys.readouterr().out


def test_an_unknown_tool_is_refused(capsys):
    assert main(['nosuch']) == 2
    assert 'no tool' in capsys.readouterr().err
