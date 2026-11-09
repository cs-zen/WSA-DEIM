import runpy
import sys
from pathlib import Path

import pytest

from engine.misc import dist_utils
from engine.solver import TASKS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ('arguments', 'expected_seed', 'expected_amp', 'expected_action'),
    [
        ([], 0, True, 'fit'),
        (['--seed', '7', '--use-amp'], 7, True, 'fit'),
        (['--test-only', '-u', 'seed=11', 'use_amp=False'], 11, False, 'val'),
    ],
)
def test_training_entry_respects_resolved_recipe(
    monkeypatch, arguments, expected_seed, expected_amp, expected_action
):
    observed = {}

    def setup_distributed(print_rank, print_method, seed):
        observed['seed'] = seed

    class Solver:
        def __init__(self, cfg):
            observed['cfg'] = cfg

        def fit(self):
            observed['action'] = 'fit'

        def val(self):
            observed['action'] = 'val'

    monkeypatch.setattr(dist_utils, 'setup_distributed', setup_distributed)
    monkeypatch.setattr(dist_utils, 'cleanup', lambda: observed.update(cleaned=True))
    monkeypatch.setitem(TASKS, 'detection', Solver)
    monkeypatch.setattr(
        sys, 'argv',
        ['train.py', '-c', str(PROJECT_ROOT / 'configs/wsa_deim_visdrone.yml'), *arguments],
    )
    runpy.run_path(str(PROJECT_ROOT / 'train.py'), run_name='__main__')
    assert observed['cfg'].seed == expected_seed
    assert observed['cfg'].use_amp is expected_amp
    assert observed['seed'] == expected_seed
    assert observed['cfg'].epoches == 300
    assert observed['action'] == expected_action
    assert observed['cleaned'] is True
