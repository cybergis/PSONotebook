import pytest
import cost_funcs.standard as std


def test_eggholder_global_min():
    assert std.eggholder({"x": [512, 404.2319]})[0] == pytest.approx(-959.6407, 0.001)


def test_michal_global_min_d2():
    assert std.michal({"x": [2.2, 1.57]})[0] == pytest.approx(-1.8013, 0.001)


def test_paraboloid_global_min():
    assert std.michal({"x": [0, 0]})[0] == pytest.approx(0, 0.001)


def test_rastrigin_global_min_d2():
    assert std.rastrigin({"x": [0, 0]})[0] == pytest.approx(0, 0.001)


def test_get_standard_funcs():
    assert len(std.get_standard_funcs()) == 6
