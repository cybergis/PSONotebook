import cost_funcs.info as info
import cost_funcs.standard as std
import pytest


def test_bounds_eggholder():
    assert info.get_bounds(std.eggholder, 2)[0][0] == -512


def test_bounds_michal_dim():
    for dim in range(10):
        assert len(info.get_bounds(std.michal, dim)) == dim


def test_function_name_eggholder():
    assert info.get_function_name(std.eggholder) == "Eggholder Function"


def test_global_minima_eggholder():
    assert info.get_global_minima(std.eggholder, 2)[0] == pytest.approx(-959.6407, 0.001)
