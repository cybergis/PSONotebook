import cost_funcs.info as info
import cost_funcs.standard as std


def test_bounds_eggholder():
    assert info.get_bounds(std.eggholder, 2)[0][0] == -512


def test_bounds_michal_dim():
    for dim in range(10):
        assert len(info.get_bounds(std.michal, dim)) == dim
