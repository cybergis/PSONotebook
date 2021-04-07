import PSOHelper   # noqa: E402


def test_str2bool_bool():
    assert PSOHelper.str2bool(True) is True
    assert PSOHelper.str2bool(False) is False


def test_str2bool_string():
    assert PSOHelper.str2bool("True") is True
    assert PSOHelper.str2bool("tRuE") is True
    assert PSOHelper.str2bool("true") is True
    assert PSOHelper.str2bool("TRUE") is True
    assert PSOHelper.str2bool("FALSE") is False
    assert PSOHelper.str2bool("false") is False
    assert PSOHelper.str2bool("fAlSe") is False
    assert PSOHelper.str2bool("faLsE") is False


def test_rmse_same_len_order_one():
    for length in range(10):
        first, second = [-1] * length, [1] * length
        assert PSOHelper.rmse(first, second, order=2) == 2 * length
