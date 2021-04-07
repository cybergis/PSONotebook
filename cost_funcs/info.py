import sys
sys.path.append("../")
from cost_funcs.JeonModelSimple import kang_simple  # noqa: E402
from cost_funcs.standard import *  # noqa: E402


def get_bounds(function, dim):
    if function == eggholder:
        return [(-512, 512), (-512, 512)]
    elif function == michal:
        bounds = []
        for i in range(dim):
            bounds.append((0, math.pi))
        return bounds
    elif function == paraboloid:
        return [(-100, 100), (-100, 100)]
    elif function == shubert:
        return [(-5.12, 5.12), (-5.12, 5.12)]
    elif function == rastrigin:
        bounds = []
        for i in range(dim):
            bounds.append((-5.12, 5.12))
        return bounds
    elif function == kang_simple:
        return [(0.0001, 0.1), (1, 4), (0.001, 1)]


def get_function_name(function):
    if function == eggholder:
        return "Eggholder Function"
    elif function == michal:
        return "Michalewicz Function"
    elif function == paraboloid:
        return "Paraboloid"
    elif function == shubert:
        return "Shubert Function"
    elif function == rastrigin:
        return "Rastrigin Function"
    elif function == kang_simple:
        return "Kang's Influenza ABM"


def get_global_minima(function, dim):
    if function == eggholder:
        return (-959.6407, (512, 404.2319))
    elif function == michal:
        if dim == 2:
            return (-1.8013, (2.20, 1.57))
        elif dim == 5:
            return (-4.687658, None)
        elif dim == 10:
            return (-9.66015, None)
    elif function == paraboloid:
        return (0, (0, 0))
    elif function == shubert:
        return (-186.7309, None)
    elif function == rastrigin:
        _x = []
        for i in range(dim):
            _x.append(0)
        return (0, _x)
    elif function == kang_simple:
        return (0, None)
