import cost_funcs.info as info
import PSO
import copy

simple_params = {
    "function": {
        "dim_description": "None",
        "dimension": 2,
        "function": "paraboloid"
    },
    "gif": "False",
    "headless": "True",
    "max_velocity": "inf",
    "metric": "rmse",
    "output_dir": "",
    "particles": 1,
    "seed": "None",
    "termination": {
        "max_iterations": 3,
        "termination_criterion": "iterations"
    },
    "threads": 1,
    "topology": "fullyconnected"
}


def test_instantiate():
    pso = PSO.PSO(info.paraboloid, info.get_bounds(info.paraboloid, 2), copy.deepcopy(simple_params))
    del pso


def test_neighborhood_graph_fullyconnected():
    for part in range(4, 20, 4):
        my_params = copy.deepcopy(simple_params)
        my_params["particles"] = part
        my_params["topology"] = "fullyconnected"
        pso = PSO.PSO(info.paraboloid, info.get_bounds(info.paraboloid, 2), my_params)
        graph = pso.neighbor_graph
        column_sums = graph.sum(axis=0)
        for column_sum in column_sums:
            assert column_sum == part
        del pso
