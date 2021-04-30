import cost_funcs.info as info
import PSO
import copy
import os


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
    PSO.PSO(info.paraboloid, info.get_bounds(info.paraboloid, 2), copy.deepcopy(simple_params))


def test_csv_out_exists():
    pso = PSO.PSO(info.paraboloid, info.get_bounds(info.paraboloid, 2), copy.deepcopy(simple_params))
    assert os.path.isfile(pso.csv_out.name)


def test_gif_simple():
    my_params = copy.deepcopy(simple_params)
    my_params["gif"] = True
    my_params["termination"]["max_iterations"] = 10
    PSO.PSO(info.paraboloid, info.get_bounds(info.paraboloid, 2), my_params)


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


def test_neighborhood_graph_ring():
    for part in range(4, 20, 4):
        my_params = copy.deepcopy(simple_params)
        my_params["particles"] = part
        my_params["topology"] = "ring"
        pso = PSO.PSO(info.paraboloid, info.get_bounds(info.paraboloid, 2), my_params)
        graph = pso.neighbor_graph
        column_sums = graph.sum(axis=0)
        for column_sum in column_sums:
            assert column_sum == 2


def test_neighborhood_graph_vonNeumann():
    for part in range(8, 20, 4):
        my_params = copy.deepcopy(simple_params)
        my_params["particles"] = part
        my_params["topology"] = "vonNeumann"
        pso = PSO.PSO(info.paraboloid, info.get_bounds(info.paraboloid, 2), my_params)
        graph = pso.neighbor_graph
        column_sums = graph.sum(axis=0)
        for column_sum in column_sums:
            assert column_sum == 4
