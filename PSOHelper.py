import argparse
import json
import math


def eval_pos_unpacker(args):
    return evaluate_position(*args)


def evaluate_position(particle, j, costFunc, func_sel):
    func_sel["particle_num"] = j
    particle.evaluate(costFunc, func_sel)
    return (j, particle)


def pprint(data):
    print(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))


def rmse(first, second, order=2):
    _len = min(len(first), len(second))
    diffs = 0
    for i in range(_len):
        diffs += math.sqrt((first[i] - second[i])**order)
    return diffs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'None'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
