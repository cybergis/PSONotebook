import warnings
from cost_funcs.standard import *
from cost_funcs import Env, Human
from PSOHelper import *
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('agg')

sys.path.append('../')

# gets rid of matplotlib font warnings in the logging
warnings.filterwarnings("ignore")


def JeonModelSimple(introRate=0.002, reproduction=2.4, infRate=0.18, city="Miami", days=148, metric="rmse"):
    DATA_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data/{}".format(city))
    try:  # try to open the preprocessed data
        with open("{}/schoolList.pickle".format(DATA_PATH), "rb") as f:
            schoolList = pickle.load(f)
    except:  # if data does not exist, write it
        schoolList = Env.settingSchools('{}'.format(DATA_PATH))
        with open("{}/schoolList.pickle".format(DATA_PATH), "wb") as f:
            pickle.dump(schoolList, f)
    try:
        with open("{}/workList.pickle".format(DATA_PATH), "rb") as f:
            workList = pickle.load(f)
    except:
        workList = Env.settingWorks('{}'.format(DATA_PATH))
        with open("{}/workList.pickle".format(DATA_PATH), "wb") as f:
            pickle.dump(workList, f)
    try:
        with open("{}/houseList.pickle".format(DATA_PATH), "rb") as f:
            houseList = pickle.load(f)
    except:
        houseList = Env.settingHouseholds('{}'.format(
            DATA_PATH), schoolList, workList, houses=houses)
        with open("{}/houseList.pickle".format(DATA_PATH), "wb") as f:
            pickle.dump(houseList, f)
    try:
        with open("{}/peopleList.pickle".format(DATA_PATH), "rb") as f:
            peopleList = pickle.load(f)
    except:
        peopleList = Human.settingHumanAgent(houseList)
        with open("{}/peopleList.pickle".format(DATA_PATH), "wb") as f:
            pickle.dump(peopleList, f)
    # the list for counting infectious people over simulation days.
    infectiousCount = []
    # initial infections based on the introRate
    peopleList = Env.initialInfection(peopleList, introRate)
    for t in range(1, days):
        exposedNum = 0
        infectiousNum = 0
        for p in range(len(peopleList)):
            person = peopleList[p]
            person.incubating()
            person.recovering()
            if (person.I is True):
                infectiousNum += 1
            if (person.E is True):
                exposedNum += 1
        infectiousCount.append(infectiousNum)
        for p in peopleList:
            # infecting function is included in Human.py
            p.infecting(peopleList, infRate, reproduction, t)
    # load the observed data
    observed, resolution = get_observations('{}/flu_observations.csv'.format(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/{}".format(city))))
    model_for_rmse = normalize(infectiousCount, _sum=700)
    model_for_rmse, actual = match_resolutions(
        model_for_rmse, observed, resolution)
    actual = normalize(actual)
    score = 0
    if metric == "rmse":
        score = rmse(model_for_rmse, actual)   # calculate RMSE
    elif metric == "max_by_rmse":
        score = max_by_rmse(model_for_rmse, actual)
    return (score, infectiousCount)


"""
Keep models above this line and metrics/helpers below
"""


def normalize(array, _sum=100.0, resolution=1):
    ''' normalizes array so it sums to _sum '''
    _array_sum = sum(array[(resolution - 1)::resolution])
    if _array_sum > 0:
        return np.asarray([(_sum * float(i)) / _array_sum for i in array])
    return np.asarray(array)

# read in observations and determine the frequency


def get_observations(path):
    actual = pd.read_csv(path)
    res = None
    if "Weekly" in list(actual.columns.values):
        actual = np.asarray(actual["Weekly"].tolist()[1:])
        res = 7
    elif "Daily" in list(actual.columns.values):
        actual = np.asarray(actual["Daily"].tolist()[1:])
        res = 1
    return actual, res


# add back interpolation here if needed
def match_resolutions(model, actual, resolution):
    model = model[(resolution - 1)::resolution]
    _min = min(len(model), len(actual))
    return model[:_min], actual[:_min]


def match_length_runs(runs, actual):  # add back interpolation here if needed
    _min = 7 * len(actual)
    for particle in range(len(runs)):
        if runs[particle] is not None:
            if len(runs[particle]) < _min:
                _min = len(runs[particle])
    for run in range(len(runs)):
        if not runs[run] is None:
            runs[run] = runs[run][:_min]
    return runs, actual[:_min]


def visualize_iteration(iteration_number, runs, city="QueenAnne", output_dir=""):
    DATA_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data/{}".format(city))
    OUTPUT_PATH = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../{}".format(output_dir))
    actual, resolution = get_observations(
        '{}/flu_observations.csv'.format(DATA_PATH))
    # match resolution and length of arrays, then normalize
    runs, actual = match_length_runs(runs, actual)
    for particle in range(len(runs)):
        if not runs[particle] is None:
            runs[particle] = normalize(
                runs[particle][:7 * len(actual)], _sum=700)
    actual = normalize(actual)
    line_types = ["-", "--", "-.", ":"]
    colors = ["g", "r", "c", "m", "y", "k"]
    for particle in range(len(runs)):
        if not runs[particle] is None:
            _x = range(1, len(runs[particle]) + 1)
            plt.plot(_x, runs[particle], "{}{}".format(colors[(particle % len(colors))], line_types[(
                particle // len(colors)) % len(line_types)]), label="Particle {}".format(particle))  # gives particle different colors
    plt.plot(range(1, len(actual) * 7 + 6)
             [6::7][:len(actual)], actual, 'bo', label="Observed")
    plt.title('infections throughout the flu season', size=20)
    plt.xlabel("days", size=15)
    plt.ylabel("proportion of people", size=15)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    fig = plt.gcf()
    fig.set_size_inches(8, 5.5)
    fig.savefig('{}/Iteration-{}.png'.format(OUTPUT_PATH,
                iteration_number), dpi=250)
    plt.clf()  # clear figure


def kang_simple(args):
    x = args["x"]
    return JeonModelSimple(introRate=x[0], reproduction=x[1], infRate=x[2], city=args["city"], days=args["days"], metric=args["metric"])


def main():
    print(JeonModelSimple())


if __name__ == "__main__":
    main()
