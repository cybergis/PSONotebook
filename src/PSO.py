from cost_funcs.info import *
from PSOHelper import str2bool
import argparse
import csv
import copy
from datetime import datetime
import imageio
import itertools
import json
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import random
import sys
import logging as log
from os.path import join
import multiprocessing as mp
import numpy as np
matplotlib.use('agg')
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# logging.getLogger("imageio").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

"""Handles Ctrl+C"""


def signal_handler(sig, frame):
    sys.exit(0)


def pprint(data):
    print(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))


class Particle:
    def __init__(self, vmax, bounds):
        self.position_i = []          # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.err_best_i = -1          # best error individual
        self.err_i = -1               # error individual
        self.step = 0
        self.vmax = vmax
        self.num_dimensions = len(bounds)
        self.bounds = bounds
        self.outputs = []

        for i in range(self.num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(random.uniform(
                min(self.bounds[i]), max(self.bounds[i])))

    def copy_from(self, particle):
        self.position_i = copy.deepcopy(particle.position_i)
        self.velocity_i = copy.deepcopy(particle.velocity_i)
        self.pos_best_i = copy.deepcopy(particle.pos_best_i)
        self.err_best_i = particle.err_best_i
        self.err_i = particle.err_i

    # evaluate current fitness
    def evaluate(self, costFunc, func_sel):
        func_sel["x"] = self.position_i
        _to_evaluate = True
        for i in range(self.num_dimensions):
            # adjust maximum position if necessary
            if self.position_i[i] < self.bounds[i][0] or self.position_i[i] > self.bounds[i][1]:
                _to_evaluate = False
                break
        func_result = costFunc(func_sel) if _to_evaluate else (float("inf"), None)
        self.err_i = func_result[0]
        self.outputs = func_result[1]

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        c1 = 2.05        # cognitive constant
        c2 = 2.05        # social constant

        phi = c1 + c2
        k = 2.0 / abs(2 - phi - math.sqrt(phi ** 2 - 4 * phi))
        '''
        above code is the constriction factor due to Clerc:

        Clerc, M. (1999). The swarm and the queen: towards a deterministic and adaptive particle swarm optimization. Proc. I999 ICEC, Washington, DC, pp 195 1 - 1957.
        '''

        for i in range(len(pos_best_g)):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = k * (self.velocity_i[i] + vel_cognitive + vel_social)
        # velocity clamping
        _norm = math.sqrt(sum([i ** 2 for i in self.velocity_i]))
        if _norm > self.vmax:
            self.velocity_i = [self.vmax * i / _norm for i in self.velocity_i]

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(len(bounds)):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
        self.step += 1


def eval_pos_unpacker(args):
    return evaluate_position(* args)


def evaluate_position(particle, j, costFunc, func_sel):
    func_sel["particle_num"] = j
    # print("...in evaluate_position")
    # pprint(func_sel)
    particle.evaluate(costFunc, func_sel)
    return (j, particle)


class PSO():
    def __init__(self, costFunc, bounds, args):
        self.costFunc = costFunc
        self.bounds = bounds
        self.num_dimensions = len(bounds)

        self.function = args["function"]
        self.gif = str2bool(args['gif'])
        self.headless = str2bool(args['headless'])
        self.num_particles = args['particles']
        if args['seed'] != "None":
            random.seed(args['seed'])
        self.threads = args['threads']
        self.topology = args['topology']
        if args["max_velocity"] == "inf":
            self.vmax = float("inf")
        else:
            self.vmax = args['max_velocity']

        if 'dim_description' in args:
            if args["function"]['dim_description'] != "None":
                self.dim_description = args["function"]['dim_description']
        else:
            self.dim_description = None

        self.termination_criterion = args['termination']['termination_criterion']
        if self.termination_criterion == "iterations":
            self.max_iterations = args['termination']['max_iterations']
        else:
            raise Exception("Select a valid termination criterion")

        self.swarm_costs = []
        self.swarm_positions = []
        self.swarm_vel = []

        DATE = datetime.utcnow().strftime('%Y-%m-%d-b%H:%M:%S.%f')
        self.function_selection_string = ""
        if (len(self.function) > 1):
            for key, value in self.function.items():
                if key not in ["function", "dim_description"] and isinstance(value, str):
                    self.function_selection_string += "-{}:{}".format(
                        key, value)
        if not os.path.exists("../results/logs/"):
            pathlib.Path("../results/logs/").mkdir(parents=True, exist_ok=True)
        self.log_file = join('../results/logs/', 'swarmp-on_{}-({}particles--{}processes){}-{}-{}.log'.format(
            self.costFunc.__name__, self.num_particles, self.threads, self.function_selection_string, DATE, hex(random.getrandbits(24))))
        log.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S %p', filename=self.log_file, level=log.DEBUG)

        log.debug("...initializing swarm....")
        self.err_best_g = float("inf")                   # best error for group
        self.pos_best_g = []                   # best position for group

        # establish the swarm
        self.swarm = []
        for i in range(self.num_particles):
            self.swarm.append(Particle(self.vmax, copy.deepcopy(self.bounds)))
        log.debug("...swarm intialized:")
        for i in range(self.num_particles):
            log.debug("Particle {}: {}".format(i, self.swarm[i].position_i))
        # set the neighborhood graph
        self.set_neighbor_graph(self.topology)

        # outputs subdirectory to write to
        if not os.path.exists("../results/outputs"):
            pathlib.Path("../results/outputs").mkdir(parents=True, exist_ok=True)
        if args['seed'] == "None":
            self.output_path = os.path.join("../results/outputs", "{}-swarm-{}-opt-{}-({}particles-{}processes){}".format(
                args["output_dir"], DATE, self.costFunc.__name__, self.num_particles, self.threads,
                hex(random.getrandbits(24))))
        else:
            self.output_path = os.path.join("../results/outputs", "{}-swarm-{}-opt-{}-({}particles-{}processes){}s{}".format(
                args["output_dir"], DATE, self.costFunc.__name__, self.num_particles, self.threads,
                hex(random.getrandbits(24)), args['seed']))
        if os.path.exists(self.output_path):
            self.output_path = self.output_path + "_{}".format(hex(random.getrandbits(24)))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.csv_out = open(join(self.output_path, 'swarmp-on_{}-({}particles-{}processes){}-{}.csv'.format(
            self.costFunc.__name__, self.num_particles, self.threads, self.function_selection_string, DATE)), "w+")

        self.function["output_dir"] = self.output_path

        self.writer = csv.writer(self.csv_out)
        header_row = ["Iteration"]
        for j in range(self.num_particles):
            if self.dim_description is None:
                for k in range(self.num_dimensions):
                    header_row.append(
                        "Particle {}'s x[{}] Position".format(j, k))
            else:
                for k in range(self.num_dimensions):
                    header_row.append("Particle {}'s {}".format(
                        j, self.dim_description[k]))
            header_row.append("Particle {}'s Error".format(j))
        header_row.append("Average Error")
        self.writer.writerow(header_row)
        log.debug(self.dim_description)

        # begin optimization loop
        try:
            self.optimize()
        except Exception as e:
            log.exception(e)
            raise e

        result_descr = ["Best Error", ""]
        if self.dim_description is None:
            for k in range(self.num_dimensions):
                result_descr.append("x[{}] Best Position".format(k))
        else:
            for k in range(self.num_dimensions):
                result_descr.append(
                    "{} Best Position".format(self.dim_description[k]))
        self.csv_out.flush()
        self.csv_out.close()
        if not self.headless:
            print('FINAL:')
            print(self.pos_best_g)
            print(self.err_best_g)
        log.debug("Final:\n    Position: {}\n    Error: {}".format(
            self.pos_best_g, self.err_best_g))
        if self.gif:
            self.plot_surface()
            log.debug("Generating gif...")
            self.generate_gif()
        elif self.costFunc == kang_simple:
            self.plot_surface(dims=(1, 2), surface=False)
        log.debug("...closing handlers")
        logger = log.getLogger()
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    def set_neighbor_graph(self, topology):  # add more topologies
        self.neighbor_graph = np.zeros(
            (self.num_particles, self.num_particles))  # turn into numpy structure
        if topology == "fullyconnected":
            for i in range(self.num_particles):
                for j in range(self.num_particles):
                    self.neighbor_graph[i, j] = 1
        elif topology == "ring":
            for i in range(self.num_particles):
                for j in range(self.num_particles):
                    if j == i - 1 or j == i + 1:
                        self.neighbor_graph[i, j] = 1
                    elif i == 0 and j == self.num_particles - 1:
                        self.neighbor_graph[i, j] = 1
                    elif i == self.num_particles - 1 and j == 0:
                        self.neighbor_graph[i, j] = 1
        elif topology == "vonNeumann":
            """ https://doi.org/10.1109/CEC.2002.1004493 """
            n = self.num_particles
            r = math.floor(math.sqrt(n))
            for i in range(n):
                self.neighbor_graph[i, (i + 1) % n] = 1
                self.neighbor_graph[(i + 1) % n, i] = 1
                self.neighbor_graph[i, (i + r) % n] = 1
                self.neighbor_graph[(i + r) % n, i] = 1
        log.debug("\n{}".format(self.neighbor_graph))

    def finished(self, iteration):
        return iteration >= self.max_iterations  # add more end conditions

    def optimize(self):
        pool = mp.Pool(processes=self.threads)
        err_best_arr = []
        for i in range(self.num_particles):
            err_best_arr.append(float("inf"))
        pos_to_update = []
        for i in range(self.num_particles):
            pos_to_update.append([])
        iteration = 0
        if not self.headless:
            print("|-----+-------------------+-----------------------------------------")
            print("| Gen | Global Best Error | Global Best Pos ")
            print("|-----+-------------------+-----------------------------------------")
        while not self.finished(iteration):
            self.function["curr_best"] = self.err_best_g
            self.function["iteration"] = iteration
            if not self.headless:
                if self.err_best_g != float("inf"):
                    print("| {} | {: 17f} | {} ".format(
                        str(iteration).ljust(3), self.err_best_g, self.pos_best_g))
                else:
                    print("| {} | {} | {} ".format(str(iteration).ljust(
                        3), str(self.err_best_g).ljust(17), self.pos_best_g))
            log.debug("\n+++++ Beginning Iteration {} +++++".format(iteration))
            log.debug("    i: {}\n    err_best: {}\n    pos_best {}".format(iteration, self.err_best_g, self.pos_best_g))
            results = []
            log.debug("...entering pool mapping...")
            results = pool.map(eval_pos_unpacker, zip(self.swarm, range(self.num_particles), itertools.repeat(self.costFunc), itertools.repeat(self.function)))
            log.debug("...pool mapping exited...")
            results.sort()
            log.debug("...pool results sorted...")
            self.swarm = [r[1] for r in results]
            log.debug("...copying results back to main thread's copy...")
            runs = [self.swarm[i].outputs for i in range(len(self.swarm))]
            if self.costFunc == kang_simple:
                visualize_iteration(iteration, runs, city=self.function["city"], output_dir=self.function["output_dir"])
            log.debug("...evaluating fitness...")
            # cycle through particles in swarm and evaluate fitness
            for j in range(self.num_particles):
                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g:
                    self.pos_best_g = list(self.swarm[j].position_i)
                    self.err_best_g = float(self.swarm[j].err_i)
            self.swarm_costs.append(
                copy.deepcopy([particle.err_i for particle in self.swarm]))
            self.swarm_positions.append(
                copy.deepcopy([particle.position_i for particle in self.swarm]))
            self.swarm_vel.append(
                copy.deepcopy([particle.velocity_i for particle in self.swarm]))

            log.debug("New best error: {}\nNew best position: {}".format(
                self.err_best_g, self.pos_best_g))

            output_row = ["{}".format(iteration)]
            avg_err = 0
            for j in range(self.num_particles):
                for k in range(self.num_dimensions):
                    output_row.append(self.swarm[j].position_i[k])
                avg_err += self.swarm[j].err_i
                output_row.append(self.swarm[j].err_i)
            output_row.append(avg_err / self.num_particles)
            self.writer.writerow(output_row)
            self.csv_out.flush()
            log.debug("...updating particle position and velocity...")
            # cycle through swarm and update velocities and position
            for j in range(self.num_particles):
                for k in range(self.num_particles):
                    if self.neighbor_graph[j][k] == 1 and self.swarm[k].err_i < err_best_arr[j]:
                        pos_to_update[j] = list(self.swarm[k].position_i)
                        err_best_arr[j] = float(self.swarm[k].err_i)
            for j in range(self.num_particles):
                self.swarm[j].update_velocity(pos_to_update[j])
                self.swarm[j].update_position(self.bounds)
            if iteration > 0 and self.gif:
                self.plot_surface_this_timestep()
            iteration += 1
            self._number_of_iterations = iteration
        if not self.headless:
            print("|-----+-------------------+-----------------------------------------")
        if self.threads > 1:
            pool.close()
            pool.join()

    def plot_costs(self):
        plt.style.use('default')
        plt.plot(np.arange(len(self.swarm_costs)), [min(
            costs) for costs in self.swarm_costs], label="Min Error and Standard Deviation")
        plt.errorbar(np.arange(len(self.swarm_costs)), [(sum(costs) / len(costs)) for costs in self.swarm_costs], yerr=[
                     np.std(costs) for costs in self.swarm_costs], label="Mean Error and Standard Deviation", elinewidth=2, capsize=3)
        plt.legend(loc="upper right", frameon=True,
                   facecolor='wheat', framealpha=0.7)
        plt.xlabel("Iteration")
        plt.gca().set_xlim([-1, 72])
        plt.ylabel("Error")
        errors = [err for costs in self.swarm_costs for err in costs]
        plt.title("Error over Iterations", fontsize=16)
        textstr = '\n'.join((r"Min={:.8f}".format(min(errors)), r"$\mu$={:.2f}".format(
            np.mean(errors)), r"$\sigma$={:.2f}".format(np.std(errors))))
        plt.text(0.02, 0.18, textstr, fontsize=14, verticalalignment='top', transform=plt.gca(
        ).transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        fig.savefig('Swarm.png', dpi=250)
        # plt.show()
        plt.close('all')

    def generate_gif(self, frames_per_iter=1, end_pause=3):
        images = []
        for plot in range(2, self._number_of_iterations + 1):
            img = imageio.imread(
                "{}/PSO-T{}.png".format(self.output_path, plot))
            for i in range(frames_per_iter):
                images.append(img)
            if plot == self._number_of_iterations:
                for i in range(end_pause - frames_per_iter):
                    images.append(img)
        imageio.mimwrite('{}/movie.gif'.format(self.output_path), images)

    def plot_surface_this_timestep(self, dims=(0, 1), dim_labels=("x", "y"), granularity=200.0):
        X, Y = self.bounds[dims[0]], self.bounds[dims[1]]
        width_x, width_y = (max(X) - min(X)) / \
            granularity, (max(Y) - min(Y)) / granularity
        X, Y = np.arange(min(X), max(X), width_x), np.arange(
            min(Y), max(Y), width_y)
        Z = []
        for i, y in enumerate(Y):
            Z.append([])
            for j, x in enumerate(X):
                Z[i].append(self.costFunc({"x": [x, y]})[0])
        im = plt.imshow(np.flipud(Z), cmap=plt.cm.RdBu, extent=(
            min(X) - width_x / 2.0, max(X) + width_x / 2.0, min(Y) - width_y / 2.0, max(Y) + width_y / 2.0))
        plt.colorbar(im)
        X_this_step, Y_this_step = [pos[dims[0]] for pos in self.swarm_positions[-1]], [
            pos[dims[1]] for pos in self.swarm_positions[-1]]
        X_last_step, Y_last_step = [pos[dims[0]] for pos in self.swarm_positions[-2]], [
            pos[dims[1]] for pos in self.swarm_positions[-2]]
        plt.scatter([pos[dims[0]] for particle in self.swarm_positions for pos in particle], [
            pos[dims[1]] for particle in self.swarm_positions for pos in particle], color="yellow", alpha=0.8)
        arrow_color = ["green", "crimson", "sienna", "cyan", "indigo", "peru", "darkorange", "saddlebrown", "lawngreen",
                       "darkgoldenrod", "magenta", "deeppink", "darkviolet", "navy", "lightcoral", "orangered", "darkgreen", "teal", "aqua"]
        for i in range(self.num_particles):
            x, y = [X_last_step[i], X_this_step[i]], [
                Y_last_step[i], Y_this_step[i]]
            u, v = np.diff(x), np.diff(y)
            pos_x, pos_y = x[:-1] + u / 2, y[:-1] + v / 2
            norm = np.sqrt(u ** 2 + v ** 2)
            plt.plot(x, y, color=arrow_color[i % len(arrow_color)], alpha=0.7)
            plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy",
                       pivot="mid", color=arrow_color[i % len(arrow_color)])
        plt.scatter([pos[dims[0]] for pos in self.swarm_positions[0]], [pos[dims[1]]
                    for pos in self.swarm_positions[0]], color="black", edgecolors="yellow")
        plt.xlabel(dim_labels[0])
        plt.ylabel(dim_labels[1])
        plt.title("PSO on {}\nt: {} - min: {:.4f} / true min: {}".format(get_function_name(self.costFunc), len(self.swarm_positions),
                  min([item for sublist in self.swarm_costs for item in sublist]), get_global_minima(self.costFunc, len(self.bounds))[0]), fontsize=10)
        axes = plt.gca()
        axes.set_xlim([min(X) - width_x, max(X) + width_x])
        axes.set_ylim([min(Y) - width_y, max(Y) + width_y])
        fig = plt.gcf()
        fig.set_size_inches(8, 6.7)
        plt.tight_layout()
        # plt.show()
        plt.savefig("{}/PSO-T{}.png".format(self.output_path,
                    len(self.swarm_positions)))
        fig.clf()

    def plot_surface(self, dims=(0, 1), dim_labels=("x", "y"), granularity=200.0, surface=True, arrows=True):
        log.debug("plotting surface")
        X, Y = self.bounds[dims[0]], self.bounds[dims[1]]
        width_x, width_y = (max(X) - min(X)) / \
            granularity, (max(Y) - min(Y)) / granularity
        X, Y = np.arange(min(X), max(X), width_x), np.arange(
            min(Y), max(Y), width_y)
        if surface:
            Z = []
            for i, x in enumerate(X):
                Z.append([])
                for j, y in enumerate(Y):
                    Z[i].append(self.costFunc({"x": [y, x]})[0])
            im = plt.imshow(np.flipud(Z), cmap=plt.cm.RdBu, extent=(
                min(X) - width_x / 2.0, max(X) + width_x / 2.0, min(Y) - width_y / 2.0, max(Y) + width_y / 2.0))
            plt.colorbar(im)
        X, Y = [pos[dims[0]] for particle in self.swarm_positions for pos in particle], [
            pos[dims[1]] for particle in self.swarm_positions for pos in particle]
        plt.scatter(X[self.num_particles:],
                    Y[self.num_particles:], color="yellow", alpha=0.5)
        if arrows:
            arrow_color = ["green", "crimson", "sienna",
                           "cyan", "indigo", "peru", "darkorange"]
            arrow_width = width_x * 0.005
            for i in range(self.num_particles):
                x, y = X[i::self.num_particles], Y[i::self.num_particles]
                u, v = np.diff(x), np.diff(y)
                pos_x, pos_y = x[:-1] + u / 2, y[:-1] + v / 2
                norm = np.sqrt(u ** 2 + v ** 2)
                plt.plot(x, y, color=arrow_color[i %
                         len(arrow_color)], alpha=0.7)
                plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy",
                           pivot="mid", color=arrow_color[i % len(arrow_color)], width=arrow_width)
        plt.scatter(X[:self.num_particles], Y[:self.num_particles],
                    color="black", edgecolors="yellow")
        plt.xlabel(dim_labels[0])
        plt.ylabel(dim_labels[1])
        plt.title("PSO on {}\nt: {} - min: {:.4f} / true min: {}".format(get_function_name(self.costFunc), len(self.swarm_positions),
                  min([item for sublist in self.swarm_costs for item in sublist]), get_global_minima(self.costFunc, len(self.bounds))[0]), fontsize=10)
        axes = plt.gca()
        axes.set_xlim([min(X), max(X)])
        axes.set_ylim([min(Y), max(Y)])
        fig = plt.gcf()
        fig.set_size_inches(12, 10)
        plt.tight_layout()
        # plt.show()
        plt.savefig("{}/surface.png".format(self.output_path))
        fig.clf()


def setup_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', type=str, required=True,
                        dest='params', help='Path to JSON file with parameters')
    parser.add_argument('-o', type=str, required=False,
                        dest='output_prefix', help='Prefix to the output directory')
    parser.add_argument('-s', type=int, required=False,
                        dest='random_seed', help='Seed for a random number generator')
    return parser.parse_args()


def main():
    parser = setup_parser()

    if parser.params:
        param_path = parser.params
        print("...loading parameters from {}...".format(param_path))
    else:
        param_path = 'params.json'

    with open(os.path.join(param_path)) as json_file:
        args = json.load(json_file)
    if parser.output_prefix is not None:
        args["output_dir"] = parser.output_prefix
    if parser.random_seed is not None:
        args["seed"] = parser.random_seed

    print(json.dumps(args, indent=4, sort_keys=True))

    print("Swarming with {} particles on {} threads using {} termination criterion with {} vmax".format(
        args['particles'], args['threads'], args['termination']['termination_criterion'], args['max_velocity']))
    if args["function"]["function"] == "eggholder":
        return PSO(eggholder, get_bounds(eggholder, 2), args)
    elif args["function"]["function"] == "michal":
        return PSO(michal, get_bounds(michal, 2), args)
    elif args["function"]["function"] == "noisey_paraboloid":
        return PSO(noisey_paraboloid, get_bounds(noisey_paraboloid, 2), args)
    elif args["function"]["function"] == "paraboloid":
        return PSO(paraboloid, get_bounds(paraboloid, 2), args)
    elif args["function"]["function"] == "shubert":
        return PSO(shubert, get_bounds(shubert, 2), args)
    elif args["function"]["function"] == "rastrigin":
        return PSO(rastrigin, get_bounds(rastrigin, args["function"]["dimension"]), args)
    elif args["function"]["function"] == "kang_simple":
        return PSO(kang_simple, get_bounds(kang_simple, 3), args)
    else:
        print("Please use a valid cost function for cost_func")


if __name__ == "__main__":
    main()
