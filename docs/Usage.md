# How to Use

The notebook located in `code/IntroToParticleSwarmOptimization.ipynb` is the best source to learn about the code and how to use it.


## Parameters

Here is a brief description of the parameters: 

* `function` gives information on the function to optimize.
  * `function/dim_description` can be a list of labels for the different dimensions. For example with the SEIR model we used `["introRate","reproduction","infRate"]`. If nothing is supplied, we use x\[0\],...x\[n\].
  * `function/dimension` gives the number of dimensions in the parameter space (the number of parameters).
  * `function/function` specifies the function name in text. This is used by `PSO.py`'s `main` function to determine the correct function to optimize.
  * you can also pass any other amount of information in this `function` dictionary in case your function/model needs other information! See [Adding Your Own Model](#customize) to learn more about adding your own model.
* `gif` is a boolean that specifies whether or not you want to make a gif of the algorithm run. **SET TO FALSE FOR COMPUTATIONALLY INTENSIVE FUNCTIONS OR MODEL** because it uses 40,000 evaluations per iteration to plot the error surface.
* `headless` is a boolean. `True` minimizes the amount printed to the terminal.
* `max_velocity` is used to specify the maximum velocity. See the velocity determination information above for more info.
* `output_dir` specifies the path you'd like to store outputs in, this is joined to "./outputs". So for example if you specified "test", the outputs would be stored in "./outputs/test/". This is useful for grouping together runs of the algorithm.
* `particles` gives the number of particles.
* `seed` is the seed for a random number generator. "None" uses no seed.
* `termination` allows us to specify when the algorithm should terminate.
  * `termination/termination_criterion` specifies how the PSO algorithm knows when to stop. `iterations` is the only supported option, but you're welcome to expand on this code and test others!
  * `termination/max_iterations` is necessary for the `iterations` termination criterion and gives the number of generations that PSO should go through.
* `threads` is the number of threads used by the `multiprocessing` pool.
* `topology` gives the topology of the particle communication network. See above for more info!

Some of these parameters may seem needless (like termination criterion), but the code was designed to be extensible so I tried to avoid hard-coding values/choices where possible. An example of parameters in `code/test.json`. Parameters are passed to the PSO object as a dictionary or using the `--params` flag on the command line.


## CLI Usage

Because the vast majority of parameters are passed through the parameters JSON file, all you have to do to run the algorithm is:

```
python3 PSO.py --params test.json
```

Other optional command line options include:

* `-o` which will overwrite the `output_dir` parameter
* `-s` which will overwrite the `seed` parameter

If on the command line and you aren't sure, you can use the `--help` flag:

```
> python3 PSO.py --help

usage: PSO.py [-h] --params PARAMS [-o OUTPUT_PREFIX] [-s RANDOM_SEED]

optional arguments:
  -h, --help        show this help message and exit
  --params PARAMS   Path to JSON file with parameters (default: None)
  -o OUTPUT_PREFIX  Prefix to the output directory (default: None)
  -s RANDOM_SEED    Seed for a random number generator (default: None)
```

The code for this is in the bottom of the `main` function of `code/PSO.py` and can be easily changed if you'd like to add more CLI options.