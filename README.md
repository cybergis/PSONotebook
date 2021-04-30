[![PEP8](https://github.com/cybergis/PSONotebook/actions/workflows/PEP8.yml/badge.svg)](https://github.com/cybergis/PSONotebook/actions/workflows/PEP8.yml)
[![Pytest](https://github.com/cybergis/PSONotebook/actions/workflows/Pytest.yml/badge.svg)](https://github.com/cybergis/PSONotebook/actions/workflows/Pytest.yml)
![GitHub](https://img.shields.io/github/license/cybergis/PSONotebook?style=plastic)

# Particle Swarm Optimization for Calibration in Spatially Explicit Agent-Based Modeling

**Authors:** [Alexander Michels](https://scholar.google.com/citations?user=EbmZrwYAAAAJ), [Jeon-Young Kang](https://scholar.google.com/citations?user=u5cevWAAAAAJ), [Shaowen Wang](https://scholar.google.com/citations?user=qcUhJIcAAAAJ)

Special thanks to [Zhiyu Li](https://scholar.google.com/citations?user=yskFOAgAAAAJ) and [Rebecca Vandewalle](https://scholar.google.com/citations?user=1WzQbAgAAAAJ) for suggestions and feedback on this notebook!

[Open with CyberGISX](https://cybergisx.cigi.illinois.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fcybergis%2FPSONotebook&urlpath=tree%2FPSONotebook%2FIntroToParticleSwarmOptimization.ipynb&branch=master)

![Gif of Particle Swarm Optimization](img/movie.gif)

This notebook is related to an upcoming publication entitled "Particle Swarm Optimization for Calibration in Spatially Explicit Agent-Based Modeling." The abstract for the paper is:

>A challenge in computational modeling of complex geospatial systems is the amount of time and resources required to tune a set of parameters that reproduces the observed patterns of phenomena of being modeled. Well-tuned parameters are necessary for models to reproduce real-world multi-scale space-time patterns, but calibration is often computationally-intensive and time-consuming. Particle Swarm Optimization (PSO) is a swarm intelligence optimization algorithm that has found wide use for complex optimization including non-convex and noisy problems. In this study, we propose to use PSO for calibrating parameters in spatially explicit agent-based models (ABMs). We use a spatially explicit ABM of influenza transmission based in Miami, Florida, USA as a case study. Further, we demonstrate that a standard implementation of PSO can be used out-of-the-box to successfully calibrate models and out-performs Monte Carlo in terms of optimization and efficiency.

[The notebook](code/IntroToParticleSwarmOptimization.ipynb) is designed to teach you about Particle Swarm Optimization (PSO) and how you can use it for parameter optimization. Particle Swarm Optimization (PSO) was first introduced in [1995 by James Kennedy and Russell Eberhart](https://doi.org/10.1109/ICNN.1995.488968). The algorithm began as a simulation of social flocking behaviors like those exhibited by flocks of birds and schools of fish, specifically of birds searching a cornfield, but was found to be useful for training feedforward multilayer pernceptron neural networks. Since then, PSO has been adapted in a variety of ways and applied to problems including [wireless-sensor networks](https://doi.org/10.1109/TSMCC.2010.2054080), [classifying biological data](https://doi.org/10.1109/SIS.2005.1501608), [scheduling workflow applications in cloud computing environments](https://doi.org/10.1109/AINA.2010.31), [Image classification](https://doi.org/10.1109/ICIP.2006.312968) and [power systems](https://doi.org/10.1109/TEVC.2007.896686). In this notebook we explore PSO's usefulness for calibration, with a focus on spatially-explicit agent-based models (ABMs).

The model is also available on CoMSES, you can find it by clicking on the badge below:


[![Open Code Badge](https://www.comses.net/static/images/icons/open-code-badge.png)](https://www.comses.net/codebases/b136ee71-22e7-410f-b810-1b39525f9919/releases/1.0.0/)

