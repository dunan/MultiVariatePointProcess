# PtPack: The C++ Multivariate Temporal Point Process Package
![Build Status](https://img.shields.io/teamcity/codebetter/bt428.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

PtPack is a C++ software library of high-dimensional temporal point processes. It aims to provide flexible modeling, learning, and inference of general multivariate temporal point processes to capture the latent dynamics governing the sheer volume of various temporal events arising from social networks, online media, financial trading, modern health-care, recommender systems, etc. Please check out the [project site](http://www.cc.gatech.edu/grads/n/ndu8/ptpack/html/index.html) for more details and documents.

## Prerequisites

- PtPack can be built on OS X, Linux, and Windows.
- [gnuplot](http://www.gnuplot.info)

## Features

- Learning sparse interdependency structure of terminating point processes with applications in continuous-time information diffusions.

- Scalable continuous-time influence estimation and maximization.

- Learning multivariate Hawkes processes with different structural constraints, like: sparse, low-rank, customized triggering kernels, etc.

- Learning low-rank Hawkes processes for time-sensitive recommendations.

- Efficient simulation of standard multivariate Hawkes processes.

- Learning multivariate self-correcting processes.

- Simulation of customized general temporal point processes.

- Basic residual analysis and model checking of customized temporal point processes.

- Visualization of triggering kernels, intensity functions, and simulated events.

## Build static library

- cd MultiVariatePointProcess
- make

The built library will be saved under the directory build/lib

## Build examples

- cd MultiVariatePointProcess/example
- make

All built examples will be saved under the directory example/build

## Windows and gnuplot

On Windows, PtPack could be build with MinGW version 4.8 or above. It is recommended to have gnuplot installed in a directory that contains no spaces. Gnuplot binary location is hard-coded to be `/c/tools/gnuplot/bin/` for Windows. To change it, you may edit the code of `3rd-party/gnuplot/gnuplot_i.hpp` where the `Gnuplot::m_sGNUPlotPath` constant is set.

