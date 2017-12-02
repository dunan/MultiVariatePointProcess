# PtPack: The C++ Multivariate Temporal Point Process Package
![Build Status](https://img.shields.io/teamcity/codebetter/bt428.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

PtPack is a C++ software library of high-dimensional temporal point processes. It aims to provide flexible modeling, learning, and inference of general multivariate temporal point processes to capture the latent dynamics governing the sheer volume of various temporal events arising from social networks, online media, financial trading, modern health-care, recommender systems, etc. Please check out the [project site](https://dunan.github.io/ptpack/html/index.html) for more details and documents.

## Prerequisites

- PtPack can be built on OS X and Linux.
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




