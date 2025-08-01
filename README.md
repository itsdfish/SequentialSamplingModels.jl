[![](docs/logo/logo.png)](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/) [![CI](https://github.com/itsdfish/SequentialSamplingModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/itsdfish/SequentialSamplingModels.jl/actions/workflows/CI.yml) [![CodeCov][codecov-img]][codecov-url] 

[codecov-img]: https://codecov.io/github/itsdfish/SequentialSamplingModels.jl/badge.svg?branch=master
[codecov-url]: https://codecov.io/github/itsdfish/SequentialSamplingModels.jl?branch=master

# SequentialSamplingModels

This package provides a unified interface for simulating and evaluating sequential sampling models (SSMs) in Julia. SSMs describe decision making as a stochastic and dynamic evidence accumulation process in which a decision is triggered by the option whose evidence hits a decision treshold first. 

![](docs/src/assets/rdm.gif)

# Feature Overview

A summary of the core features is provided below. Please see the [documentation](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/) for more information.

## Supported Models
The following SSMs are supported:

### Single Choice Models 
- [Ex-Gaussian](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/ex_gaussian/)
- [Shifted Log Normal](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/shifted_lognormal/)
- [Wald](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/wald/) 

### Multi-Choice Models 
#### Single Attribute
- [Attentional Drift Diffusion](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/aDDM/)
- [Leaky Competing Accumulator](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lca/)
- [Drift Diffusion](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/DDM/)
- [Linear Ballistic Accumulator](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) 
- [Log Normal Race](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lnr/) 
- [Poisson Race](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/poisson_race)
- [Racing Diffusion](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/rdm/) 

#### Multi-Attribute 

- [Multi-attribute Attentional Drift Diffusion](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/maaDDM/)
- [Multi-attribute Decision Field Theory](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/mdft/)
- [Multi-attribute Linear Ballistic Accumulator](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/mlba/)

### Alternative Geometries 

- [Circular Drift Diffusion](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/cddm/) 

## API

The core API consists of the following 

- rand: generate simulated data 
- pdf: evaluate the probability density of the data
- logpdf: evaluate the log probability density of the data
- simulate: generate samples from the internal evidence accumulation process

## Ecosystem Integration

SSMs work with the following packages (and possibly more):

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl): functions for probability distributions
- [Pigeons.jl](http://pigeons.run/dev/): Bayesian parameter estimation and Bayes factors
- [Plots.jl](https://github.com/JuliaPlots/Plots.jl): extended plotting tools for SSMs
- [Turing.jl](https://turinglang.org/dev/docs/using-turing/get-started): Bayesian parameter estimation

## Installation

You can install a stable version of *SequentialSamplingModels* by running the following in the Julia REPL:

```julia
] add SequentialSamplingModels
```

## Quick Example

In the example below, we instantiate a Linear Ballistic Accumulator (LBA) model, and generate data from it.

```julia
using SequentialSamplingModels

# Create LBA distribution with known parameters
dist = LBA(; ν=[2.75,1.75], A=0.8, k=0.5, τ=0.25)
# Sample 1000 random data points from this distribution
choice, rt = rand(dist, 1000)
```