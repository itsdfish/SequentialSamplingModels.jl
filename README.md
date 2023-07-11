[![](docs/logo/logo.png)](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/) [![CI](https://github.com/itsdfish/SequentialSamplingModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/itsdfish/SequentialSamplingModels.jl/actions/workflows/CI.yml)

# SequentialSamplingModels

This package provides a unified interface for sequential sampling models (such as DDM, LBA, LNR, LCA, ...) in Julia, based on the Distributions.jl API, that can be used within [**Turing**](https://turing.ml/) framework for Bayesian estimation.

Please see the [documentation](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/) for more information.

- [**Bayesian Modeling: Usage with Turing**](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/turing/)
- [**Developer Guide**](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/developer_guide/)

## Installation

You can install a stable version of *SequentialSamplingModels* by running the following in the Julia REPL:

```julia
] add SequentialSamplingModels
```

## Quick Example

In the example below, we instantiate a Linear Ballistic Accumulator (LBA) model, and generate data from it.

```julia
using SequentialSamplingModels
using StatsPlots
using Random

# Create LBA distribution with known parameters
dist = LBA(; ν=[2.75,1.75], A=0.8, k=0.5, τ=0.25)
# Sample 1000 random data points from this distribution
choice, rt = rand(dist, 1000)
```