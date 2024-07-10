# Computing Model Comparison using PSIS-LOO

## Overview

We are often interested in comparing how well different models account for the same data. In this tutorial, we will compare models based on their expected log pointwise predictive density (ELPD), which we will estimate using Pareto smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).

First, we'll simulate data from a [`LBA`](@ref) with fixed parameters. Then, we'll define three LBA models with different fixed values for the `k` parameter. Finally, we'll use [ParetoSmooth.jl](https://docs.juliahub.com/General/ParetoSmooth/stable/) to perform model comparison via PSIS-LOO.

## Load Packages

Before proceeding, we will load the required packages.

```@example loo_example
using Random
using SequentialSamplingModels
using LinearAlgebra
using ParetoSmooth
using Turing
```

## Data-Generating Model

The next step is to generate simulated data for comparing the models. Here, we'll use an LBA as the true data-generating model:

```@example loo_example
Random.seed!(5000)

dist = LBA(ν=[3.0, 2.0], A=0.8, k=0.3, τ=0.3)
data = rand(dist, 1000)
```

## Specify Turing Models

The following code block defines the model along with its prior distributions using Turing.jl. We'll use this model with different fixed values for the `k` parameter.

```@example loo_example
@model function model_LBA(data, k; min_rt=minimum(data.rt))
    # Priors
    ν ~ MvNormal(fill(3.0, 2), I * 2)
    A ~ truncated(Normal(0.8, 0.4), 0.0, Inf)
    τ ~ Uniform(0.0, min_rt)

    # Likelihood
    data ~ LBA(; ν, A, k, τ)
end
```

## Estimate the Parameters

Now we'll estimate the parameters using three different fixed `k` values:

```@example loo_example
chain_LBA1 = sample(model_LBA(data, 2.0), NUTS(), 1000)
chain_LBA2 = sample(model_LBA(data, 0.3), NUTS(), 1000)
chain_LBA3 = sample(model_LBA(data, 1.0), NUTS(), 1000)
```

## Compute PSIS-LOO

Next we will use the `psis_loo` function to compute the PSIS-LOO for each model:

```@example loo_example
res1 = psis_loo(model_LBA(data, 2.0), chain_LBA1)
res2 = psis_loo(model_LBA(data, 0.3), chain_LBA2)
res3 = psis_loo(model_LBA(data, 1.0), chain_LBA3)
show(stdout, MIME"text/plain"(), ans)
```

## Compare Models

Finally, we can compare the models using the `loo_compare` function:

```@example loo_example
loo_compare((LBA1 = res1, LBA2 = res2, LBA3 = res3))
show(stdout, MIME"text/plain"(), ans)
```

Here we indeed correctly identified the generative model we simulated. It is of note that some researchers have criticized using model comparison metrics such as leave-one-out cross-validation. See Gronau et al. (2019) for more information.

# References

Gronau, Q. F., & Wagenmakers, E. J. (2019). Limitations of Bayesian leave-one-out cross-validation for model selection. Computational brain & behavior, 2, 1-11.

Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017).
