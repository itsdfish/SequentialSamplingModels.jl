# Computing Model Comparison using PSIS-LOO

## Overview

We are often interested in comparing how well different models account for the same data. In this tutorial, we will compare models based on their expected log pointwise predictive density (ELPD), which we will estimate using Pareto smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).

First, we'll simulate data from a LBA with fixed parameters. Then, we'll define three LBA models with different fixed values for the `k` parameter. Finally, we'll use [ParetoSmooth.jl](https://docs.juliahub.com/General/ParetoSmooth/stable/) to perform model comparison via PSIS-LOO.

## Load Packages

Before proceeding, we will load the required packages.

```julia
using Random
using SequentialSamplingModels
using LinearAlgebra
using ParetoSmooth
using Turing
```

## Data-Generating Model

The next step is to generate simulated data for comparing the models. Here, we'll use an LBA as the true data-generating model:

```julia
Random.seed!(5000)

dist = LBA(ν=[3.0, 2.0], A=0.8, k=0.3, τ=0.3)
data = rand(dist, 1000)
```

## Specify Turing Models

The following code block defines the model along with its prior distributions using Turing.jl. We'll use this model with different fixed values for the `k` parameter.

```julia
@model function model_LBA(data, k; min_rt=0.2)
    # Priors
    ν ~ MvNormal(fill(3.0, 2), I * 2)
    A ~ truncated(Normal(0.8, 0.4), 0.0, Inf)
    τ ~ Uniform(0.0, min_rt)

    # Likelihood
    for i in 1:length(data)
        data[i] ~ LBA(; ν, A, k, τ)
    end
end
```

## Estimate the Parameters

Now we'll estimate the parameters using three different fixed `k` values:

```julia
#prepare the data for model fitting
dat = [(choice=data.choice[i], rt=data.rt[i]) for i in 1:length(data.rt)]
min_rt = minimum(data.rt)

chain_LBA1 = sample(model_LBA(dat, 2,  min_rt=min_rt), NUTS(), 1000)
chain_LBA2 = sample(model_LBA(dat, 0.3,min_rt=min_rt), NUTS(), 1000)
chain_LBA3 = sample(model_LBA(dat, 1,  min_rt=min_rt), NUTS(), 1000)

```

## Compute PSIS-LOO

Next we will use the `psis_loo` function to compute the PSIS-LOO for each model:

```julia
res1 = psis_loo(model_LBA(dat, 2,   min_rt=min_rt), chain_LBA1)
res2 = psis_loo(model_LBA(dat, 0.3, min_rt=min_rt), chain_LBA2)
res3 = psis_loo(model_LBA(dat, 1, min_rt=min_rt), chain_LBA3)
```

```julia 
┌───────────┬────────┬──────────┬───────┬─────────┐
│           │  total │ se_total │  mean │ se_mean │
├───────────┼────────┼──────────┼───────┼─────────┤
│   cv_elpd │ 307.70 │    32.23 │  0.31 │    0.03 │
│ naive_lpd │ 310.53 │    32.12 │  0.31 │    0.03 │
│     p_eff │   2.83 │     0.14 │  0.00 │    0.00 │
└───────────┴────────┴──────────┴───────┴─────────┘
```

## Compare Models

Finally, we can compare the models using the `loo_compare` function:

```julia
loo_compare((LBA1=res1, LBA2=res2, LBA3 = res3))
```

```julia
┌──────┬──────────┬────────┬────────┐
│      │  cv_elpd │ cv_avg │ weight │
├──────┼──────────┼────────┼────────┤
│ LBA2 │     0.00 │   0.00 │   1.00 │ #correct
│ LBA1 │   -34.58 │  -0.03 │   0.00 │ #incorrect
│ LBA3 │ -4034.13 │  -4.03 │   0.00 │ #incorrect
└──────┴──────────┴────────┴────────┘
```

Here we indeed correctly identified the generative model we simulated. It is of note that some researchers have criticized using model comparison metrics such as leave-one-out cross-validation. See Gronau et al. (2019) for more information.

# References

Gronau, Q. F., & Wagenmakers, E. J. (2019). Limitations of Bayesian leave-one-out cross-validation for model selection. Computational brain & behavior, 2, 1-11.

Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017).
