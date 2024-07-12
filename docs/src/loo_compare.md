# Computing Model Comparison using PSIS-LOO

## Overview

We are often interested in comparing how well different models account for the same data. In this tutorial, we will compare models based on their expected log pointwise predictive density (ELPD), which we will estimate using Pareto smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).

First, we'll simulate data from a [`LBA`](@ref) with fixed parameters. Then, we'll define three LBA models with different fixed values for the `k` parameter. Finally, we'll use [ParetoSmooth.jl](https://docs.juliahub.com/General/ParetoSmooth/stable/) to perform model comparison via PSIS-LOO.

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
data = map(_ -> rand(dist), 1:1000)
```

## Specify Turing Models

The following code block defines the model along with its prior distributions using Turing.jl. We'll use this model with different fixed values for the `k` parameter.

```julia
@model function model(data, k; min_rt=minimum(x -> x.rt, data))
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
Note that [iterating](https://turinglang.org/ParetoSmooth.jl/stable/turing/#For-Loop-Method) over the data with a for loop necessary to produce correct PSIS-LOO estimates. 

## Estimate the Parameters

Now we'll estimate the parameters using three different fixed `k` values:

```julia
chain_lba1 = sample(model(data, 2.0), NUTS(), 1000)
chain_lba2 = sample(model(data, 0.3), NUTS(), 1000)
chain_lba3 = sample(model(data, 1.0), NUTS(), 1000)
```

## Compute PSIS-LOO

Next we will use the `psis_loo` function to compute the PSIS-LOO for each model:

### Model 1

```julia
loo1 = psis_loo(model_LBA(data, 2.0), chain_lba1)
```

```julia 
[ Info: No source provided for samples; variables are assumed to be from a Markov Chain. If the samples are independent, specify this with keyword argument `source=:other`.
Results of PSIS-LOO-CV with 1000 Monte Carlo samples and 1000 data points. Total Monte Carlo SE of 0.14.
┌───────────┬────────┬──────────┬───────┬─────────┐
│           │  total │ se_total │  mean │ se_mean │
├───────────┼────────┼──────────┼───────┼─────────┤
│   cv_elpd │ 359.33 │    29.11 │  0.36 │    0.03 │
│ naive_lpd │ 364.34 │    28.90 │  0.36 │    0.03 │
│     p_eff │   5.01 │     0.32 │  0.01 │    0.00 │
└───────────┴────────┴──────────┴───────┴─────────┘
```
### Model 2

```julia
loo2 = psis_loo(model_LBA(data, 0.3), chain_lba2)
```

```julia
Results of PSIS-LOO-CV with 1000 Monte Carlo samples and 1000 data points. Total Monte Carlo SE of 0.077.
┌───────────┬────────┬──────────┬───────┬─────────┐
│           │  total │ se_total │  mean │ se_mean │
├───────────┼────────┼──────────┼───────┼─────────┤
│   cv_elpd │ 378.04 │    28.87 │  0.38 │    0.03 │
│ naive_lpd │ 381.63 │    28.71 │  0.38 │    0.03 │
│     p_eff │   3.59 │     0.26 │  0.00 │    0.00 │
└───────────┴────────┴──────────┴───────┴─────────┘
```
### Model 3

```julia
loo3 = psis_loo(model_LBA(data, 1.0), chain_lba3)
```

```julia
Results of PSIS-LOO-CV with 1000 Monte Carlo samples and 1000 data points. Total Monte Carlo SE of 0.14.
┌───────────┬────────┬──────────┬───────┬─────────┐
│           │  total │ se_total │  mean │ se_mean │
├───────────┼────────┼──────────┼───────┼─────────┤
│   cv_elpd │ 359.33 │    29.11 │  0.36 │    0.03 │
│ naive_lpd │ 364.34 │    28.90 │  0.36 │    0.03 │
│     p_eff │   5.01 │     0.32 │  0.01 │    0.00 │
└───────────┴────────┴──────────┴───────┴─────────┘
```

## Compare Models

Finally, we can compare the models using the `loo_compare` function:

```julia
loo_compare((lba1 = loo1, lba2 = loo2, lba3 = loo3))
```

```julia 
┌──────┬─────────┬────────┬────────┐
│      │ cv_elpd │ cv_avg │ weight │
├──────┼─────────┼────────┼────────┤
│ lba2 │    0.00 │   0.00 │   1.00 │
│ lba3 │  -18.71 │  -0.02 │   0.00 │
│ lba1 │  -24.53 │  -0.02 │   0.00 │
└──────┴─────────┴────────┴────────┘
```

Here we indeed correctly identified the generative model we simulated. It is of note that some researchers have criticized using model comparison metrics such as leave-one-out cross-validation. See Gronau et al. (2019) for more information.

# References

Gronau, Q. F., & Wagenmakers, E. J. (2019). Limitations of Bayesian leave-one-out cross-validation for model selection. Computational brain & behavior, 2, 1-11.

Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017).
