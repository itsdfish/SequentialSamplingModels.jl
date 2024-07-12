# A Simple Turing Model

It is possible to use [Turing.jl](https://turinglang.org/stable/) to perform Bayesian parameter estimation on models defined in SequentialSamplingModels.jl. Below, we show you how to estimate the parameters for the [Linear Ballistic Accumulator (LBA)](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) and to use it to estimate effects.

Note that you can easily swap the LBA model from this example for other [SSM models](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/api/) simply by changing the names of the parameters.

## Load Packages

The first step is to load the required packages. You will need to install each package in your local
environment in order to run the code locally. We will also set a random number generator so that the results are reproducible.

```julia
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
using StatsPlots
using Random

Random.seed!(45461)
```

## Generate Data

We will use the [LBA](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) distribution to simulate data (100 trials) with fixed parameters (those we want to recover only from the data using Bayesian modeling).

```julia
# Generate some data with known parameters
dist = LBA(ν=[3.0, 2.0], A = .8, k = .2, τ = .3)
data = rand(dist, 100)
```

The `rand()` function will sample random draws from the distribution, and store that into a named tuple of 2 vectors (one for `choice` and one for `rt`). The individual vectors can be accessed by their names using `data.choice` and `data.rt`.

## Specify Turing Model

The code snippet below defines a model in Turing. The model function accepts a tuple containing
a vector of choices and a vector of reaction times. The sampling statements define the prior distributions for each parameter. The non-decision time parameter $\tau$ must be founded by the minimum reaction time, `min_rt`. The last sampling statement defines the likelihood of the data given the sampled parameter values.

```julia
# Specify LBA model
@model function model(data; min_rt = minimum(data.rt))
    # Priors
    ν ~ MvNormal(zeros(2), I * 2)
    A ~ truncated(Normal(.8, .4), 0.0, Inf)
    k ~ truncated(Normal(.2, .2), 0.0, Inf)
    τ  ~ Uniform(0.0, min_rt)

    # Likelihood
    data ~ LBA(;ν, A, k, τ )
end
```


## Estimate the Parameters

Finally, we perform parameter estimation with `sample()`, which takes the model, and details about the sampling algorithm:

1. `model(data)`: the Turing model with data passed
2. `NUTS(1000, .65)`: a sampler object for the No U-Turn Sampler for 1000 warmup samples.
3. `MCMCThreads()`: instructs Turing to run each chain on a separate thread
4. `n_iterations`: the number of iterations performed after warmup
5. `n_chains`: the number of chains

```julia
# Estimate parameters
chain = sample(model(data), NUTS(1000, .85), MCMCThreads(), 1000, 4)
```

```julia
Chains MCMC chain (1000×17×4 Array{Float64, 3}):

Iterations        = 1001:1:2000
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 8.09 seconds
Compute duration  = 29.97 seconds
parameters        = ν[1], ν[2], A, k, τ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

        ν[1]    2.8193    0.4348    0.0119   1351.1694   1590.9543    1.0023       45.0871
        ν[2]    1.6437    0.3820    0.0103   1375.9971   1509.2023    1.0015       45.9155
           A    0.7280    0.1761    0.0049   1270.3337   1012.5202    1.0026       42.3897
           k    0.2399    0.1178    0.0033   1156.3022    886.1148    1.0028       38.5846
           τ    0.2808    0.0281    0.0008   1192.9074    714.0924    1.0040       39.8060

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

        ν[1]    2.0183    2.5259    2.7910    3.0938    3.7300
        ν[2]    0.9455    1.3753    1.6333    1.8883    2.4329
           A    0.3969    0.6108    0.7166    0.8416    1.1063
           k    0.0562    0.1503    0.2244    0.3161    0.5013
           τ    0.2202    0.2630    0.2836    0.3018    0.3273
```
## Posterior Summary

We can compute a description of the posterior distributions.

```julia
# Summarize posteriors
summarystats(chain)
```

```julia 
Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

        ν[1]    2.8193    0.4348    0.0119   1351.1694   1590.9543    1.0023       45.0871
        ν[2]    1.6437    0.3820    0.0103   1375.9971   1509.2023    1.0015       45.9155
           A    0.7280    0.1761    0.0049   1270.3337   1012.5202    1.0026       42.3897
           k    0.2399    0.1178    0.0033   1156.3022    886.1148    1.0028       38.5846
           τ    0.2808    0.0281    0.0008   1192.9074    714.0924    1.0040       39.8060
```

As you can see, based on the mean values of the posterior distributions, the original parameters (`ν=[3.0, 2.0], A = .8, k = .2, τ = .3`) are successfully recovered from the data (the accuracy would increase with more data).


## Evaluation

It is important to verify that the chains converged. We see that the chains converged according to $\hat{r} \leq 1.05$, and the trace plots below show that the chains look like "hairy caterpillars", which indicates the chains did not get stuck. As expected, the posterior distributions are close to the data generating parameter values.

```julia
plot(chain)
```

![](assets/simple_turing_example_plot.png)