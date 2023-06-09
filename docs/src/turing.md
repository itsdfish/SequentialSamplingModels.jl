## Parameter Estimation with Turing

It is possible to use [Turing.jl](https://turinglang.org/stable/) to perform Bayesian parameter estimation on models defined in SequentialSamplingModels.jl. Below, we show you how to estimate the parameters for the Linear Ballistic Accumulator (LBA).

```@setup Turing 
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
using StatsPlots

@model model(data) = begin
    min_rt = minimum(data[2])
    ν ~ MvNormal(zeros(2), I * 2)
    A ~ truncated(Normal(.8, .4), 0.0, Inf)
    k ~ truncated(Normal(.2, .2), 0.0, Inf)
    τ  ~ Uniform(0.0, min_rt)
    data ~ LBA(;ν, A, k, τ )
end

# generate some data
Random.seed!(254)
dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
data = rand(dist, 100)

# estimate parameters
chain = sample(model(data), NUTS(1000, .65), MCMCThreads(), 1000, 4)
```

# Example

## Load Packages
The first step is to load the required packages. You will need to install each package in your local
environment in order to run the code locally.

```@example Turing 
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
```

## Define Turing Model
The code snippet below defines a model in Turing. The model function accepts a tuple containing
a vector of choices and a vector of reaction times. The sampling statements define the prior distributions for each parameter. The non-decision time parameter $\tau$ must be founded by the minimum reaction time, `min_rt`. The last sampling statement defines the likelihood of the data given the sampled parameter values. 
```@example Turing 
@model model(data) = begin
    min_rt = minimum(data[2])
    ν ~ MvNormal(zeros(2), I * 2)
    A ~ truncated(Normal(.8, .4), 0.0, Inf)
    k ~ truncated(Normal(.2, .2), 0.0, Inf)
    τ  ~ Uniform(0.0, min_rt)
    data ~ LBA(;ν, A, k, τ )
end
```
## Generate Simulated Data
In the code snippet below, we set a seed for the random number generator and generate $100$ simulated
trials from the LBA from which we will estimate parameters.
```@example Turing 
# generate some data
Random.seed!(45461)
dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
data = rand(dist, 100)
```
## Estimate the Parameters
Finally, we perform parameter estimation with `sample`, which accepts the following inputs:

1. `model(data)`: the Turing model with data passed
2. `NUTS(1000, .65)`: a sampler object for the No U-Turn Sampler for 1000 warmup samples.
3. `MCMCThreads()`: instructs turing to run each chain on a seperate thread
4. `n_iterations`: the number of iterations performed after warmup
5. `n_chains`: the number of chains

```@example Turing 
# estimate parameters
chain = sample(model(data), NUTS(1000, .85), MCMCThreads(), 1000, 4)
```

## Evaluation

It is important to verify that the chains converged. We see that the chains converged according to $\hat{r} \leq 1.05$, and the trace plots below show that the chains look like "hairy catipillars", whichin indictes the chains did not get stuck. As expected, the posterior distributions are close to the data generating parameter values.

```@example Turing 
plot(chain, grid=false)
```

