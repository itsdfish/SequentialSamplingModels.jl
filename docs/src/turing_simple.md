# A Simple Turing Model

It is possible to use [Turing.jl](https://turinglang.org/stable/) to perform Bayesian parameter estimation on models defined in SequentialSamplingModels.jl. Below, we show you how to estimate the parameters for the [Linear Ballistic Accumulator (LBA)](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) and to use it to estimate effects.

Note that you can easily swap the LBA model from this example for other [SSM models](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/api/) simply by changing the names of the parameters.



## Load Packages

The first step is to load the required packages. You will need to install each package in your local
environment in order to run the code locally. We will also set a random number generator so that the results are reproducible.

```@setup turing_simple
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
using StatsPlots

Random.seed!(45461)
```

## Generate Data

We will use the [LBA](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) distribution to simulate data (100 trials) with fixed parameters (those we want to recover only from the data using Bayesian modeling).

```@example turing_simple
# Generate some data with known parameters
dist = LBA(ν=[3.0, 2.0], A = .8, k = .2, τ = .3)
data = rand(dist, 100)
```

The `rand()` function will sample random draws from the distribution, and store that into a tuple of 2 vectors (one for `choice` and one for `rt`).

## Specify Turing Model

The code snippet below defines a model in Turing. The model function accepts a tuple containing
a vector of choices and a vector of reaction times. The sampling statements define the prior distributions for each parameter. The non-decision time parameter $\tau$ must be founded by the minimum reaction time, `min_rt`. The last sampling statement defines the likelihood of the data given the sampled parameter values.

```@example turing_simple
# Specify LBA model
@model function model_lba(data; min_rt = minimum(data[2]))
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

```@example turing_simple
# Estimate parameters
chain = sample(model_lba(data), NUTS(1000, .85), MCMCThreads(), 1000, 4)
```

## Posterior Summary

We can compute a description of the posterior distributions.

```@example turing_simple
# Summarize posteriors
summarystats(chain)
```

As you can see, based on the mean values of the posterior distributions, the original parameters (`ν=[3.0, 2.0], A = .8, k = .2, τ = .3`) are successfully recovered from the data (the accuracy would increase with more data).


## Evaluation

It is important to verify that the chains converged. We see that the chains converged according to $\hat{r} \leq 1.05$, and the trace plots below show that the chains look like "hairy caterpillars", which indicates the chains did not get stuck. As expected, the posterior distributions are close to the data generating parameter values.

```@example turing_simple
plot(chain)
```

## Posterior Predictive Distribution

The next step is to generate predictions from the posterior distributions. For this, we need to pass a dataset with empty (`missing`) values (so that Turing knows what to predict).

We can then use the `predict()` method to generate predictions from this model. However, because the most of `SequentialSamplingModels` distributions return a tuple (choice and RT), the predicted output has the two types of variables mixed together. We can delineate the two by taking every 2nd values to get the predicted choice and RTs, respectively.


```@example turing_simple
predictions = predict(model_lba(missing; min_rt = minimum(data.rt)), chain)

pred_choice = Array(predictions)[:, 1:2:end]
pred_rt = Array(predictions)[:, 2:2:end]
```

In the following code block, we plot the predictive distributions for each choice.

```@example turing_simple
# Get RTs for option 1 and 2
rts1 = pred_rt[pred_choice .== 1]
rts2 = pred_rt[pred_choice .== 2]

# Specify plot layout
histogram(layout=(2, 1), xlabel="RT", ylabel="Density", legend=false, xlims=(0, 1), ylim=(0, 150))
# Add data
histogram!(rts1, subplot=1, color=:green, norm=false, title="Choice 1", bins=0:0.01:1)
histogram!(rts2, subplot=2, color=:red, norm=false, title="Choice 2", bins=0:0.01:1)
```
