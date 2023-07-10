# Parameter Estimation with Turing


It is possible to use [Turing.jl](https://turinglang.org/stable/) to perform Bayesian parameter estimation on models defined in SequentialSamplingModels.jl. Below, we show you how to estimate the parameters for the Linear Ballistic Accumulator (LBA) and to use it to estimate effects.




## Simple Example

### Load Packages

The first step is to load the required packages. You will need to install each package in your local
environment in order to run the code locally. We will also set a random number generator so that the results are reproducible.

```@example Turing
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra

Random.seed!(45461)
```

### Generate Data

We will use the [LBA](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) distribution to simulate data (100 trials) with fixed parameters (those we want to recover only from the data using Bayesian modeling).

```@example Turing
# Generate some data with known parameters
dist = LBA(ν=[3.0, 2.0], A = .8, k = .2, τ = .3)
data = rand(dist, 100)
```

### Define Turing Model

The code snippet below defines a model in Turing. The model function accepts a tuple containing
a vector of choices and a vector of reaction times. The sampling statements define the prior distributions for each parameter. The non-decision time parameter $\tau$ must be founded by the minimum reaction time, `min_rt`. The last sampling statement defines the likelihood of the data given the sampled parameter values.

```@example Turing
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


### Estimate the Parameters

Finally, we perform parameter estimation with `sample()`, which takes the model, and details about the sampling algorithm:

1. `model(data)`: the Turing model with data passed
2. `NUTS(1000, .65)`: a sampler object for the No U-Turn Sampler for 1000 warmup samples.
3. `MCMCThreads()`: instructs turing to run each chain on a seperate thread
4. `n_iterations`: the number of iterations performed after warmup
5. `n_chains`: the number of chains

```@example Turing
# Estimate parameters
chain = sample(model_lba(data), NUTS(1000, .85), MCMCThreads(), 1000, 4)
```

### Posterior Summary

We can compute a description of the posterior distributions.

```@example Turing
# Summarize posteriors
summarystats(chain)
```

As one can see, the parameters (`ν=[3.0, 2.0], A = .8, k = .2, τ = .3`) are successfully recovered from the data.


### Evaluation

It is important to verify that the chains converged. We see that the chains converged according to $\hat{r} \leq 1.05$, and the trace plots below show that the chains look like "hairy caterpillars", which indicates the chains did not get stuck. As expected, the posterior distributions are close to the data generating parameter values.

```@example Turing
plot(chain, grid=false)
```

### Posterior Predictive Distribution

With `predict`, it is possible to sample from the posterior predictive distribution, as follows.
```@example Turing
predictions = predict(model(missing; min_rt = minimum(data[2])), chain)
```
In the following code block, we plot the predictive distributions for each choice.
```@example Turing
choices = predictions.value[:,1,:][:]
rts = predictions.value[:,2,:][:]
# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,1), ylims=(0,4))
histogram!(rts1, subplot=1, color=:grey, bins = 300, norm=true, title="Choice 1")
histogram!(rts2, subplot=2, color=:grey, bins = 300, norm=true, title="Choice 2")
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
```

### Maximum Likelihood Estimation (MLE)

Note that one can also perform a maximum likelihood estimation to estimate the parameters under the Frequentist framework.

```@example Turing
# using Optim

# mle = optimize(model_lba(data), MLE())
```

*TODO: replace with `coeftable` once [this](https://github.com/TuringLang/Turing.jl/) is merged*


## Effect of a Condition on Multiple Parameters

### Generate Data

In this example, we will get closer to real use-cases by starting with the data stored in a `DataFrame`. This dataframe will be a combination of data generated from two different distributions with different parameters, corresponding to two experimental conditions (e.g., Speed vs. Accuracy).

```@example Turing
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
using Distributions
using DataFrames
using StatsPlots
using StatsModels


# Generate data with different drifts for two conditions A vs. B
Random.seed!(254)
df1 = DataFrame(rand(LBA(ν=[3.0, -1.0], A=0.5, k=0.5, τ=0.3), 500))
df1[!, :condition] = repeat(["A"], nrow(df1))
df2 = DataFrame(rand(LBA(ν=[2.0, -1.5], A=0.5, k=0.2, τ=0.3), 500))
df2[!, :condition] = repeat(["B"], nrow(df2))

df = vcat(df1, df2)
```

These 2 conditions *A* and *B* differ on their drift rates (`[3.0, -1.0]` vs. `[2.0, -1.5]`) and on threshold *k* (`0.5` vs. `0.2`)


**WIP.**


## Random Effects

**WIP.**