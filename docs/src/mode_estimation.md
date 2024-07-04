```@setup mode_estimation
using Turing
using SequentialSamplingModels
using Random
Random.seed!(654)
map_estimate = nothing
mle_estimate = nothing
lb = [0.0]
ub = [0.0]
```
# Mode Estimation 

Mode estimation can be useful when full Bayesian inference is not desired, or when one wishes to initialize an MCMC sampling algorithm near the mode of the posterior distribution. Below, we show how to estimate the mode of a posterior distribution using [Turing's](https://github.com/TuringLang/Turing.jl) capabilities for maximum likelihood estimation (MLE) and maximum a postiori (MAP).

# Example 

For this simple example, we will estimate the mode of the posterior distribution of the [Wald](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/wald/) model. In the code block below, we will load the required packages. 

## Load Packages
```@example mode_estimation
using Turing
using SequentialSamplingModels
using Random
Random.seed!(654)
```

## Generate Data
We will generate 50 simulated reaction times from the Wald model. 
```@example mode_estimation
n_samples = 50
rts = rand(Wald(ν=1.5, α=.8, τ=.3), n_samples)
```

## Define Turing Model
Below, we define a Turing model with prior distributions specified. Note that the prior 
distributions are only applicable to MAP.

```@example mode_estimation
@model function model(rts; min_rt = minimum(rts))
    ν ~ truncated(Normal(1.5, 1), 0, Inf)
    α ~ truncated(Normal(.8, 1), 0, Inf)
    τ ~ Uniform(0, min_rt)
    rts ~ Wald(ν, α, τ)
    return (;ν, α, τ)
end
```

## Set Parameter Bounds

Specifying lower and upper bounds for the parameters is necessary to prevent the MLE and MAP 
estimators from searching invalid regions of the parameter space. 
```@example mode_estimation
lb = [0,0,0]
ub = [10, 10, minimum(rts)]
```

## MLE 
MLE is performed via the function `maximum_likelihood`.
```@example mode_estimation
mle_estimate = maximum_likelihood(model(rts); lb, ub)
```

## MAP

MAP is performed via the function `maximum_a_posteriori`.
```@example mode_estimation
map_estimate = maximum_a_posteriori(model(rts); lb, ub)
```
In both cases, the estimates are in the proximity of the data-generating values. 

## Accessing Estimates

The estimates are located in the field `values`, which can be accessed as follows:
```@example mode_estimation
map_estimate.values
```
which returns a named vector. To obtain a regular vector, append `.array` to the code above.

## Seeding MCMC Sampler

You can seed the MCMC sampler as illustrated below:

```julia
chain = sample(model, NUTS(), 1_000; initial_params=map_estimate.values.array)
```

Additional details can be found in the [Turing documentation](https://turinglang.org/docs/versions/v0.33.0/tutorials/docs-17-mode-estimation/index.html)