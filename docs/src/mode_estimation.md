# Mode Estimation 

Mode estimation can be useful when full Bayesian inference is not desired, or when one wishes to initialize an MCMC sampling algorithm near the mode of the posterior distribution. Below, we show how to estimate the mode of a posterior distribution using [Turing's](https://github.com/TuringLang/Turing.jl) capabilities for maximum likelihood estimation (MLE) and maximum a postiori (MAP).

# Example 

For this simple example, we will estimate the mode of the posterior distribution of the [Wald](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/wald/) model. In the code block below, we will load the required packages. 

## Load Packages
```julia
using Turing
using SequentialSamplingModels
using Random
Random.seed!(654)
```

## Generate Data
We will generate 50 simulated reaction times from the Wald model. 
```julia
n_samples = 50
rts = rand(Wald(ν=1.5, α=.8, τ=.3), n_samples)
```

## Define Turing Model
Below, we define a Turing model with prior distributions specified. Note that the prior 
distributions are only applicable to MAP.

```julia
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
```julia
lb = [0,0,0]
ub = [10, 10, minimum(rts)]
```

## MLE 
MLE is performed via the function `maximum_likelihood`.
```julia
mle_estimate = maximum_likelihood(model(rts); lb, ub)
```

```julia 
ModeResult with maximized lp of -7.62
[1.441581500744421, 0.6990063775377103, 0.32611139519154697]
```
## MAP

MAP is performed via the function `maximum_a_posteriori`.
```julia
map_estimate = maximum_a_posteriori(model(rts); lb, ub)
```

```julia
ModeResult with maximized lp of -8.20
[1.4488716348067334, 0.7022870580892082, 0.3255810892442324]
```

In both cases, the estimates are in the proximity of the data-generating values. 

## Accessing Estimates

The estimates are located in the field `values`, which can be accessed as follows:
```julia
map_estimate.values
```
which returns a named vector. To obtain a regular vector, append `.array` to the code above.

## Seeding MCMC Sampler

You can seed the MCMC sampler as illustrated below:

```julia
chain = sample(model(rts), NUTS(), 1_000; initial_params=map_estimate.values.array)
```

```julia 
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (1000×15×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 3.97 seconds
Compute duration  = 3.97 seconds
parameters        = ν, α, τ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

           ν    1.5597    0.3184    0.0228   202.3201   154.0195    1.0087       50.9366
           α    0.8230    0.2030    0.0185   163.9590    80.6732    1.0096       41.2787
           τ    0.2937    0.0474    0.0043   174.7682   106.7551    1.0117       44.0000

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           ν    0.9220    1.3475    1.5532    1.7466    2.2150
           α    0.5375    0.6884    0.7853    0.9231    1.3999
           τ    0.1533    0.2720    0.3055    0.3261    0.3516
```

Additional details can be found in the [Turing documentation](https://turinglang.org/docs/versions/v0.33.0/tutorials/docs-17-mode-estimation/index.html)