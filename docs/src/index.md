# SequentialSamplingModels.jl

## Overview 
This package provides a unified interface for simulating and evaluating popular sequential sampling models (SSMs), which integrates with the following packages:

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl): functions for probability distributions
- [NeuralEstimators.jl](https://github.com/msainsburydale/NeuralEstimators.jl): amortized inference using neural networks
- [Pigeons.jl](http://pigeons.run/dev/): Bayesian parameter estimation and Bayes factors
- [Plots.jl](https://github.com/JuliaPlots/Plots.jl): extended plotting tools for SSMs
- [Turing.jl](https://turinglang.org/dev/docs/using-turing/get-started): Bayesian parameter estimation

## Background
SSMs, also known as an evidence accumulation models, are a broad class of dynamic models of human decision making in which evidence for each option accumulates until the evidence for one option reaches a decision threshold. Models within this class make different assumptions about the nature of the evidence accumulation process. An example of the evidence accumulation process is illustrated below for the Racing Diffusion Model (RDM):

```@setup accumulation
# using Plots
# using Random 
# using SequentialSamplingModels 

# Random.seed!(50)
# model = RDM(ν = [1.0,1.5,2.0])
# α = model.A + model.k
# times, evidence = simulate(model)
# color = [RGB(.251, .388, .847) RGB(.584, .345, .689) RGB(.796, .235, .20) ]
# animation = @animate for i ∈ 1:length(times)
#     plot(times[1:i], evidence[1:i,:], xlims = extrema(times); color,
#     ylims = (-.5, α + .1), ticks = :none, linewidth = 1.5)
#     hline!([α], color = :black, linestyle = :dash, leg = false)
# end
# gif(animation, "rdm.gif", fps = 30)
```

![](assets/rdm.gif)
# Installation

You can install a stable version of `SequentialSamplingModels` by running the following in the Julia REPL:

```julia
] add SequentialSamplingModels
```

The package can then be loaded with:

```julia
using SequentialSamplingModels
```

# Quick Example

The example belows shows how to perform three common tasks:

1. generate simulated data
2. evaluate the log likelihood of data
3. plot the predictions of the model

```@example quick_example
using SequentialSamplingModels
using Plots
using Random

Random.seed!(2054)

# Create LBA distribution with known parameters
dist = LBA(; ν=[2.75,1.75], A=0.8, k=0.5, τ=0.25)
# Sample 10,000 simulated data from the LBA
sim_data = rand(dist, 10_000)
# compute log likelihood of simulated data 
logpdf(dist, sim_data)
# Plot the RT distribution for each choice
histogram(dist)
plot!(dist; t_range=range(.3,2.5, length=100), xlims=(0, 2.5))
```
# Contributing

If you are interested in contributing, please review the [developer guidelines](developer_guide.md) before beginning. 

# References
Evans, N. J. & Wagenmakers, E.-J. Evidence accumulation models: Current limitations and future directions. Quantitative Methods for Psychololgy 16, 73–90 (2020).

Forstmann, B. U., Ratcliff, R., & Wagenmakers, E. J. (2016). Sequential sampling models in cognitive neuroscience: Advantages, applications, and extensions. Annual Review of Psychology, 67, 641-666.

Jones, M., & Dzhafarov, E. N. (2014). Unfalsifiability and mutual translatability of major modeling schemes for choice reaction time. Psychological Review, 121(1), 1.