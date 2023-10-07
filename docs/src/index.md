# SequentialSamplingModels.jl

## Overview 
This package provides a unified interface for simulating and evaluating popular sequential sampling models (SSMs), which integrates with the following packages:

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl): functions for probability distributions
- [Pigeons.jl](http://pigeons.run/dev/): Bayesian parameter estimation and Bayes factors
- [SSMPlots.jl](https://itsdfish.github.io/SSMPlots.jl/dev/): plotting tools for SSMs
- [Turing.jl](https://turinglang.org/dev/docs/using-turing/get-started): Bayesian parameter estimation

## Background
SSMs, also known as an evidence accumulation models, are a broad class of dynamic models of human decision making in which evidence for each option accumulates until the evidence for one option reaches a decision threshold. Models within this class make different assumptions about the nature of the evidence accumulation process. An example of the evidence accumulation process is illustrated below for the Leaking Competing Accumulator (LCA):

```@setup accumulation
using Plots
using Random
using Colors
using SequentialSamplingModels
using SequentialSamplingModels: increment!
Random.seed!(8437)

parms = (α = 1.5,
            β=0.20,
             λ=0.10,
             ν=[2.5,2.0],
             Δt=.001,
             τ=.30,
             σ=1.0)
model = LCA(; parms...)
time_steps,evidence = simulate(model)
lca_plot = plot(time_steps, evidence, xlabel="Time (seconds)", ylabel="Evidence",
    label=["option1" "option2"], ylims=(0, 2.0), grid=false, linewidth = 2,
    color =[RGB(148/255, 90/255, 147/255) RGB(90/255, 112/255, 148/255)])
hline!(lca_plot, [model.α], color=:black, linestyle=:dash, label="threshold", linewidth = 2)
savefig("lca_plot.png")
```

![](lca_plot.png)
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

# References
Evans, N. J. & Wagenmakers, E.-J. Evidence accumulation models: Current limitations and future directions. Quantitative Methods for Psychololgy 16, 73–90 (2020).

Forstmann, B. U., Ratcliff, R., & Wagenmakers, E. J. (2016). Sequential sampling models in cognitive neuroscience: Advantages, applications, and extensions. Annual Review of Psychology, 67, 641-666.

Jones, M., & Dzhafarov, E. N. (2014). Unfalsifiability and mutual translatability of major modeling schemes for choice reaction time. Psychological Review, 121(1), 1.