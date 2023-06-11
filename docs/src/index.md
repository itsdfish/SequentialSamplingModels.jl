# SequentialSamplingModels.jl

Documentation is under construction.

This package is a collection of sequential sampling models and is based on the Distributions.jl API.
Sequential sampling models, also known as an evidence accumulation models, are a broad class of dynamic models of human decision making in which evidence for each option accumulates until the evidence for one option reaches a decision threshold. Models within this class make different assumptions about the nature of the evidence accumulation process. See the references below for a broad overview of sequential sampling models. An example of the evidence accumulation process is illustrated below for the leaking competing accumulator. 

```@setup accumulation
using Plots
using Random
using Colors
using SequentialSamplingModels
using SequentialSamplingModels: increment!
Random.seed!(8437)

function sim(model)
    
    n = length(model.ν)
    x = fill(0.0, n)
    μΔ = fill(0.0, n)
    ϵ = fill(0.0, n)
    t = 0.0
    Δt = .005
    evidence = Vector{Vector{Float64}}()
    while all(x .< model.α)
        t += Δt
        increment!(model, x, μΔ, ϵ)
        push!(evidence, copy(x))
    end  
    return t,evidence
end
parms = (α = 1.5, 
            β=0.20,
             λ=0.10, 
             ν=[2.5,2.0], 
             Δt=.001, 
             τ=.30, 
             σ=1.0)
model = LCA(; parms...)
t,evidence = sim(model)
n_steps = length(evidence)
time_steps = range(0, t, length=n_steps)
lca_plot = plot(time_steps, hcat(evidence...)', xlabel="Time (seconds)", ylabel="Evidence", 
    label=["option1" "option2"], ylims=(0, 2.0), grid=false, linewidth = 2,
    color =[RGB(148/255, 90/255, 147/255) RGB(90/255, 112/255, 148/255)])
hline!(lca_plot, [model.α], color=:black, linestyle=:dash, label="threshold", linewidth = 2)
```

```@example accumulation 
lca_plot
```
# Installation

You can install a stable version of SequentialSamplingModels by running the following in the Julia REPL:

```julia
] add SequentialSamplingModels
```

The package can then be loaded with:

```julia
using SequentialSamplingModels
```

# References
Evans, N. J. & Wagenmakers, E.-J. Evidence accumulation models: Current limitations and future directions. Quantitative Methods for Psychololgy 16, 73–90 (2020).

Forstmann, B. U., Ratcliff, R., & Wagenmakers, E. J. (2016). Sequential sampling models in cognitive neuroscience: Advantages, applications, and extensions. Annual Review of Psychology, 67, 641-666.

Jones, M., & Dzhafarov, E. N. (2014). Unfalsifiability and mutual translatability of major modeling schemes for choice reaction time. Psychological Review, 121(1), 1.