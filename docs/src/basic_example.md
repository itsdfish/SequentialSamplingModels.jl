# Basic Example 

This package provides plotting functionality for sequential sampling models. The code block below provides an example:

```@example basic_example 
using SequentialSamplingModels
using Plots
using Random 
Random.seed!(85)

ν = [1.0,0.50]
k = 0.50
A = 1.0
τ = 0.30

dist = RDM(;ν, k, A, τ)
histogram(dist; xlims=(0,2.5))
plot!(dist; t_range=range(.301, 2.5, length=100))
```
You can overwrite the default plot options by passing keyword arguments. The code block below provides an example:

```@example basic_example 
using SequentialSamplingModels
using Plots
using Random 
Random.seed!(85)

ν = [1.0,0.50]
k = 0.50
A = 1.0
τ = 0.30

dist = RDM(;ν, k, A, τ)
histogram(dist; xlims=(0,2.5))
plot!(dist; t_range=range(.301, 2.5, length=100), color=:darkorange)
```