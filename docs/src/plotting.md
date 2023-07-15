## Plotting 

SequentialSamplingModels.jl provides the following convienence plotting methods:

- `plot`
- `plot!`
- `histogram`
- `histogram!`

which will work with all SSMs available in the package. Here is a simple example:

```@example 
using SequentialSamplingModels
using Plots 

dist = RDM(;ν=[1,2,3], k=.30, A=.70, τ=.20)
histogram(dist)
plot!(dist)
```
You can overwrite the default plot options by passing keyword arguments. The code block below shows how to change the color of the line:

```@example 
using SequentialSamplingModels
using Plots 

dist = RDM(;ν=[1,2,3], k=.30, A=.70, τ=.20)
histogram(dist)
plot!(dist; color=:darkorange)
```