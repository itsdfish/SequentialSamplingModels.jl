## Plotting 

Plots.jl contains plotting functionality for sequential sampling models (SSMs). The code block below provides a simple example of plotting the predictions of SSMs:
```@example plot_example
using SequentialSamplingModels
using Plots 

dist = RDM(;ν=[1,2,3], k=.30, A=.70, τ=.20)
histogram(dist)
plot!(dist)
```
More details on plotting SSMs can be found in the [documentation](https://itsdfish.github.io/Plots.jl/dev/) of Plots.jl.