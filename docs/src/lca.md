# Leaky Competing Accumulator

The Leaky Competing Accumulator (LCA; Usher & McClelland, 2001) is a sequential sampling model in which evidence for options races independently. The LBA makes an additional simplification that evidence accumulates in a linear and ballistic fashion, meaning there is no intra-trial noise. Instead, evidence accumulates deterministically and linearly until it hits the threshold.

# Example
In this example, we will demonstrate how to use the LBA in a generic two alternative forced choice task. 
```@setup lca
using SequentialSamplingModels
using Plots
using Random

ν = [2.5,2.0]
α = 1.5
β = 0.20
λ = 0.10 
σ = 1.0
τ = 0.30
Δt = .001

dist = LCA(; ν, α, β, λ, τ, σ, Δt)
choices,rts = rand(dist, 500)

# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2 
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,5), ylims=(0,.5))
histogram!(rts1, subplot=1, color=:grey, bins = 100, norm=true, title="Choice 1")
histogram!(rts2, subplot=2, color=:grey, bins = 100, norm=true, title="Choice 2")
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
```

## Load Packages
The first step is to load the required packages.

```@example lca
using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Drift Rates

The drift rates control the speed with which information accumulates. Typically, there is one drift rate per option. 

```@example lca
ν = [2.5,2.0]
```
### Threshold
The threshold $\alpha$ represents the amount of evidence required to make a decision.
```@example lca 
α = 1.5
```

### Lateral Inhibition 
The parameter $\beta$ inhibits evidence of competing options proportionally to their evidence value.
```@example lca 
β = 0.20
```
### Leak Rate
The parameter $\lambda$ controls the rate with which evidence decays or "leaks".
```@example lca 
λ = 0.10 
```
### Diffusion Noise
Diffusion noise is the amount of within trial noise in the evidence accumulation process. 
```@example lca 
σ = 1.0
```
### Time Step
The time step parameter $\Delta t$ is the precision of the discrete time approxmation. 

```@example lca 
Δt = .001
```

### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time. 
```@example lca 
τ = 0.30
```
### LCA Constructor 

Now that values have been asigned to the parameters, we will pass them to `LCA` to generate the model object.

```@example lca 
dist = LCA(; ν, α, β, λ, τ, σ, Δt)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example lca 
 choices,rts = rand(dist, 10_000)
```
## Plot Simulation
The code below plots a histogram for each option.
 ```@example lca 
# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2 
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,2), ylims=(0,1.75))
histogram!(rts1, subplot=1, color=:grey, bins = 50, norm=true, title="Choice 1")
histogram!(rts2, subplot=2, color=:grey, bins = 50, norm=true, title="Choice 2")
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
```
# References

Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. Psychological Review, 108 3, 550–592. https://doi.org/10.1037/0033-295X.108.3.550
