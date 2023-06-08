# Attentional Drift Diffusion Model 


# Example

```@setup aDDM
using SequentialSamplingModels
using StatsBase
using Plots

mutable struct Transition
    state::Int 
    n::Int
    mat::Array{Float64,2} 
 end

 function Transition(mat)
    n = size(mat,1)
    state = rand(1:n)
    return Transition(state, n, mat)
 end
 
 function attend(transition)
     (;mat,n,state) = transition
     w = mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end

 model = aDDM()
 
 tmat = Transition([.98 .015 .005;
                    .015 .98 .005;
                    .45 .45 .1])

 choices,rts = rand(model, 10_000, attend, tmat)

# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2 
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density")
histogram!(rts1, subplot=1, color=:grey, bins = 50, norm=true, title="Choice 1")
histogram!(rts2, subplot=2, color=:grey, bins = 50, norm=true, title="Choice 2")
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
```
## Load Packages

```@example aDDM
using SequentialSamplingModels
using StatsBase
```

## Define Transition Type 

```@example aDDM 
mutable struct Transition
    state::Int 
    n::Int
    mat::Array{Float64,2} 
 end

function Transition(mat)
    n = size(mat,1)
    state = rand(1:n)
    return Transition(state, n, mat)
 end
```


## Define Transition Matrix 
 ```@example aDDM 
tmat = Transition([.98 .015 .005;
                    .015 .98 .005;
                    .45 .45 .1])
```
## Create Model Object
 ```@example aDDM 
 model = aDDM()
```
## Simulate Model
 ```@example aDDM 
 choices,rts = rand(model, 10_000, attend, tmat)
```
## Plot Simulation
 ```@example aDDM 
# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2 
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density")
histogram!(rts1, subplot=1, color=:grey, bins = 50, norm=true, title="Choice 1")
histogram!(rts2, subplot=2, color=:grey, bins = 50, norm=true, title="Choice 2")
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
```
# References

Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of value in simple choice. Nature neuroscience, 13(10), 1292-1298.