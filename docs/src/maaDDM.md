 # Attentional Drift Diffusion Model 

The multi-attribute attentional drift diffusion model (MAADDM; Yang & Krajbich, 2023) describes how attentional processes drive drive decision making. Much like the ADDM, in the MAADDM preference for the currently attended option accrues faster than preference for non-attended options. However, the MAADDM has been extended to model shifts in attention for alternatives with two attributes. As with other sequential sampling models, the first option to hit a decision threshold determines the resulting choice and reaction time.

# Example

```@setup maaDDM
using SequentialSamplingModels
using StatsBase
using Plots
using Random

Random.seed!(5487)

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

ν₁₁ = 4.0 
ν₁₂ = 5.0 
ν₂₁ = 5.0 
ν₂₂ = 4.0
α = 1.0 
z = 0.0
θ = .3
ϕ = .50
ω = .70
σ = .02
Δ = .0004

dist = maaDDM(; ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)

tmat = Transition([.98 .015 .0025 .0025;
                .015 .98 .0025 .0025;
                .0025 .0025 .98 .015;
                .0025 .0025 .015 .98])

 choices,rts = rand(dist, 100, attend, tmat)

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

In this example, we will develope a MAADDM for binary choice and generate its predictions. Unlike many other sequential sampling models, it is necessary to specify the attentional process, or supply fixation patterns from eye tracking data. 
## Load Packages
The first step is to load the required packages.

```@example maaDDM
using SequentialSamplingModels
using StatsBase
using Plots

Random.seed!(9854)
```

## Define Transition Type 

To represent the transition of attention from one option to the other, we will definite a `Transition` type and constructor. The fields of the `Transition` type are:

1. `state`: an index for the current state
2. `n`: the number of states
3. `mat`: an $n\times n$ transition matrix

The constructor accepts a transition matrix, extracts the number of states, and initializes the first state randomly with equal probability.

```@example maaDDM
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

The transition matrix is defined below in the constructor for `Transition`. As shown in the table below, the model's attention can be in one of three states: option 1, option 2, or non-option, which is any area except the two options.

|          	|              	| Option 1     	|             	| Option 1     	|             	|
|----------	|--------------	|--------------	|-------------	|--------------	|-------------	|
|          	|              	| **Attribute 1**  	| **Attribute 2** 	| **Attribute 1**  	| **Attribute 2** 	|
| **Option 1** 	| **Attribute 1**  	| 0.980         |0.015          |    0.0025    	| 0.0025        |
|          	| **Attribute 2**  	|    0.015      | 0.980          |     0.0025    | 0.0025        |
| **Option 1** 	| **Attribute 1**  	|    0.0025    	| 0.0025        | 0.980          |    0.015      |
|          	| **Attribute 2**  	|    0.0025    	| 0.0025       	|0.015          | 0.980          |


The transition matrix above embodies the following assumptions:

1. Once the model attends to an option, it dwells on the option for some time.
2. There is not a bias for one option over the other.
3. There is a larger chance of transitioning between attributes within the same alternative than transitioning between alternatives
4. Transitions are Markovian in that they only depend on the previous state.

 ```@example maaDDM
 tmat = Transition([.98 .015 .0025 .0025;
                    .015 .98 .0025 .0025;
                    .0025 .0025 .98 .015;
                    .0025 .0025 .015 .98])

```

## Attend Function 

The function below generates the next attention location based on the previous location. 

```@example maaDDM
 function attend(transition)
     (;mat,n,state) = transition
     w = mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end
```
## Create Model Object
The code snippets assign values to parameters of the MAADDM and create a model object.


### Drift Rate Components
In the decision making task, there are two alternatives with two attributes each. This leads to four components of the drift rates: $\nu_{1,1}, \nu_{1,2},\nu_{2,1},\nu_{2,2}$ where the first index corresponds to alternative and the second index corresponds to attribute.  To form the drift rate, each component is weighted by non-attention bias and then a difference is computed.
```@example maaDDM
ν₁₁ = 4.0 
ν₁₂ = 5.0 
ν₂₁ = 5.0 
ν₂₂ = 4.0
```

### Threshold
The threshold hold represents the amount of evidence required to make a decision. This parameter is typically fixed at $\alpha = 1$.
```@example maaDDM
α = 1.0
```

### Starting Point
The starting point of the evidence accumulation process is denoted $z$ and is typically fixed to $0$.
```@example maaDDM
z = 0.0
```

### Non-Attend Bias Alternative
The non-attend bias parameter $\theta$ determines how much the non-attended option contributes to the 
evidence accumulation process. In the standard DDM, $\theta=1$. 
```@example maaDDM
θ = 0.30
```

### Non-Attend Bias Attribute
The non-attend bias parameter $\psi$ determines how much the non-attended option contributes to the 
evidence accumulation process. In the standard DDM, $\psi=1$. 
```@example maaDDM
ϕ = .50
```

### Attribute Weight
The parameter $\omega$ denotes the weight of the first attribute.
```@example maaDDM
ω = .70
```

### Diffusion Noise
Diffusion noise, $\sigma$ represents intra-trial noise during the evidence accumulation process.
```@example maaDDM
σ = 0.02
```

### Drift Rate Scalar
The drift rate scalar controls how quickly evidence accumulates for each option. 
```@example maaDDM
Δ = 0.0004 
```
### Model Object
Finally, we pass the parameters to the `maaDDM` constructor to initialize the model.
 ```@example maaDDM
dist = maaDDM(; ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. The `rand` function accepts the model object, the number of simulated trials, the `attend` function, and the transition matrix object. 

 ```@example maaDDM
 choices,rts = rand(dist, 10_000, attend, tmat)
```
## Plot Simulation
Finally, we can generate histograms of the reaction times for each decision option. 
 ```@example maaDDM
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
# References

Yang, X., & Krajbich, I. (2023). A dynamic computational model of gaze and choice in multi-attribute decisions. Psychological Review, 130(1), 52.

Fisher, G. (2021). A multiattribute attentional drift diffusion model. Organizational Behavior and Human Decision Processes, 165, 167-182.