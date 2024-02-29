# Attentional Drift Diffusion Model 

The attentional drift diffusion model (ADDM; Krajbich, Armel, & Rangel, 2010) describes how attentional processes drive drive decision making. In the ADDM, preference for the currently attended option accrues faster than preference for non-attended options. As with other sequential sampling models, the first option to hit a decision threshold determines the resulting choice and reaction time.

# Example

```@setup aDDM
using SequentialSamplingModels
using StatsBase
using Plots
using Random
```

In this example, we will develope a ADDM for binary choice and generate its predictions. Unlike many other sequential sampling models, it is necessary to specify the attentional process, or supply fixation patterns from eye tracking data. 
## Load Packages
The first step is to load the required packages.

```@example aDDM
using SequentialSamplingModels
using StatsBase
using Plots

Random.seed!(5487)
```

## Define Transition Type 

To represent the transition of attention from one option to the other, we will definite a `Transition` type and constructor. The fields of the `Transition` type are:

1. `state`: an index for the current state
2. `n`: the number of states
3. `mat`: an $n\times n$ transition matrix

The constructor accepts a transition matrix, extracts the number of states, and initializes the first state randomly with equal probability.

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

The transition matrix is defined below in the constructor for `Transition`. As shown in the table below, the model's attention can be in one of three states: option 1, option 2, or non-option, which is any area except the two options. 

|             	| option 1  	| option 2 	| non-option  	|
|-------------	|-----------	|----------	|-------------	|
| **option 1**   	| 0.98      	| 0.015    	| 0.005       	|
| **option 2**    	| 0.015     	| 0.98     	| 0.005       	|
| **non-option**  	| 0.45      	| 0.45     	| 0.10         	|


The transition matrix above embodies the following assumptions:

1. Once the model attends to an option, it dwells on the option for some time.
2. There is not a bias for one option over the other.
3. The chance of fixating on a non-option is small, and such fixations are brief when they do occur.
4. Transitions are Markovian in that they only depend on the previous state.



 ```@example aDDM 
tmat = Transition([.98 .015 .005;
                    .015 .98 .005;
                    .45 .45 .1])
```

## Attend Function 

The function below generates the next attention location based on the previous location. 

```@example aDDM 
 function attend(transition)
     (;mat,n,state) = transition
     w = @view mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end
```
## Create Model Object
The code snippets assign values to parameters of the ADDM and create a model object.

### Drift Rate Components
The ADDM has two drift rates components corresponding to the utlity of each option. To form the drift rate, each component is weighted by non-attention bias and then a difference is computed.
```@example aDDM 
ν = [6.0,5.0]
```
### Threshold
The threshold hold represents the amount of evidence required to make a decision. This parameter is typically fixed at $\alpha = 1$.
```@example aDDM 
α = 1.0
```
### Starting Point
The starting point of the evidence accumulation process is denoted $z$ and is typically fixed to $0$.
```@example aDDM 
z = 0.0
```
### Non-Attend Bias
The non-attend bias parameter $\theta$ determines how much the non-attended option contributes to the 
evidence accumulation process. In the standard DDM, $\theta=1$. 
```@example aDDM 
θ = 0.30
```
### Diffusion Noise
Diffusion noise, $\sigma$ represents intra-trial noise during the evidence accumulation process.
```@example aDDM 
σ = 0.02
```
### Drift Rate Scalar
The drift rate scalar controls how quickly evidence accumulates for each option. 
```@example aDDM 
Δ = 0.0004 
```
### Model Object
Finally, we pass the parameters to the `aDDM` constructor to initialize the model.
 ```@example aDDM 
 model = aDDM(; ν, α, z, θ, σ, Δ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. The `rand` function accepts the model object, the number of simulated trials, the `attend` function, and the transition matrix object. 

 ```@example aDDM 
 choices,rts = rand(model, 10_000, attend, tmat)
```
## Plot Simulation
Finally, we can generate histograms of the reaction times for each decision option. 
 ```@example aDDM 
m_args = (attend,tmat)
histogram(model; m_args)
plot!(model; m_args, t_range=range(0.0, 5, length=100), xlims=(0,5))
```
# References

Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of value in simple choice. Nature neuroscience, 13(10), 1292-1298.