# Leaky Competing Accumulator

The Leaky Competing Accumulator (LCA; Usher & McClelland, 2001) is a sequential sampling model in which evidence for options races independently. The LCA is similar to the Linear Ballistic Accumulator (LBA), but additionally assumes an intra-trial noise and leakage (in contrast, the LBA assumes that evidence accumulates in a ballistic fashion, i.e., linearly and deterministically until it hits the threshold).

# Example
In this example, we will demonstrate how to use the LCA in a generic two alternative forced choice task. 
```@setup lca
using SequentialSamplingModels
using Plots 
using Random
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

### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time. 
```@example lca 
τ = 0.30
```
### LCA Constructor 

Now that values have been asigned to the parameters, we will pass them to `LCA` to generate the model object.

```@example lca 
dist = LCA(; ν, α, β, λ, τ, σ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example lca 
 choices,rts = rand(dist, 10_000)
```
In the code block above, `rand` has a keyword argument `Δt` which controls the precision of the discrete approximation. The default value is `Δt = .001`.

## Compute Choice Probability
The choice probability $\Pr(C=c)$ is computed by passing the model and choice index to `cdf` along with a large value for time as the second argument.
 ```@example lca 
cdf(dist, 1, Inf)
```

## Plot Simulation
The code below plots a histogram for each option.
 ```@example lca 
histogram(dist)
```
# References

Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. Psychological Review, 108 3, 550–592. https://doi.org/10.1037/0033-295X.108.3.550
