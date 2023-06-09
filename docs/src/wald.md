# Wald Model

The Wald model is used to model single choice decisions. It is formally equivalent to a drift diffusion model with one decision threshold and no starting point or across trial drift rate variability.

# Example
In this example, we will demonstrate how to use the LNR in a generic two alternative forced choice task. 
```@setup wald
using SequentialSamplingModels
using Plots
using Random

ν = 3.0
α = 0.50
θ = 0.130

dist = Wald(ν, α, θ)

rts = rand(dist, 10_000)
t_range = range(θ, 1, length=100)
pdf1 = pdf.(dist, t_range)
# histogram of retrieval times
hist = histogram(rts, leg=false, grid=false, norm=true,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,1))
plot!(t_range, pdf1, subplot=1, color=:darkorange, linewidth=2)
hist
```

## Load Packages
The first step is to load the required packages.

```@example wald
using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Mean Log Time

The parameter $\mu$ represents the mean processing time of each accumulator in log space.

```@example wald
ν = 3.0
```


dist = Wald(ν, α, θ)
### Threshold

The parameter $\sigma$ repesents the mean processing time in log space.

```@example wald 
α = 0.50
```
### Non-Decision Time
Non-decision time is an additive constant representing encoding and motor response time. 
```@example wald 
θ = 0.130
```
### Wald Constructor 

Now that values have been asigned to the parameters, we will pass them to `LNR` to generate the model object.

```@example wald 
dist = Wald(ν, α, θ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example wald 
rts = rand(dist, 1000)
```

## Compute  PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example wald 
pdf.(dist, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example wald 
logpdf.(dist, rts)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example wald 
rts = rand(dist, 10_000)
t_range = range(θ, 1, length=100)
pdf1 = pdf.(dist, t_range)
# histogram of retrieval times
hist = histogram(rts, leg=false, grid=false, norm=true, color=:grey,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,1))
plot!(t_range, pdf1, subplot=1, color=:darkorange, linewidth=2)
hist
```
# References

Heathcote, A., & Love, J. (2012). Linear deterministic accumulator models of simple choice. Frontiers in psychology, 3, 292.

Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). The lognormal race: A cognitive-process model of choice and latency with desirable psychometric properties. Psychometrika, 80, 491-513.