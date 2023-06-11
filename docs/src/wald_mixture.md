# Wald Mixture Model

The Wald mixture model is a sequential sampling model for single choice decisions. It extends the Wald model by allowing the drift rate to vary randomly across trials. 

# Example
In this example, we will demonstrate how to use the Wald mixture model in a generic single choice decision task. 
```@setup wald_mixture
using SequentialSamplingModels
using Plots
using Random

ν = 3.0
σ = 0.20
α = 0.50
θ = 0.130

dist = WaldMixture(ν, σ, α, θ)

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

```@example wald_mixture_mixture
using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Drift Rate

The parameter $\nu$ represents the evidence accumulation rate.

```@example wald_mixture
ν = 3.0
```

### Drift Rate

The parameter $\nu$ represents the evidence accumulation rate.

```@example wald_mixture
ν = 3.0
```

### Drift Rate Variability

The parameter $\sigma$ represents the standard deviation of the evidence accumulation rate across trials.

```@example wald_mixture
σ = 0.20
```

### Threshold

The parameter $\alpha$ the amount of evidence required to make a decision.

```@example wald_mixture 
α = 0.50
```
### Non-Decision Time
Non-decision time is an additive constant representing encoding and motor response time. 
```@example wald_mixture 
θ = 0.130
```
### Wald Constructor 

Now that values have been asigned to the parameters, we will pass them to `WaldMixture` to generate the model object.

```@example wald_mixture 
dist = WaldMixture(ν, σ, α, θ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example wald_mixture 
rts = rand(dist, 1000)
```

## Compute  PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example wald_mixture 
pdf.(dist, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example wald_mixture 
logpdf.(dist, rts)
```

## Plot Simulation
The code below overlays the PDF on reaction time histogram.
 ```@example wald_mixture 
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

Steingroever, H., Wabersich, D., & Wagenmakers, E. J. (2021). Modeling across-trial variability in the Wald drift rate parameter. Behavior Research Methods, 53, 1060-1076.

