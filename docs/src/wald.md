# Wald Model

The Wald model, also known as the inverse Gaussian, a sequential sampling model for single choice decisions. It is formally equivalent to a drift diffusion model with one decision threshold and no starting point or across trial drift rate variability.

# Example
In this example, we will demonstrate how to use the Wald model in a generic single choice decision task. 
```@setup wald
using SequentialSamplingModels
using SSMPlots 
using Random

ν = 3.0
α = 0.50
τ = 0.130

dist = Wald(ν, α, τ)

rts = rand(dist, 10_000)
```

## Load Packages
The first step is to load the required packages.

```@example wald
using SequentialSamplingModels
using SSMPlots 
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Drift Rate

The parameter $\nu$ represents the evidence accumulation rate.

```@example wald
ν = 3.0
```
### Threshold

The parameter $\alpha$ the amount of evidence required to make a decision.

```@example wald 
α = 0.50
```
### Non-Decision Time
Non-decision time is an additive constant representing encoding and motor response time. 
```@example wald 
τ = 0.130
```
### Wald Constructor 

Now that values have been asigned to the parameters, we will pass them to `Wald` to generate the model object.

```@example wald 
dist = Wald(ν, α, τ)
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
The code below overlays the PDF on reaction time histogram.
 ```@example wald 
histogram(dist)
plot!(dist; t_range=range(.130, 1, length=100))
```
# References

Anders, R., Alario, F., & Van Maanen, L. (2016). The shifted Wald distribution for response time data analysis. Psychological methods, 21(3), 309.

Folks, J. L., & Chhikara, R. S. (1978). The inverse Gaussian distribution and its statistical application—a review. Journal of the Royal Statistical Society: Series B (Methodological), 40(3), 263-275.

Steingroever, H., Wabersich, D., & Wagenmakers, E. J. (2021). Modeling across-trial variability in the Wald drift rate parameter. Behavior Research Methods, 53, 1060-1076.

