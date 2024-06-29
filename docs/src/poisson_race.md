# Poisson Race

The Poisson race model is one of the first sequential sampling models, with origins dating back to 1962. In this model, evidence accumulates in discrete steps until the first accumulator reaches a threshold. The time between increments follows an exponential distribution. The first passage time follows a gamma distribution because it is the sum of exponential random variables.  

# Example
In this example, we will demonstrate how to use the Poisson race model in a generic two alternative forced choice task.
```@setup poisson_race
using SequentialSamplingModels
using Plots 
using Random
```

## Load Packages
The first step is to load the required packages.

```@example poisson_race
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(65)
```
## Create Model Object
In the code below, we will define parameters for the Poisson race and create a model object to store the parameter values.

### Mean processing time

The parameter $\nu$ represents the mean processing of each count. Note that $\nu = \frac{1}{\lambda}$, where $\lambda$ is the rate parameter. 

```@example poisson_race
ν = [.04, .05]
```

### Threshold

The parameter $\alpha$ is a vector of thresholds. Each threshold is an integer because it represents a discrete count.

```@example poisson_race
α = [4,4]
```
### Non-Decision Time
Non-decision time is an additive constant representing encoding and motor response time.
```@example poisson_race
τ = 0.30
```
### Poisson race Constructor

Now that values have been asigned to the parameters, we will pass them to `LNR` to generate the model object.

```@example poisson_race
dist = PoissonRace(;ν, α, τ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`.

 ```@example poisson_race
 choices,rts = rand(dist, 10_000)
```
## Compute PDF
The PDF for each observation can be computed as follows:
 ```@example poisson_race
pdf.(dist, choices, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example poisson_race
logpdf.(dist, choices, rts)
```

## Compute Choice Probability
The choice probability $\Pr(C=c)$ is computed by passing the model and choice index to `cdf` along with a large value for time as the second argument.
 ```@example poisson_race 
cdf(dist, 1, Inf)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example poisson_race
histogram(dist)
plot!(dist; t_range=range(.301, 1, length=100))
```
# References

LaBerge, D. A. (1962). A recruitment model of simple behavior. Psychometrika, 27, 375-395.
