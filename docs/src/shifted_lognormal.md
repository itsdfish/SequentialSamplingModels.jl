# Shifted Lognormal Model

The shifted lognormal model a sequential sampling model for single choice decisions in which the first passage time follows a lognormal distribution. The decision time distribution is shifted by a constant representing encoding and response execution processing time. Note that the shifted lognormal model is a special case of the [log normal race model](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lnr/) with a single accumulator. 

# Example
In this example, we will demonstrate how to use the shifted lognormal model in a generic single choice decision task. 
```@setup shifted_lognormal
using SequentialSamplingModels
using Plots 
using Random
```

## Load Packages
The first step is to load the required packages.

```@example shifted_lognormal
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Mean Log Time

The parameter $\nu$ represents the mean processing time of the accumulator in log space.

```@example shifted_lognormal
ν = -1
```

### Log Standard Deviation

The parameter $\sigma$ represents the standard deviation of processing time in log space.

```@example shifted_lognormal
σ = .50
```
### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time.
```@example shifted_lognormal
τ = 0.30
```
### Shifted Lognormal Constructor 

Now that values have been asigned to the parameters, we will pass them to `shifted lognormal` to generate the model object.

```@example shifted_lognormal 
dist = ShiftedLogNormal(ν, σ, τ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example shifted_lognormal 
rts = rand(dist, 1000)
```

## Compute  PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example shifted_lognormal 
pdf.(dist, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

```@example shifted_lognormal 
logpdf.(dist, rts)
```

## Compute CDF
The cumulative probability density $\Pr(T \leq t)$ is computed by passing the model and a value $t$ to `cdf`.

```@example shifted_lognormal 
cdf(dist, .75)
```

## Plot Simulation
The code below overlays the PDF on reaction time histogram.
```@example shifted_lognormal 
histogram(dist)
plot!(dist; t_range=range(.130, 1.5, length=100))
```
# References

Heathcote, A., & Bohlscheid, E. Analysis and Modeling of Response Time using the Shifted Lognormal Distribution.

