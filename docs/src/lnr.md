# Lognormal Race Model

The Lognormal Race model (LNR) assumes evidence for each option races independently and that the first passage time for each option is lognormally distributed. One way in which the LNR has been used is to provide a likelihood function for the ACT-R cognitive architecture. An example of such an application can be found in [ACTRModels.jl](https://itsdfish.github.io/ACTRModels.jl/dev/example2/). We will present a simplified version below.

# Example
In this example, we will demonstrate how to use the LNR in a generic two alternative forced choice task.
```@setup lnr
using SequentialSamplingModels
using Plots 
using Random
```

## Load Packages
The first step is to load the required packages.

```@example lnr
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values.

### $\nu$ 

We will set $\nu [-1,-1.5]$.

```@example lnr
ν = [-1,-1.5]
```

### $\sigma$

We will set the parameter $\sigma = [-1,-1.5]$ 

```@example lnr
σ = [0.50,0.50]
```

The lognormal has the following relationship to the normal distribution:

$X \sim \mathrm{lognormal(\nu, \sigma)} \iff \log(X) \sim \mathrm{normal}(\nu, \sigma)$. 

This means that $E[\log(X)] = \nu$ and $\mathrm{Var}[\log(X)] = \sigma^2$. Note that $\nu$ and $\sigma$ affect both the mean and variance of the lognormal distribution. See ACT-R for a possible theoretical intepretation of parameters $\nu$ and $\sigma$.

### Non-Decision Time
Non-decision time is an additive constant representing encoding and motor response time.
```@example lnr
τ = 0.30
```
### LNR Constructor

Now that values have been asigned to the parameters, we will pass them to `LNR` to generate the model object.

```@example lnr
dist = LNR(ν, σ, τ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`.

 ```@example lnr
 choices,rts = rand(dist, 10_000)
```
## Compute PDF
The PDF for each observation can be computed as follows:
 ```@example lnr
pdf.(dist, choices, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example lnr
logpdf.(dist, choices, rts)
```

## Compute Choice Probability
The choice probability $\Pr(C=c)$ is computed by passing the model and choice index to `cdf`.

 ```@example lnr 
cdf(dist, 1, Inf)
```
To compute the joint probability of choosing $c$ within $t$ seconds, i.e., $\Pr(T \leq t \wedge C=c)$, pass a third argument for $t$.

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example lnr
histogram(dist)
plot!(dist; t_range=range(.301, 1, length=100))
```
# References

Heathcote, A., & Love, J. (2012). Linear deterministic accumulator models of simple choice. Frontiers in psychology, 3, 292.

Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). The lognormal race: A cognitive-process model of choice and latency with desirable psychometric properties. Psychometrika, 80, 491-513.