# Lognormal Race Model

The Lognormal Race model (LNR) assumes evidence for each option races independently and that the first passage time for each option is lognormally distributed. One way in which the LNR has been used is to provide a likelihood function for the ACT-R cognitive architecture. An example of such an application can be found in [ACTRModels.jl](https://itsdfish.github.io/ACTRModels.jl/dev/example2/). We will present a simplified version below.

# Example
In this example, we will demonstrate how to use the LNR in a generic two alternative forced choice task.
```@setup lnr
using SequentialSamplingModels
using Plots
using Random

μ = [-1,-1.5]
σ = 0.50
ϕ = 0.30

dist = LNR(μ, σ, ϕ)

choices,rts = rand(dist, 1000)

# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
t_range = range(.31, 1, length=100)
# pdf for choice 1
pdf1 = pdf.(dist, (1,), t_range)
# pdf for choice 2
pdf2 = pdf.(dist, (2,), t_range)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,1.5))
histogram!(rts1, subplot=1, color=:grey, bins = 100, norm=true, title="Choice 1")
plot!(t_range, pdf1, subplot=1, color=:darkorange, linewidth=2)
histogram!(rts2, subplot=2, color=:grey, bins = 100, norm=true, title="Choice 2")
plot!(t_range, pdf2, subplot=2, color=:darkorange, linewidth=2)
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
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

### Mean Log Time

The parameter $\mu$ represents the mean processing time of each accumulator in log space.

```@example lnr
μ = [-1,-1.5]
```

### Log Standard Deviation

The parameter $\sigma$ represents the standard deviation of processing time in log space.

```@example lnr
σ = 0.50
```
### Non-Decision Time
Non-decision time is an additive constant representing encoding and motor response time.
```@example lnr
ϕ = 0.30
```
### LNR Constructor

Now that values have been asigned to the parameters, we will pass them to `LNR` to generate the model object.

```@example lnr
dist = LNR(μ, σ, ϕ)
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

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example lnr
# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
t_range = range(.31, 1, length=100)
# pdf for choice 1
pdf1 = pdf.(dist, (1,), t_range)
# pdf for choice 2
pdf2 = pdf.(dist, (2,), t_range)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,1.5))
histogram!(rts1, subplot=1, color=:grey, bins = 100, norm=true, title="Choice 1")
plot!(t_range, pdf1, subplot=1, color=:darkorange, linewidth=2)
histogram!(rts2, subplot=2, color=:grey, bins = 100, norm=true, title="Choice 2")
plot!(t_range, pdf2, subplot=2, color=:darkorange, linewidth=2)
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
```
# References

Heathcote, A., & Love, J. (2012). Linear deterministic accumulator models of simple choice. Frontiers in psychology, 3, 292.

Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). The lognormal race: A cognitive-process model of choice and latency with desirable psychometric properties. Psychometrika, 80, 491-513.