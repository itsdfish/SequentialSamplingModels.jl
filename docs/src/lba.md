# Linear Ballistic Accumulator

The Linear Ballistic Accumulator (LBA; Brown & Heathcote, 2008) is a sequential sampling model in which evidence for options races independently. The LBA makes an additional simplification that evidence accumulates in a linear and ballistic fashion, meaning there is no intra-trial noise. Instead, evidence accumulates deterministically and linearly until it hits the threshold.

# Example
In this example, we will demonstrate how to use the LBA in a generic two alternative forced choice task. 
```@setup lba
using SequentialSamplingModels
using SSMPlots 
using Random

ν=[2.75,1.75]
A = .8
k = .5
τ = .3

dist = LBA(;ν, A, k, τ) 
choices,rts = rand(dist, 100)
plot(dist)
histogram!(dist)
```

## Load Packages
The first step is to load the required packages.

```@example lba
using SequentialSamplingModels
using SSMPlots 
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Drift Rates

The drift rates control the speed with which information accumulates. Typically, there is one drift rate per option. 

```@example lba
ν=[2.75,1.75]
```

### Maximum Starting Point

The starting point of each accumulator is sampled uniformly between $[0,A]$.

```@example lba 
A = 0.80
```
### Threshold - Maximum Starting Point

Evidence accumulates until accumulator reaches a threshold $\alpha = k +A$. The threshold is parameterized this way to faciliate parameter estimation and to ensure that $A \le \alpha$.
```@example lba 
k = 0.50
```
### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time. 
```@example lba 
τ = 0.30
```
### LBA Constructor 

Now that values have been asigned to the parameters, we will pass them to `LBA` to generate the model object.

```@example lba 
dist = LBA(; ν, A, k, τ) 
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example lba 
 choices,rts = rand(dist, 10_000)
```
## Compute PDF
The PDF for each observation can be computed as follows:
 ```@example lba 
pdf.(dist, choices, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example lba 
logpdf.(dist, choices, rts)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example lba 
histogram(dist)
plot!(dist; t_range=range(.3,2.5, length=100), xlims=(0, 2.5))

```
# References

Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice response time: Linear ballistic accumulation. Cognitive psychology, 57(3), 153-178.