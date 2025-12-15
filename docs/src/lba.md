# Linear Ballistic Accumulator

The Linear Ballistic Accumulator (LBA; Brown & Heathcote, 2008) is a sequential sampling model in which evidence for options races independently. The LBA makes an additional simplification that evidence accumulates in a linear and ballistic fashion, meaning there is no intra-trial noise. Instead, evidence accumulates deterministically and linearly until it hits the threshold.

# Example
In this example, we will demonstrate how to use the LBA in a generic two alternative forced choice task. 
```@setup lba
using SequentialSamplingModels
using Plots 
using Random
```

## Load Packages
The first step is to load the required packages.

```@example lba
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values. 

### Mean Drift Rates

The drift rates control the speed with which evidence accumulates for each option. In the standard LBA, drift rates vary across trials according to a normal distribution with mean $\nu$:

```@example lba
ν = [2.75,1.75]
```

### Standard Deviation of Drift Rates

The standard deviation of the drift rate distribution is given by $\sigma$, which is commonly fixed to 1 for each accumulator.

```@example lba
σ = [1.0,1.0]
```

### Maximum Starting Point

The starting point of each accumulator is sampled uniformly between $[0,A]$.

```@example lba 
A = 0.80
```
### Threshold - Maximum Starting Point

Evidence accumulates until accumulator reaches a threshold $\alpha = k + A$. The threshold is parameterized this way to faciliate parameter estimation and to ensure that $A \le \alpha$.
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

## Compute Choice Probability
The choice probability $\Pr(C=c)$ is computed by passing the model and choice index to `cdf` along with a large value for time as the second argument.
 ```@example lba 
cdf(dist, 1, 100)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example lba 
histogram(dist)
plot!(dist; t_range=range(.3, 2.5, length=100), xlims=(0, 2.5))

```
# References

Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice response time: Linear ballistic accumulation. Cognitive psychology, 57(3), 153-178.