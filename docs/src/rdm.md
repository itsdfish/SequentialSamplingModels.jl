# Racing Diffusion Model

The Diffusion Race Model (DRM; Tillman, Van Zandt, & Logan, 2020) is a sequential sampling model in which evidence for options races independently. The DRM is similar to the Linear Ballistic Accumulator, except it assumes noise occurs during the within-trial evidence accumulation process, but the drift rate is constant across trials. 

# Example
In this example, we will demonstrate how to use the DRM in a generic two alternative forced choice task. 
```@setup drm
using SequentialSamplingModels
using Plots
using Random

ν = [1.0,0.50]
k = 0.50
A = 1.0
τ = 0.30

dist = RDM(;ν, k, A, τ)
choices,rts = rand(dist, 1000)
```

## Load Packages
The first step is to load the required packages.

```@example drm
using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
```
## Create Model Object
In the code below, we will define parameters for the DRM and create a model object to store the parameter values. 

### Drift Rates

The drift rates control the speed with which information accumulates. Typically, there is one drift rate per option. 

```@example drm
ν = [1.0,0.50]
```

### Maximum Starting Point

The starting point of each accumulator is sampled uniformly between $[0,A]$.

```@example drm 
A = 0.80
```
### Threshold - Maximum Starting Point

Evidence accumulates until accumulator reaches a threshold $\alpha = k +A$. The threshold is parameterized this way to faciliate parameter estimation and to ensure that $A \le \alpha$.
```@example drm 
k = 0.50
```
### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time. 
```@example drm 
τ  = 0.30
```
### LBA Constructor 

Now that values have been asigned to the parameters, we will pass them to `RDM` to generate the model object.

```@example drm 
dist = RDM(;ν, k, A, τ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example drm 
 choices,rts = rand(dist, 10_000)
```
## Compute PDF
The PDF for each observation can be computed as follows:
 ```@example drm 
pdf.(dist, choices, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example drm 
logpdf.(dist, choices, rts)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example drm 
histogram(dist; xlims=(0,2.5))
plot!(dist; t_range=range(.301, 2.5, length=100))
```
# References

Tillman, G., Van Zandt, T., & Logan, G. D. (2020). Sequential sampling
models without random between-trial variability: The racing diffusion model
of speeded decision making. Psychonomic Bulletin & Review, 27, 911-936.
