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
τ = 0.20

dist = RDM(;ν, k, A, τ)
choices,rts = rand(dist, 1000)

# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2 
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
t_range = range(.31, 2, length=100)
# pdf for choice 1
pdf1 = pdf.(dist, (1,), t_range)
# pdf for choice 2
pdf2 = pdf.(dist, (2,), t_range)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,1.5))
histogram!(rts1, subplot=1, color=:grey, bins = 200, norm=true, title="Choice 1")
plot!(t_range, pdf1, subplot=1, color=:darkorange, linewidth=2)
histogram!(rts2, subplot=2, color=:grey, bins = 150, norm=true, title="Choice 2")
plot!(t_range, pdf2, subplot=2, color=:darkorange, linewidth=2)
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
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
# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2 
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
t_range = range(.31, 2, length=100)
# pdf for choice 1
pdf1 = pdf.(dist, (1,), t_range)
# pdf for choice 2
pdf2 = pdf.(dist, (2,), t_range)
# histogram of retrieval times
hist = histogram(layout=(2,1), leg=false, grid=false,
     xlabel="Reaction Time", ylabel="Density", xlims = (0,1.5))
histogram!(rts1, subplot=1, color=:grey, bins = 200, norm=true, title="Choice 1")
plot!(t_range, pdf1, subplot=1, color=:darkorange, linewidth=2)
histogram!(rts2, subplot=2, color=:grey, bins = 150, norm=true, title="Choice 2")
plot!(t_range, pdf2, subplot=2, color=:darkorange, linewidth=2)
# weight histogram according to choice probability
hist[1][1][:y] *= p1
hist[2][1][:y] *= (1 - p1)
hist
```
# References

Tillman, G., Van Zandt, T., & Logan, G. D. (2020). Sequential sampling
models without random between-trial variability: The racing diffusion model
of speeded decision making. Psychonomic Bulletin & Review, 27, 911-936.
