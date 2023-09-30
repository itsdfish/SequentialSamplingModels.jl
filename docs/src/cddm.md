# Circular Drift Diffusion Model

The Circular Drift Diffusion Model (CDDM; Brown & Heathcote, 2008) is a sequential sampling model for continuous responding on a circular domain. The CDDM is often used to model visual working memory. In these visual working memory tasks, subjects are briefly presented with a variable number of squares of different colors. After the stimuli are removed, subjects are prompted to use a color wheel to judge the color of a randomly selected square. Currently, the model is restricted to a 2D disk, but future versions may support modeling diffusion processes in hyperspheres. 

# Example
In this example, we will demonstrate how to use the CDDM in a generic two alternative forced choice task. 
```@setup CDDM
using LinearAlgebra
using SequentialSamplingModels
using SSMPlots 
using Random
using Revise
```

## Load Packages
The first step is to load the required packages.

```@example CDDM
using LinearAlgebra
using SequentialSamplingModels
using SSMPlots 
using Random

Random.seed!(5874)
```
## Create Model Object
In the code below, we will define parameters for the CDDM and create a model object to store the parameter values. 

### Drift Rates

The mean drift rates $\boldsymbol{\nu}$ control the speed with which information accumulates in the x and y direction.

```@example CDDM
ν = [1.5,1.0]
```
The magnitude of the mean drift rate vector $||\boldsymbol{\nu}||$ is interpreted as the mean accumulation rate.

```@example CDDM
norm(ν)
```
The average direction of the accumulation process is given by $\mathrm{arctan}(\frac{\nu_2}{\nu_1})$:
```@example CDDM
atan(ν[2], ν[1])
```

### Drift Rate Standard Deviation

The standard deviation of the drift rate $\boldsymbol{\eta}$ is inteprpreted as variability in the evidence accumulation across trials. 

```@example CDDM 
η = [.50,.50]
```
### Threshold

Evidence starts at the center of a circle $(0,0)$ and terminates at a threshold defined by the circumference of the circle. The distance between the starting point and any point on the circumference is given by the radius $\alpha$:
```@example CDDM 
α = 1.50
```
### Diffusion

Intra-trial variability in the accumulation process is governed by parameter $\sigma$
```@example CDDM 
σ = 1.0
```

### Non-Decision Time

Non-decision time is an additive constant representing encoding and motor response time. 
```@example CDDM 
τ = 0.30
```
### CDDM Constructor 

Now that values have been asigned to the parameters, we will pass them to `CDDM` to generate the model object.

```@example CDDM 
dist = CDDM(ν, η, σ, α, τ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 
The simulated data is a 2D array in which the first column contains the observed angular responses and the second column contains the corresponding reaction times.
 ```@example CDDM 
 data = rand(dist, 10_000)
```
## Compute PDF
The PDF for each observation can be computed as follows:
 ```@example CDDM 
pdf(dist, data)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example CDDM 
logpdf(dist, data)
```

## Plot Simulation
The code below overlays the PDF on the marginal histograms for angle and reaction time.
 ```@example CDDM 
#histogram(dist)
#plot!(dist)
```
# References

Smith, P. L. (2016). Diffusion theory of decision making in continuous report. Psychological Review, 123(4), 425.

Smith, P. L., Garrett, P. M., & Zhou, J. (2023). Obtaining Stable Predicted Distributions of Response Times and Decision Outcomes for the Circular Diffusion Model. 
Computational Brain & Behavior, 1-13.