# Diffusion Decision Model

The Diffusion Decision Model (DDM; Ratcliff et al., 2016) is a model of speeded decision-making in two-choice tasks. The DDM assumes that evidence accumulates over time, starting from a certain position, until it crosses one of two boundaries and triggers the corresponding response (Ratcliff & McKoon, 2008; Ratcliff & Rouder, 1998; Ratcliff & Smith, 2004). Like other Sequential Sampling Models, the DDM comprises psychologically interpretable parameters that collectively form a generative model for reaction time distributions of both responses.

The drift rate (ν) determines the rate at which the accumulation process approaches a decision boundary, representing the relative evidence for or against a specific response. The distance between the two decision boundaries (referred to as the evidence threshold, α) influences the amount of evidence required before executing a response. Non-decision-related components, including perceptual encoding, movement initiation, and execution, are accounted for in the DDM and reflected in the τ parameter. Lastly, the model incorporates a bias in the evidence accumulation process through the parameter z, affecting the starting point of the drift process in relation to the two boundaries. The z parameter in DDM is relative to a (i.e. it ranges from 0 to 1).

One last parameter is the within-trial variability in drift rate (σ), or the diffusion coefficient. The diffusion coefficient is the standard deviation of the evidence accumulation process within one trial. It is a scaling parameter and by convention it is kept fixed. Following Navarro & Fuss, (2009), we use the σ = 1 version.

# Example
In this example, we will demonstrate how to use the DDM in a generic two alternative forced choice task.

## Load Packages
The first step is to load the required packages.

```@setup DDM
using SequentialSamplingModels
using Plots 
using Random
```

```@example DDM
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(8741)
```

## Create Model Object
In the code below, we will define parameters for the DDM and create a model object to store the parameter values. 

### Drift Rate

The average slope of the information accumulation process. The drift gives information about the speed and direction of the accumulation of information. Typical range: -5 < ν < 5

```@example DDM
ν=1.0
```

### Boundary Separation

The amount of information that is considered for a decision. Large values indicates response caution. Typical range: 0.5 < α < 2

```@example DDM 
α = 0.80
```

### Non-Decision Time

The duration for a non-decisional processes (encoding and response execution). Typical range: 0.1 < τ < 0.5 

```@example DDM 
τ = 0.30
```

### Starting Point

An indicator of an an initial bias towards a decision. The z parameter is relative to a (i.e. it ranges from 0 to 1).

```@example DDM 
z = 0.50
```

### DDM Constructor 

Now that values have been assigned to the parameters, we will pass them to `DDM` to generate the model object.

```@example DDM 
dist = DDM(ν, α, τ, z)
```

## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example DDM 
 choices,rts = rand(dist, 10_000)
```

## Compute PDF
The PDF for each observation can be computed as follows:

 ```@example DDM 
pdf.(dist, choices, rts)
```
## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example DDM 
logpdf.(dist, choices, rts)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.

 ```@example DDM 
histogram(dist)
plot!(dist; t_range=range(.301, 1, length=100))
```

# References

Navarro, D., & Fuss, I. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. https://doi.org/10.1016/J.JMP.2009.02.003

Ratcliff, R., & McKoon, G. (2008). The Diffusion Decision Model: Theory and Data for Two-Choice Decision Tasks. Neural Computation, 20(4), 873–922. https://doi.org/10.1162/neco.2008.12-06-420

Ratcliff, R., & Rouder, J. N. (1998). Modeling Response Times for Two-Choice Decisions. Psychological Science, 9(5), 347–356. https://doi.org/10.1111/1467-9280.00067

Ratcliff, R., & Smith, P. L. (2004). A comparison of sequential sampling models for two-choice reaction time. Psychological Review, 111 2, 333–367. https://doi.org/10.1037/0033-295X.111.2.333

Ratcliff, R., Smith, P. L., Brown, S. D., & McKoon, G. (2016). Diffusion Decision Model: Current Issues and History. Trends in Cognitive Sciences, 20(4), 260–281. https://doi.org/10.1016/j.tics.2016.01.007
