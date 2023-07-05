# Ratcliff Diffusion Model

The Ratcliff Diffusion Model (Ratcliff DDM; Ratcliff et al., 2016) is similar to the DDM. Like the DDM, the model assumes that evidence accumulates over time, starting from a certain position, until it crosses one of two boundaries and triggers the corresponding response (Ratcliff & McKoon, 2008; Ratcliff & Rouder, 1998; Ratcliff & Smith, 2004). The drift rate (ν) determines the rate at which the accumulation process approaches a decision boundary, representing the relative evidence for or against a specific response. The distance between the two decision boundaries (referred to as the evidence threshold, α) influences the amount of evidence required before executing a response. Non-decision-related components, including perceptual encoding, movement initiation, and execution, are accounted for in the DDM and reflected in the τ parameter. Lastly, the model incorporates a bias in the evidence accumulation process through the parameter z, affecting the starting point of the drift process in relation to the two boundaries. The z parameter in DDM is relative to a (i.e. it ranges from 0 to 1).

However, the model differs in the inclusion of across-trial variability parameters. These parameters were developed to explain specific discrepancies between the DDM and experimental data (Anderson, 1960; Laming, 1968; Blurton et al., 2017). The data exhibited a difference in mean RT between correct and error responses that could not be captured by the DDM. As a result, two parameters for across-trial variability were introduced to explain this difference: across-trial variability in the starting point to explain fast errors (Laming, 1968), and across-trial variability in drift rate to explain slow errors (Ratcliff, 1978; Ratcliff and Rouder, 1998). Additionally, the DDM also showed a sharper rise in the leading edge of the response time distribution than observed in the data. To capture this leading edge effect, across-trial variability in non-decision time was introduced. 

Previous work has validated predictions of these across-trial variability parameters (Wagenmakers et al., 2009). When compared to the DDM, the Ratcliff DDM improves the fit to the data. Researchers now often assume that the core parameters of sequential sampling models, such as drift rates, non-decision times, and starting points vary between trials.

One last parameter is the within-trial variability in drift rate (σ), or the diffusion coefficient. The diffusion coefficient is the standard deviation of the evidence accumulation process within one trial. It is a scaling parameter and by convention it is kept fixed. Following Navarro & Fuss, (2009), we use the σ = 1 version.

# Example
In this example, we will demonstrate how to use the DDM in a generic two alternative forced choice task.

## Load Packages
The first step is to load the required packages.

```@example RatcliffDDM
using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
```

## Create Model Object
In the code below, we will define parameters for the DDM and create a model object to store the parameter values. 

### Drift Rate

The average slope of the information accumulation process. The drift gives information about the speed and direction of the accumulation of information. Typical range: -5 < ν < 5

Across-trial-variability of drift rate. Standard deviation of a normal distribution with mean v describing the distribution of actual drift rates from specific trials. Values different from 0 can predict slow errors. Typical range: 0 < η < 2. Default is 0.

```@example RatcliffDDM
ν=1.0
η = 0.16
```

### Boundary Separation

The amount of information that is considered for a decision. Large values indicates response caution. Typical range: 0.5 < α < 2


```@example RatcliffDDM 
α = 0.80
```

### Non-Decision Time

The duration for a non-decisional processes (encoding and response execution). Typical range: 0.1 < τ < 0.5 

Across-trial-variability of non-decisional components. Range of a uniform distribution with mean τ + st/2 describing the distribution of actual τ values across trials. Accounts for response times below t0. Reduces skew of predicted RT distributions. Typical range: 0 < τ < 0.2. Default is 0.

```@example RatcliffDDM 
τ = 0.30
st = 0.10
```

### Starting Point

An indicator of an an initial bias towards a decision. The z parameter is relative to a (i.e. it ranges from 0 to 1).

Across-trial-variability of starting point. Range of a uniform distribution with mean z describing the distribution of actual starting points from specific trials. Values different from 0 can predict fast errors. Typical range: 0 < sz < 0.5. Default is 0.

```@example RatcliffDDM 
z = 0.25
sz = 0.05
```

### Ratcliff Diffusion Model Constructor 

Now that values have been assigned to the parameters, we will pass them to `RatcliffDDM` to generate the model object.

```@example RatcliffDDM 
dist = RatcliffDDM(ν, α, τ, z, η, sz, st, σ)
```

## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 

 ```@example RatcliffDDM 
 choices,rts = rand(dist, 10_000)
```

## Compute PDF
The PDF for each observation can be computed as follows:

 ```@example RatcliffDDM 
pdf.(dist, choices, rts)
```
## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example RatcliffDDM 
logpdf.(dist, choices, rts)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.

 ```@example RatcliffDDM 
# rts for option 1
rts1 = rts[choices .== 1]
# rts for option 2 
rts2 = rts[choices .== 2]
# probability of choosing 1
p1 = length(rts1) / length(rts)
t_range = range(.30, 2, length=100)
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

Navarro, D., & Fuss, I. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. https://doi.org/10.1016/J.JMP.2009.02.003

Ratcliff, R., & McKoon, G. (2008). The Diffusion Decision Model: Theory and Data for Two-Choice Decision Tasks. Neural Computation, 20(4), 873–922. https://doi.org/10.1162/neco.2008.12-06-420

Ratcliff, R., & Rouder, J. N. (1998). Modeling Response Times for Two-Choice Decisions. Psychological Science, 9(5), 347–356. https://doi.org/10.1111/1467-9280.00067

Ratcliff, R., & Smith, P. L. (2004). A comparison of sequential sampling models for two-choice reaction time. Psychological Review, 111 2, 333–367. https://doi.org/10.1037/0033-295X.111.2.333

Ratcliff, R., Smith, P. L., Brown, S. D., & McKoon, G. (2016). Diffusion Decision Model: Current Issues and History. Trends in Cognitive Sciences, 20(4), 260–281. https://doi.org/10.1016/j.tics.2016.01.007

Wagenmakers, E.-J. (2009). Methodological and empirical developments for the Ratcliff diffusion model of response times and accuracy. European Journal of Cognitive Psychology, 21(5), 641-671.


