# Starting-time Drift Diffusion Model (stDDM)

The relative starting time drift diffusion model (stDDM) characterizes the contributions of multiple unique attributes to the rate of evidence accumulation. Compared to the DDM, which assumes a constant evidence accumulation rate within each trial, the stDDM allows different attributes to enter the evidence accumulation process at various time points relative to one another. By doing so, the stDDM quantifies both the weights given to each attribute and their onset times (Amasino et al., 2019; Barakchian et al., 2021; Chen et al., 2022; Maier et al., 2020; Sullivan and Huettel, 2021).

# Example
In this example, we will demonstrate how to use the stDDM in a generic two-alternative forced-choice task with two arbitrary attributes.

## Load Packages
The first step is to load the required packages.

```@example stDDM
using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
```

## Create Model Object
In the code below, we will define parameters for the stDDM and create a model object to store the parameter values. 

### Drift Rates
The drift rates control the speed and direction with which information accumulates, with one drift rate per attribute (e.g., taste and health, payoff and delay, self and other).
```@example stDDM
ν = [2.5,2.0]
```

### Threshold
The threshold α represents the amount of evidence required to make a decision.
```@example stDDM 
α = 1.5
```

### Non-Decision Time
Non-decision time is an additive constant representing encoding and motor response time. 
```@example stDDM 
τ = 0.30
```

### starting time
The starting time parameter \(s\) denotes how much earlier one attribute begins to affect the evidence accumulation process relative to the other(s). If \(s\) is negative, attribute 1 evidence is accumulated before attribute 2 evidence; if \(s\) is positive, attribute 1 evidence is accumulated after attribute 2 evidence. The absolute value of \(s\) indicates the difference in starting times for the two attributes.
```@example stDDM 
s = 0.10 
```

### Starting Point
An indicator of an an initial bias towards a decision. The z parameter is relative to a (i.e. it ranges from 0 to 1).
```@example stDDM 
z = 0.50
```

### Drift Rates Dispersion
Dispersion parameters of the drift rate are drawn from a multivariate normal distribution, with the mean vector ν describing the distribution of actual drift rates from specific trials. The standard deviation or across-trial variability is captured by the η vector, and the corresponding correlation between the two attributes is denoted by ρ.
```@example stDDM
η = [1.0,1.0]
ρ = 0.3
```

### Diffusion Noise
Diffusion noise is the amount of within trial noise in the evidence accumulation process. 
```@example stDDM 
σ = 1.0
```
### Time Step
The time step parameter $\Delta t$ is the precision of the discrete time approxmation. 
```@example stDDM 
Δt = .001
```

### stDDM Constructor 
Now that values have been asigned to the parameters, we will pass them to `stDDM` to generate the model object.
```@example stDDM 
dist = stDDM(;ν, α, τ, s, z, η, ρ, σ, Δt)
```

## Simulate Model
Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`. 
 ```@example stDDM 
 choices,rts = rand(dist, 10_000)
```

## Compute Choice Probability
The choice probability $\Pr(C=c)$ is computed by passing the model and choice index to `cdf`.
 ```@example stDDM 
cdf(dist, 1)
```
To compute the joint probability of choosing $c$ within $t$ seconds, i.e., $\Pr(T \leq t \wedge C=c)$, pass a third argument for $t$.

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example stDDM 
histogram(dist)
plot!(dist; t_range=range(.301, 3.0, length=100))
```

# References

Amasino, D.R., Sullivan, N.J., Kranton, R.E. et al. Amount and time exert independent influences on intertemporal choice. Nat Hum Behav 3, 383–392 (2019). https://doi.org/10.1038/s41562-019-0537-2

Barakchian, Z., Beharelle, A.R. & Hare, T.A. Healthy decisions in the cued-attribute food choice paradigm have high test-retest reliability. Sci Rep, (2021). https://doi.org/10.1038/s41598-021-91933-6

Chen, HY., Lombardi, G., Li, SC. et al. Older adults process the probability of winning sooner but weigh it less during lottery decisions. Sci Rep, (2022). https://doi.org/10.1038/s41598-022-15432-y

Maier, S.U., Raja Beharelle, A., Polanía, R. et al. Dissociable mechanisms govern when and how strongly reward attributes affect decisions. Nat Hum Behav 4, 949–963 (2020). https://doi.org/10.1038/s41562-020-0893-y

Sullivan, N.J., Huettel, S.A. Healthful choices depend on the latency and rate of information accumulation. Nat Hum Behav 5, 1698–1706 (2021). https://doi.org/10.1038/s41562-021-01154-0