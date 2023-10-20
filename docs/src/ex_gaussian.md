# Ex-Gaussian Model

The Ex-Gaussian is the convolution of a Gaussian and exponential distribution sometimes used to model reaction time distributions:

$\mathrm{rt} \sim \mathrm{normal}(\mu,\sigma) + \mathrm{exponential}(\tau)$

When the Ex-Gaussian was initially developed, some researchers thought that the Gaussian and exponential components represented motor and decision processes, respectively. More recent evidence casts doubt on this interpretation and shows that the parameters do not have a simple mapping to psychologically distinct processes in the drift diffusion model (Matzke & Wagenmakers, 2009). Perhaps this is unsurprising given that the models do not have the same number of parameters. Although the Ex-Gaussian is not technically a sequential sampling model, it is included in the package due to its historical role in reaction time modeling and its simple implementation. 

# Example
In this example, we will demonstrate how to use the Ex-Gaussian for a simulated detection task in which a stimulus appears and the subject responds as quickly as possible.
```@setup ex_gaussian
using SequentialSamplingModels
using Plots 
using Random
```

## Load Packages
The first step is to load the required packages.

```@example ex_gaussian
using SequentialSamplingModels
using Plots 
using Random

Random.seed!(21095)
```
## Create Model Object
In the code below, we will define parameters for the LBA and create a model object to store the parameter values.

### Mean of Gaussian Component

The parameter $\mu$ represents the mean processing time of each accumulator in log space.

```@example ex_gaussian
μ = .80
```

### Standard Deviation of Gaussian Component

The parameter $\sigma$ represents the standard deviation of the Gaussian component.

```@example ex_gaussian
σ = .20
```
### Mean of Exponential Component
The parameter $\tau$ represents the mean of the exponential component.
```@example ex_gaussian
τ = 0.30
```
### Ex-Gaussian Constructor

Now that values have been asigned to the parameters, we will pass them to `ExGaussian` to generate the model object.

```@example ex_gaussian
dist = ExGaussian(μ, σ, τ)
```
## Simulate Model

Now that the model is defined, we will generate $10,000$ choices and reaction times using `rand`.

 ```@example ex_gaussian
rts = rand(dist, 10_000)
```
## Compute PDF
The PDF for each observation can be computed as follows:
 ```@example ex_gaussian
pdf.(dist, rts)
```

## Compute Log PDF
Similarly, the log PDF for each observation can be computed as follows:

 ```@example ex_gaussian
logpdf.(dist, rts)
```

## Plot Simulation
The code below overlays the PDF on reaction time histograms for each option.
 ```@example ex_gaussian
histogram(dist)
plot!(dist; t_range=range(.301, 2.5, length=100))
```
# References

Matzke, D., & Wagenmakers, E. J. (2009). Psychological interpretation of the ex-Gaussian and shifted Wald parameters: A diffusion model analysis. Psychonomic bulletin & review, 16, 798-817.