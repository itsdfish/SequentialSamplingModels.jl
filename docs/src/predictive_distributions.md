# Prior and Posterior Predictive Distributions

This tutorial explains the steps required for constructing and plotting prior and posterior predictive distributions of a sequential sampling models (SSMs). The primary function we will be using is `predict_distribution`, which allows you to generate prior or posterior predictive distributions from a given model. 

## Full Code 

You can reveal copy-and-pastable version of the full code by clicking the ▶ below.

```@raw html
<details>
<summary><b>Show Full Code</b></summary>
```
```julia
using Distributions
using Plots
using Random
using SequentialSamplingModels
using Turing
using TuringUtilities
Random.seed!(1124)

n_samples = 50
rts = rand(Wald(ν = 1.5, α = 0.8, τ = 0.3), n_samples)

@model function wald_model(rts)
    ν ~ truncated(Normal(1.5, 1), 0, Inf)
    α ~ truncated(Normal(0.8, 1), 0, Inf)
    τ = 0.3
    rts ~ Wald(ν, α, τ)
    return (; ν, α, τ)
end

model = wald_model(rts)

prior_chain = sample(model, Prior(), 1000)

pred_model = predict_distribution(;
    simulator = Θ -> rand(Wald(; Θ...), n_samples),
    model,
    func = mean
)
prior_preds = returned(pred_model, prior_chain)

post_chain = sample(model, NUTS(1000, 0.85), 1000)
post_preds = returned(pred_model, post_chain)

histogram(
    prior_preds[:],
    xlims = (0, 4),
    xlabel = "Mean RT",
    ylabel = "Density",
    norm = true,
    color = :grey,
    label = "prior",
    grid = false
)

histogram!(
    post_preds[:],
    alpha = 0.7,
    color = :darkred,
    norm = true,
    label = "posterior",
    grid = false
)

vline!([mean(rts)], linestyle = :dash, color = :black, linewidth = 2, label = "data")

pred_quantiles = predict_distribution(;
    simulator = Θ -> rand(Wald(; Θ...), n_samples),
    model,
    func = compute_quantiles
)
post_quantile_preds = returned(pred_quantiles, post_chain)
q_data = compute_quantiles(rts)
plot_quantiles(q_data, post_quantile_preds)
```
```@raw html
</details>
```

# Example 
The first step is to load the required packages and set the seed for the random number generator.

```julia 
using Distributions
using Plots
using Random
using SequentialSamplingModels
using Turing
using TuringUtilities
Random.seed!(1124)
```

## Generate Simulated Data
We will use the [Wald](wald.md) model as a simple example to illustrate how to create predictive distributions. The `Wald` model describes the evidence accumulation process underlying single detection decisions, such as respending when a stimulus appears. In the code block below, we will generate 50 data points.
```julia
n_samples = 50
rts = rand(Wald(ν = 1.5, α = 0.8, τ = 0.3), n_samples)
```

## Define Turing Model 

Next, we will develop a Turing model for generating prior and posterior predictive distributions. You may develop the Turing model as usual, with one minor exception: you must return a `NamedTuple` of parameters. In the example below, $\nu$ and $\alpha$ are estimated, but $\tau$ is fixed. You may use any combination of estimated and fixed parameters.  

```julia
@model function wald_model(rts)
    ν ~ truncated(Normal(1.5, 1), 0, Inf)
    α ~ truncated(Normal(0.8, 1), 0, Inf)
    τ = 0.3
    rts ~ Wald(ν, α, τ)
    return (; ν, α, τ)
end
```
In the next code block, we will pass the data and create a model object.
```julia 
model = wald_model(rts)
```
## Generate Prior Predictive Distribution
Generating a prior predictive distribution involves two steps: (1) sample from the prior, and (2) predict data or a statistic with the model evaluated at the prior samples. Below, we will sample 1,000 parameter vectors from the model. 
```julia 
prior_chain = sample(model, Prior(), 1000)
```

For the next step, we will generate predictions from the model using the parameters sampled from the prior distribution. When `Turing` is loaded, `SequentialSamplingModels` automatically loads `predict_distribution` into your session. The signature for `predict_distribution` is as follows:

```julia 
predict_distribution(args...; simulator, model, func, kwargs...)
```
`
The function `simulator` accepts a `NamedTuple` of sampled parameters and returns simulated data. `func` computes a statistic from simulated data of the model and has the general form `func(sim_data, args...; kwargs...)`. Thus, the only constraint is that `func` must recieve the simulated data as its first argument. `args...` and `kwargs...` are optionally pased to `func`.  The keyword `model` the Turing model object. 

As a simple illustration, we will compute the prior predictive mean by calling the following two functions. The first function creates a new function to sample from the predictive distribution and the second function `returned` performs the sampling.

```julia 
pred_model = predict_distribution(;
    simulator = Θ -> rand(Wald(; Θ...), n_samples),
    model,
    func = mean
)
prior_preds = returned(pred_model, prior_chain)
```

## Generate Posterior Predictive Distribution 

Generating a posterior predictive distribution involves a similar process. First, we will estimate the parameters from the data to obtain a chain of posterior samples. Next, we will generate the posterior predictive distribution using `returned`:

```julia 
post_chain = sample(model, NUTS(1000, 0.85), 1000)
post_preds = returned(pred_model, post_chain)
```

## Plot the Distributions

Now that we have generated the predictive distributions, we can compare them to the data by plotting them as a histogram. The histogram below reveals two insights: first, the data are centered near the prior and posterior predictive distributions, indicating they predict the data accurately; second, the posterior distribution is concentrated more closely around the data, indicating the information gain acquired during parameter estimation. 

```julia
histogram(
    prior_preds[:],
    xlims = (0, 4),
    xlabel = "Mean RT",
    ylabel = "Density",
    norm = true,
    color = :grey,
    label = "prior",
    grid = false
)

histogram!(
    post_preds[:],
    alpha = 0.7,
    color = :darkred,
    norm = true,
    label = "posterior",
    grid = false
)

vline!([mean(rts)], linestyle = :dash, color = :black, linewidth = 2, label = "data")

```
![](assets/wald_predictive_means.png)
## Posterior Predictive Distribution of Quantiles
One goal of SSMs is to accurately characterize the distribution of reaction times. The previous example only evaluated one aspective of the model---namely, the predicted mean. Given the interest in characterizing the shape of the RT distribution, we need a different method. One method for evaluating the model's ability to capture the shape of the distribution is to compare the quantiles. In the example below, the quantiles of the data and model are evaluated at the deciles: $[.1,.2,\dots, .9]$. If the model matches the data accurately, the quantiles will fall along the identity line.  

```julia
pred_quantiles = predict_distribution(;
    simulator = Θ -> rand(Wald(; Θ...), n_samples),
    model,
    func = compute_quantiles
)
post_quantile_preds = returned(pred_quantiles, post_chain)
q_data = compute_quantiles(rts)
plot_quantiles(q_data, post_quantile_preds)
```

![](assets/wald_predictive_qq_plots.png)

The posterior predictive quantile-quantile plot above shows that the model fits the reaction time distribution well. This close match is to be expected, as we generated the data from the same model. 
