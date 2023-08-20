# Estimate Effect on Drift Rate

This advanced example illustrates how to estimate the effect of an experimental condition on the drift rate parameter. The drift rate could be manipulated in various ways. For example, the drift rate could be manipulated by varying the similarity of visual stimuli, or emphasizing speed or accuracy in task instructions.

## Generate Data

In this example, we will get closer to real use-cases by starting with the data stored in a `DataFrame`. This dataframe will be a combination of data generated from two different distributions with different parameters, corresponding to two experimental conditions (e.g., Speed vs. Accuracy).

```@example turing_advanced
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
using Distributions
using DataFrames
using StatsPlots
using StatsModels
using KernelDensity

# Generate data with different drifts for two conditions A vs. B
Random.seed!(6)

n_obs = 50
df1 = DataFrame(rand(LBA(ν=[1.5, 0.5], A=0.5, k=0.2, τ=0.3), n_obs))
df2 = DataFrame(rand(LBA(ν=[2.5, 1.5], A=0.5, k=0.2, τ=0.3), n_obs))
df = vcat(df1, df2)
df.condition = repeat(["A", "B"], inner=n_obs)
```

These 2 conditions *A* and *B* differ on their drift rates (`[1.5, 0.5]` vs. `[2.5, 1.5]`). In other words, the *effect* of condition *B* over condition *A* (the baseline condition, i.e., the *intercept*) is `[1, 1]` (because both drift rates increase by 1 between condition *A* and *B*).

## Exclude Outliers

Next, we are going to remove outliers, i.e., implausible RTs (RTs that likely do not reflect the processes of interest). In our case, we consider that RTs shorter than 0.2 seconds are too short for the cognitive process of interest to unfold, and that RTs longer than 3 seconds are too long to carry meaningful information.

```@example turing_advanced
# Remove outliers
df = df[(df.rt .> 0.2).&(df.rt .< 3), :]
first(df)
```

Note that standard outlier detection methods, such as *z*-scores (mean +- SD), are not necessarily appropriate for RTs, given the skewed nature of their distribution. Their asymmetric distribution is in fact accounted for by the models that we use. The outlier exclusion done here is more theory-driven (i.e., excluding extreme trials that likely do not reflect well the cognitive processes of interest) than data-driven (to better fit the model). That said, outlier exclusion should always be explicitly documented and justified.

!!! note
    For users coming from other languages, note the usage of the [vectorization dot](https://julialang.org/blog/2017/01/moredots/) `.` in front of the `<` and `>` symbols. This means that we want to apply the logical test for all individual elements of the `rt` vector.

## Visualize Data

We can visualize the RT distribution for each response choice by looping through the conditions.

```@example turing_advanced
# Make histogram
histogram(layout=(2, 1), xlabel="Reaction Time", ylims=(0, 60), xlims=(0, 2), legend=false)
for (i, cond) in enumerate(["A", "B"])
    histogram!(df.rt[(df.choice.==1).&(df.condition.==cond)], subplot=1, color=[:blue, :red][i], alpha=0.5, bins=range(0, 3, length=25))
    histogram!(df.rt[(df.choice.==2).&(df.condition.==cond)], subplot=2, color=[:blue, :red][i], alpha=0.5, bins=range(0, 2, length=25))
end
plot!()
```

## Format Predictors

One additional step that we need to do here is to transform the dataframe into an input suited for modelling with Turing. For that, we will leverage the features of `StatsModels` to build an input matrix based on a formula.

```@example turing_advanced
# Format input data
f = @formula(rt ~ 1 + condition)
f = apply_schema(f, schema(f, df))

_, predictors = coefnames(f)
X = modelmatrix(f, df)
```

In this case, the model matrix is pretty simple: the key part is the second column that is simply a binary vector indicating whenever `condition == "B"`. However, using formulas is a good way of dealing with more complex model specifications.

## Specify Turing Model

In this model, the priors for the parameters that we want to vary between conditions are split, with one prior for their intercept (condition A) and another for the effect of condition B (relative to the intercept).

Because the *drift* parameters is a vector of length 2, the priors for both the intercept and condition effect drifts have themselves to be a vector of 2 distributions, which is done via `filldist(prior_distribution, 2)`.

Next, we need to specify these parameters as the result of a (linear) equation. Note that:
- We have added a keyword argument, `condition`, to let the user pass the condition data vector.
- Since we're computing parameters as the results of an equation, we need to use a `for` loop that loops through all the observations.
- Because the priors for the drift is a `filldist` (i.e., a vector of distributions), we need to broadcast the addition (`.+` instead of `+`).

```julia
@model function model_lba(data; min_rt=0.2, condition=nothing)
    # Priors for auxiliary parameters
    A ~ truncated(Normal(0.8, 0.4), 0.0, Inf)
    k ~ truncated(Normal(0.2, 0.2), 0.0, Inf)
    tau ~ Uniform(0.0, min_rt)

    # Priors for coefficients
    drift_intercept ~ filldist(Normal(0, 1), 2)
    drift_condition ~ filldist(Normal(0, 1), 2)

    for i in 1:length(data)
        drifts = drift_intercept .+ drift_condition * condition[i]
        data[i] ~ LBA(; τ=tau, A=A, k=k, ν=drifts)
    end
end
```

Importantly, although we have the data as a dataframe, we will need to convert to a tuple, as it is the shape that the `LBA()` distribution expects. However, since we're iterating on each observation, we need to come up with an indexable version of the data: a **vector of tuples**.

```julia
# Format the data to match the input type
data = [(choice=df.choice[i], rt=df.rt[i]) for i in 1:nrow(df)]
```

## Prior Predictive Check

### Sample from Priors

Before we fit the model, we want to inspect our priors to make sure that they are okay. To do that, we sample the model parameters from priors only. Note that `condition` is supplied as the 2nd column of the model matrix.

```julia
chain = sample(model_lba(data; min_rt=minimum(df.rt), condition=X[:, 2]), Prior(), 1000)
plot(chain; size=(800, 1200))
```

### Plot Prior Predictive Check

The next step is to generate predictions from this model (i.e., from the priors). For this, we need to pass a dataset with empty (`missing`) values. Since the `data` used above was a vector (of tuples) of length 1000, we will create a vector of `(missing)` of the same length.

We can then use the `predict()` method to generate predictions from this model. However, because the most of `SequentialSamplingModels` distributions return a tuple (choice and RT), the predicted output has the two types of variables mixed together. We can delineate the two by taking every 2nd values to get the predicted choice and RTs, respectively.

```julia
datamissing = [(missing) for i in 1:nrow(df)]

pred = predict(model_lba(datamissing; min_rt=minimum(df.rt), condition=X[:, 2]), chain)

priorpred_choice = Array(pred)[:, 1:2:end]
priorpred_rt = Array(pred)[:, 2:2:end]
```

These objects have arrays of size 10,000 x 1000 : with 10,000 draws for each of the 1000 observations.

We can plot the predicted distributions by looping through a number of draws (e.g., 100), and then plotting the density for each condition and each choice.

```julia
plot(layout=(2, 1), xlabel="Reaction Time", xlims = (0, 3), ylim=(0, 5), legend = false)
for i in 1:100
    choice = priorpred_choice[i, :]
    rt = priorpred_rt[i, :]
    for (j, cond) in enumerate([0, 1])
        U1 = kde(rt[(choice .== 1) .& (X[:, 2] .== cond)], boundary=(0, 5))
        plot!(U1.x, U1.density, subplot=1, color = [:red, :blue][j], alpha=0.1)
        U2 = kde(rt[(choice .== 2) .& (X[:, 2] .== cond)], boundary=(0, 5))
        plot!(U2.x, U2.density, subplot=2, color = [:red, :blue][j], alpha=0.1)
    end
end
plot!()
```

We can see that the bulk of the predicted RTs fall within 0 - 1.5 seconds, which is realistic, but that the same time it's all over the place, which means that the priors are not too informative.

## Parameters Estimation

```julia
chain = sample(model_lba(data; min_rt=minimum(df.rt), condition=X[:, 2]), NUTS(), 1000)

summarystats(chain)
```

```julia
plot(chain; size = (800,1200))
```

## Posterior Predictive Check

Next, we will run a posterior predictive check by first sampling from the posteriors. For that, we will re-use the code for the prior predictive check, including the `datamissing` empty data.


```julia
# Sample from posteriors
pred = predict(model_lba(datamissing; min_rt=minimum(df.rt), condition=X[:, 2]), chain)
pred_choice = Array(pred)[:, 1:2:end]
pred_rt = Array(pred)[:, 2:2:end]
```

Next, we will plot the predicted distributions on top of the observed distribution of data (the thick black lines).

```julia
# Observed density
plot(layout=(2, 1), xlabel="Reaction Time", xlims=(0, 2.5), legend=false)
for cond in ["A", "B"]
    d_A = kde(df.rt[(df.choice.==1).&(df.condition.==cond)], boundary=(0, 5))
    plot!(d_A.x, d_A.density, subplot=1, color=:black, linewidth=3)
    d_B = kde(df.rt[(df.choice.==2).&(df.condition.==cond)], boundary=(0, 5))
    plot!(d_B.x, d_B.density, subplot=2, color=:black, linewidth=3)
end

# Predicted densities
for i in 1:100
    choice = pred_choice[i, :]
    rt = pred_rt[i, :]
    for (j, cond) in enumerate([0, 1])
        U1 = kde(rt[(choice.==1).&(X[:, 2].==cond)], boundary=(0, 5))
        plot!(U1.x, U1.density, subplot=1, color=[:red, :blue][j], alpha=0.05)
        U2 = kde(rt[(choice.==2).&(X[:, 2].==cond)], boundary=(0, 5))
        plot!(U2.x, U2.density, subplot=2, color=[:red, :blue][j], alpha=0.05)
    end
end
plot!()
```