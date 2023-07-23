# Hierarchical Models

```@setup turing_mixed
using Random

Random.seed!(6)
```

In this example, we will fit a model with random factors and estimate individual parameters. This tutorial will build on the previous ones, so make sure you have followed them first. Let's start by loading all the packages and setting a reproducible seed.

```@example turing_mixed
using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
using Distributions
using DataFrames
using StatsPlots
using StatsModels
using CSV
using Optim

Random.seed!(6)
```

## Generate Data

We will use the [LBA](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) distribution to simulate data for 10 participants in two conditions with 100 trials per condition (repeated measures design). The drift rates for condition A are sampled from normal distributions, and the drift rates for condition B are set by sampling a departure from (i.e., the difference with) condition A. In other words, each participant has different drift rates for condition A (the *intercept*, i.e., the baseline condition) and a different "effect" magnitude of condition B (the offset from condition A to condition B).

```@example turing_mixed
# Generate data with different drifts for two conditions A vs. B
df = DataFrame()
params = DataFrame()
for participant in 1:10
    # Intercept (condition A)
    drifts = [rand(Normal(1.5, 0.2)), rand(Normal(0.5, 0.1))]
    param = join(round.(drifts, digits=2), ", ")  # Format and save params
    df1 = DataFrame(rand(LBA(ν=drifts, A=0.5, k=0.5, τ=0.3), 100))
    df1[!, :condition] = repeat(["A"], nrow(df1))
    df1[!, :participant] = repeat([participant], nrow(df1))

    # Effect of condition B
    drifts2 = [rand(Normal(0.5, 0.15)), rand(Normal(0.5, 0.05))]
    param = [param, join(round.(drifts2, digits=2), ", ")]
    df2 = DataFrame(rand(LBA(ν=drifts .+ drifts2, A=0.5, k=0.5, τ=0.3), 100))
    df2[!, :condition] = repeat(["B"], nrow(df2))
    df2[!, :participant] = repeat([participant], nrow(df1))

    # Assemble and store parameters (to compare with estimation)
    df = vcat(df, df1, df2)
    params = vcat(params, DataFrame(permutedims(param), [:drift_intercept, :drift_condition]))
end
```

We can visualize the individual distributions for the two type of responses and for the conditions (condition A in red and B in blue).

```@example turing_mixed
density(layout=(2, 1), ylims=(0, 5), xlims=(0, 3), legend=false)
for p in unique(df.participant)
    for (i, cond) in enumerate(["A", "B"])
        density!(df.rt[(df.choice.==1).&(df.condition.==cond).&(df.participant.==p)],
            subplot=1, color=[:blue, :red][i], title="Choice = 1")
        density!(df.rt[(df.choice.==2).&(df.condition.==cond).&(df.participant.==p)],
            subplot=2, color=[:blue, :red][i], title="Choice = 2", xlabel="Reaction Time (s)")
    end
end
plot!()
```

## Model Specification

First, we will transform our predictor data into an model matrix. This essentially transform our favor column with "A" and "B" to a binary vector.

We will also transform our outcome data (RTs and choice) into a list of tuples (see [this example for more explanation](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/turing_advanced/)).

```@example turing_mixed
# Format input data
f = @formula(rt ~ 1 + condition)
f = apply_schema(f, schema(f, df))
_, predictors = coefnames(f)
X = modelmatrix(f, df)

# Format the data to match the input type
data = [(choice=df.choice[i], rt=df.rt[i]) for i in 1:nrow(df)]
```

Now, the model is a bit more complex:


```@example turing_mixed
@model function model_lba(data; min_rt=0.2, condition=nothing, participant=nothing)

    # Priors for auxiliary parameters
    A ~ truncated(Normal(0.8, 0.4), 0.0, Inf)
    k ~ truncated(Normal(0.2, 0.2), 0.0, Inf)
    tau ~ Uniform(0.0, min_rt)

    # Priors for population-level coefficients
    drift_intercept_1 ~ Normal(0, 1)
    drift_intercept_2 ~ Normal(0, 1)
    drift_condition_1 ~ Normal(0, 1)
    drift_condition_2 ~ Normal(0, 1)

    # Prior for random intercepts (requires thoughtful specification)
    # Group-level intercepts' SD
    drift_intercept_random_sd ~ truncated(Cauchy(0, 0.1), 0.0, Inf)
    # Group-level intercepts
    drift_intercept_random_1 ~ filldist(
        Normal(0, drift_intercept_random_sd),
        length(unique(participant))
    )
    drift_intercept_random_2 ~ filldist(
        Normal(0, drift_intercept_random_sd),
        length(unique(participant))
    )

    for i in 1:length(data)
        # Formula for intercept
        drifts_intercept_1 = drift_intercept_1 .+ drift_intercept_random_1[participant[i]]
        drifts_intercept_2 = drift_intercept_2 .+ drift_intercept_random_2[participant[i]]

        # Combine with condition
        drifts_1 = drift_intercept_1 + drift_condition_1 * condition[i]
        drifts_2 = drift_intercept_2 + drift_condition_2 * condition[i]
        data[i] ~ LBA(; τ=tau, A=A, k=k, ν=[drifts_1, drifts_2])
    end
end
```


Note that for now, these types of model are *very* slow to run in Turing.