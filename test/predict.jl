using LinearAlgebra
using SequentialSamplingModels
using Test
using Turing
using Random

Random.seed!(2001)

@model function model(data; min_rt = minimum(data[2]))
    ν ~ MvNormal(zeros(2), I * 2)
    A ~ truncated(Normal(0.8, 0.4), 0.0, Inf)
    k ~ truncated(Normal(0.2, 0.2), 0.0, Inf)
    τ ~ Uniform(0.0, min_rt)
    data ~ LBA(; ν, A, k, τ)
end

# generate some data
dist = LBA(ν = [3.0, 2.0], A = 0.8, k = 0.2, τ = 0.3)
data = rand(dist, 100)

# estimate parameters
chain = sample(model(data), NUTS(200, 0.65), 100)
predictions = predict(model(missing; min_rt = minimum(data[2])), chain)
@test isa(predictions, Chains)
