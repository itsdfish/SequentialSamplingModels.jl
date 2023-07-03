using Turing
using SequentialSamplingModels
using Random
using LinearAlgebra
using Test

@model function model(data; min_rt = minimum(data[2]))
    ν ~ MvNormal(zeros(2), I * 2)
    A ~ truncated(Normal(.8, .4), 0.0, Inf)
    k ~ truncated(Normal(.2, .2), 0.0, Inf)
    τ  ~ Uniform(0.0, min_rt)
    data ~ LBA(;ν, A, k, τ )
end

# generate some data
Random.seed!(254)
dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
data = rand(dist, 100)

# estimate parameters
chain = sample(model(data), NUTS(200, .65), 100)
predictions = predict(model(missing; min_rt = minimum(data[2])), chain)
@test true