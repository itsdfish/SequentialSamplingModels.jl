cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using Revise
using SequentialSamplingModels
using SequentialSamplingModels: simulate

model = OU(; α = .3, β=0.0, λ=0.0, ν=[.8,.7])

rts = simulate(model, 100)
mean.(rts)

dist = DiffusionRace(;ν= [2,.1], k = .8, A = 1.0, θ = .2)
rts = rand(dist, 1000)
rts1 = filter(x -> x[1] == 1, rts)
rts1 = map(x -> x[2], rts1)
rts2 = filter(x -> x[1] == 2, rts)
rts2 = map(x -> x[2], rts2)