cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using Revise, Plots
using SequentialSamplingModels
using SequentialSamplingModels: simulate, simulate_trial

model = LCA(; α = .5, β=0.2, λ=0.1, ν=[2.5,3.0], Δt=.005)

rts = simulate(model, 100_000)
mean.(rts)





using SequentialSamplingModels: increment!

model = LCA(; α = .3, β=0.3, λ=0.1, ν=[.8,.5])
n = length(model.ν)
x = fill(0.0, n)
μΔ = fill(0.0, n)
ϵ = fill(0.0, n)
t = 0.0
Δt = .005
evidence = Vector{Vector{Float64}}()
while all(x .< model.α)
    t += Δt
    increment!(model, x, μΔ, ϵ)
    push!(evidence, copy(x))
end  

plot(hcat(evidence...)')