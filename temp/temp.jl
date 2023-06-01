cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using Revise, Plots
using SequentialSamplingModels
using SequentialSamplingModels: simulate, simulate_trial

model = LCA(; α = 1.5, β=0.20, λ=0.10, ν=[2.5,2.0], Δt=.001, τ=.30, σ=1.0)

choice,rt = rand(model, 100_000)
mean(choice .== 1)
mean(rt[choice .== 1])
mean(rt[choice .== 2])
std(rt[choice .== 1])
std(rt[choice .== 2])





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