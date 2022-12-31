cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using Revise, Plots
using SequentialSamplingModels
using SequentialSamplingModels: simulate, simulate_trial

model = OU(; α = .5, β=0.2, λ=0.0, ν=[2.5,3.0], Δt=.005)

rts = simulate(model, 100_000)
mean.(rts)

using SequentialSamplingModels: compute_mean_evidence!, add_noise!
n = length(model.ν)
x = fill(0.0, n)
y = fill(0.0, n)
μ = fill(0.0, n)
ϵ = fill(0.0, n)
(;β,λ,ν,σ,Δt) = model
simulate_trial(model, x, μ, ϵ)
compute_mean_evidence!(ν, β, λ, x, μ)
add_noise!(ν, σ, Δt, x, μ, ϵ)




using SequentialSamplingModels: increment!

model = OU(; α = .3, β=0.3, λ=0.1, ν=[.8,.5])
x = fill(0.0, n)
μ = fill(0.0, n)
ϵ = fill(0.0, n)
t = 0.0
evidence = Vector{Vector{Float64}}()
while all(x .< model.α)
    t += Δt
    increment!(model, x, μ, ϵ)
    push!(evidence, copy(x))
end  

plot(hcat(evidence...)')



# function sim(;v = 1.7, α=1.5, σ=1.0, Δt=.0005, ter=.00)
#     x = α / 2
#     t = ter
#     while (x < α) && (x > 0)
#         t += Δt
#         x += (v * Δt + randn() * σ * sqrt(Δt))
#     end
#     c = x > α ? 1 : 2
#     return c, t
# end

using Random, ThreadTools

Random.seed!(8554)

x = tmap(_ -> rand(), 1:1000)