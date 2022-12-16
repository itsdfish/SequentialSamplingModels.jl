# https://web.archive.org/web/20210715113857id_/https://pure.uva.nl/ws/files/54450607/1_s2.0_S0010028520300219_main.pdf
# Double responding: A new constraint for models of speeded decision making
@concrete mutable struct OU <: SequentialSamplingModel
    ν
    α
    β
    λ
    τ
    σ
    Δt
end

function OU(;
    ν = [.1,.15], 
    α = .1, 
    β = .1, 
    λ = .1, 
    τ = .3, 
    σ = .1, 
    Δt = .001
    )
    return OU(ν, α, β, λ, τ, σ, Δt)
end

function inhabition(x, i)
    v = 0.0
    for j in 1:length(x)
        v += j ≠ i ? x[j] : 0.0
    end
    return v
end

function compute_mean_evidence!(ν, β, λ, x, y, μ)
    n = length(ν)
    for i in 1:n
        y[i] = inhabition(x, i)
        μ[i] = ν[i] - λ * x[i] - β * y[i]
    end
    return nothing
end

function add_noise!(ν, σ, Δt, x, μ, ϵ)
    n = length(ν)
    ϵ .= rand(Normal(0, σ), n)
    x .+= μ * Δt .+ ϵ * √(Δt)
    x .= max.(x, 0.0)
    return nothing
end

function increment!(ν, β, λ, σ, Δt, x, y, μ, ϵ)
    compute_mean_evidence!(ν, β, λ, x, y, μ)
    add_noise!(ν, σ, Δt, x, μ, ϵ)
    return nothing
end

function increment!(dist, x, y, μ, ϵ)
    (;ν, β, λ, σ, Δt) = dist
    return increment!(ν, β, λ, σ, Δt, x, y, μ, ϵ)
end

function simulate_trial(dist)
    n = length(dist.ν)
    x = fill(0.0, n)
    y = fill(0.0, n)
    μ = fill(0.0, n)
    ϵ = fill(0.0, n)
    return simulate_trial(dist, x, y, μ, ϵ)
end

function simulate_trial(dist, x, y, μ, ϵ)
    (;Δt, α, τ) = dist
    t = 0.0
    while all(x .< α)
        t += Δt
        increment!(dist, x, y, μ, ϵ)
    end    
    _,choice = findmax(x) 
    rt = t + τ
    return choice,rt
end

function simulate(dist, n_sim=10_000)
    n = length(dist.ν)
    x = fill(0.0, n)
    y = fill(0.0, n)
    μ = fill(0.0, n)
    ϵ = fill(0.0, n)
    rts = [Vector{Float64}() for _ in 1:n]
    for i in 1:n_sim
        choice,rt = simulate_trial(dist, x, y, μ, ϵ)
        push!(rts[choice], rt)
        x .= 0.0
    end
    return rts 
end

function make_kdes(sim_data)
    ns = length.(sim_data)
    probs = ns ./ sum(ns)
    kdes = kernel.(sim_data)
    dists = InterpKDE.(kdes)
    return dists, probs
end