# https://web.archive.org/web/20210715113857id_/https://pure.uva.nl/ws/files/54450607/1_s2.0_S0010028520300219_main.pdf
# Double responding: A new constraint for models of speeded decision making
@concrete mutable struct LCA <: SequentialSamplingModel
    ν
    α
    β
    λ
    τ
    σ
    Δt
end

function LCA(;
    ν = [.1,.15], 
    α = .1, 
    β = .1, 
    λ = .1, 
    τ = .3, 
    σ = .1, 
    Δt = .001
    )
    return LCA(ν, α, β, λ, τ, σ, Δt)
end

function rand(dist::LCA)
    # number of trials 
    n = length(dist.ν)
    # evidence for each alternative
    x = fill(0.0, n)
    # mean change in evidence for each alternative
    Δμ = fill(0.0, n)
    # noise for each alternative 
    ϵ = fill(0.0, n)
    return simulate_trial(dist, x, Δμ, ϵ)
end

function rand(dist::LCA, n_sim::Int)
    n = length(dist.ν)
    x = fill(0.0, n)
    Δμ = fill(0.0, n)
    ϵ = fill(0.0, n)
    choices = fill(0, n_sim)
    rts = fill(0.0, n_sim)
    for i in 1:n_sim
        choices[i],rts[i] = simulate_trial(dist, x, Δμ, ϵ)
        x .= 0.0
    end
    return choices,rts 
end

function simulate_trial(dist, x, Δμ, ϵ)
    (;Δt, α, τ) = dist
    t = 0.0
    while all(x .< α)
        increment!(dist, x, Δμ, ϵ)
        t += Δt
    end    
    _,choice = findmax(x) 
    rt = t + τ
    return choice,rt
end

function increment!(ν, β, λ, σ, Δt, x, Δμ, ϵ)
    n = length(ν)
    # compute change of mean evidence: νᵢ - λxᵢ - βΣⱼxⱼ
    compute_mean_evidence!(ν, β, λ, x, Δμ)
    # sample noise 
    ϵ .= rand(Normal(0, σ), n)
    # add mean change in evidence plus noise 
    x .+= Δμ * Δt .+ ϵ * √(Δt)
    # ensure that evidence is non-negative 
    x .= max.(x, 0.0)
    return nothing
end

function increment!(dist, x, Δμ, ϵ)
    (;ν, β, λ, σ, Δt) = dist
    return increment!(ν, β, λ, σ, Δt, x, Δμ, ϵ)
end

function compute_mean_evidence!(ν, β, λ, x, Δμ)
    for i in 1:length(ν)
        Δμ[i] = ν[i] - λ * x[i] - β * inhibit(x, i)
    end
    return nothing
end

function inhibit(x, i)
    v = 0.0
    for j in 1:length(x)
        v += j ≠ i ? x[j] : 0.0
    end
    return v
end

function make_kdes(sim_data)
    ns = length.(sim_data)
    probs = ns ./ sum(ns)
    kdes = kernel.(sim_data)
    dists = InterpKDE.(kdes)
    return dists, probs
end

