"""
    LCA{T<:Real} <: AbstractLCA

A model type for the Leaky Competing Accumulator. 
    
# Parameters 

- `ν`: drift rates 
- `σ`: diffusion noise 
- `β`: lateral inhabition 
- `λ`: leak rate
- `α`: evidence threshold 
- `τ`: non-decision time 

# Constructors 

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    LCA(ν, σ, β, λ, α, τ)

The second constructor uses keywords with default values, and is not order dependent: 

    LCA(; ν = [2.5, 2.0], α = 1.5, β = 0.20, λ = 0.10, τ = 0.30, σ = 1.0)
        
# Example 

```julia 
using SequentialSamplingModels 
ν = [2.5,2.0]
α = 1.5
β = 0.20
λ = 0.10 
σ = 1.0
τ = 0.30

dist = LCA(; ν, α, β, λ, τ, σ, Δt)
choices,rts = rand(dist, 500)
```
# References

Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. Psychological Review, 108 3, 550–592. https://doi.org/10.1037/0033-295X.108.3.550
"""
mutable struct LCA{T<:Real} <: AbstractLCA
    ν::Vector{T}
    σ::T
    β::T
    λ::T
    α::T
    τ::T
end

function LCA(ν, σ, β, λ, α, τ)
    _, σ, β, λ, α, τ = promote(ν[1], σ, β, λ, α, τ)
    ν = convert(Vector{typeof(τ)}, ν)
    return LCA(ν, σ, β, λ, α, τ)
end

function LCA(; ν = [2.5, 2.0], α = 1.5, β = 0.20, λ = 0.10, τ = 0.30, σ = 1.0)
    return LCA(ν, σ, β, λ, α, τ)
end

function params(d::AbstractLCA)
    return (d.ν, d.σ, d.β, d.λ, d.α, d.τ)
end

get_pdf_type(d::AbstractLCA) = Approximate

"""
    rand(dist::AbstractLCA; Δt = 0.001)

Generate a random choice-rt pair for the Leaky Competing Accumulator.

# Arguments
- `dist`: model object for the Leaky Competing Accumulator. 
- `Δt = 0.001`: time step size 
"""
function rand(rng::AbstractRNG, dist::AbstractLCA; Δt = 0.001)
    # number of trials 
    n = length(dist.ν)
    # evidence for each alternative
    x = fill(0.0, n)
    # mean change in evidence for each alternative
    Δμ = fill(0.0, n)
    # noise for each alternative 
    ϵ = fill(0.0, n)
    return simulate_trial(rng, dist, x, Δμ, ϵ; Δt)
end

"""
    rand(dist::AbstractLCA, n_sim::Int; Δt = 0.001)

Generate `n_sim` random choice-rt pairs for the Leaky Competing Accumulator.

# Arguments

- `dist`: model object for the Leaky Competing Accumulator.
- `n_sim::Int`: the number of simulated choice-rt pairs 

# Keywords
- `Δt = 0.001`: time step size
"""
function rand(rng::AbstractRNG, dist::AbstractLCA, n_sim::Int; Δt = 0.001)
    n = length(dist.ν)
    x = fill(0.0, n)
    Δμ = fill(0.0, n)
    ϵ = fill(0.0, n)
    choices = fill(0, n_sim)
    rts = fill(0.0, n_sim)
    for i = 1:n_sim
        choices[i], rts[i] = simulate_trial(rng, dist, x, Δμ, ϵ; Δt)
        x .= 0.0
    end
    return (; choices, rts)
end

function simulate_trial(rng::AbstractRNG, dist, x, Δμ, ϵ; Δt = 0.001)
    (; α, τ) = dist
    t = 0.0
    while all(x .< α)
        increment!(rng, dist, x, Δμ, ϵ; Δt)
        t += Δt
    end
    _, choice = findmax(x)
    rt = t + τ
    return (; choice, rt)
end

increment!(ν, β, λ, σ, Δt, x, Δμ, ϵ) =
    increment!(Random.default_rng(), ν, β, λ, σ, Δt, x, Δμ, ϵ)

function increment!(rng::AbstractRNG, ν, β, λ, σ, Δt, x, Δμ, ϵ)
    n = length(ν)
    # compute change of mean evidence: νᵢ - λxᵢ - βΣⱼxⱼ
    compute_mean_evidence!(ν, β, λ, x, Δμ)
    # sample noise 
    ϵ .= rand(rng, Normal(0, σ), n)
    # add mean change in evidence plus noise 
    x .+= Δμ * Δt .+ ϵ * √(Δt)
    # ensure that evidence is non-negative 
    x .= max.(x, 0.0)
    return nothing
end

increment!(dist, x, Δμ, ϵ; Δt = 0.001) = increment!(Random.default_rng(), dist, x, Δμ, ϵ; Δt)

function increment!(rng::AbstractRNG, dist, x, Δμ, ϵ; Δt = 0.001)
    (; ν, β, λ, σ) = dist
    return increment!(rng, ν, β, λ, σ, Δt, x, Δμ, ϵ)
end

function compute_mean_evidence!(ν, β, λ, x, Δμ)
    for i = 1:length(ν)
        Δμ[i] = ν[i] - λ * x[i] - β * inhibit(x, i)
    end
    return nothing
end

function inhibit(x, i)
    v = 0.0
    for j = 1:length(x)
        v += j ≠ i ? x[j] : 0.0
    end
    return v
end

"""
    simulate(model::AbstractLCA; _...)

Returns a matrix containing evidence samples of the LCA decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstrctLCA`: an LCA model object
"""
function simulate(model::AbstractLCA; Δt = 0.001, _...)
    (; α) = model
    n = length(model.ν)
    x = fill(0.0, n)
    μΔ = fill(0.0, n)
    ϵ = fill(0.0, n)
    t = 0.0
    evidence = [fill(0.0, n)]
    time_steps = [t]
    while all(x .< α)
        t += Δt
        increment!(model, x, μΔ, ϵ; Δt)
        push!(evidence, copy(x))
        push!(time_steps, t)
    end
    return time_steps, reduce(vcat, transpose.(evidence))
end
