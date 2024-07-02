"""
    MLBA{T <: Real} <: AbstractMLBA

# Fields 

- `ν::Vector{T}`: a vector of drift rates, which is a function of β₀, λₚ, λₙ, γ
- `β₀::T`: baseline input for drift rate 
- `λₚ::T`: decay constant for attention weights of positive differences
- `λₙ::T`: decay constant for attention weights of negative differences  
- `γ::T`: risk aversion exponent for subjective values
- `σ::Vector{T}`: a vector of drift rate standard deviation
- `A::T`: max start point
- `k::T`: A + k = b, where b is the decision threshold
- `τ::T`: an encoding-response offset

# Constructors

    MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k, τ)
    
    MLBA(;
        n_alternatives = 3,
        ν = fill(0.0, n_alternatives),
        β₀ = 1.0,
        λₚ = 1.0,
        λₙ = 1.0,
        γ = 0.70,
        τ = 0.3,
        A = 0.8,
        k = 0.5,
        σ = fill(1.0, n_alternatives)
    )

# References 

Trueblood, J. S., Brown, S. D., & Heathcote, A. (2014). The multiattribute linear ballistic accumulator model of context effects in multialternative choice. Psychological Review, 121(2), 179.
"""
mutable struct MLBA{T <: Real} <: AbstractMLBA
    ν::Vector{T}
    β₀::T
    λₚ::T
    λₙ::T
    γ::T
    σ::Vector{T}
    A::T
    k::T
    τ::T
end

function MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k, τ)
    _, β₀, λₚ, λₙ, γ, _, A, k, τ = promote(ν[1], β₀, λₚ, λₙ, γ, σ[1], A, k, τ)
    ν = convert(Vector{typeof(k)}, ν)
    σ = convert(Vector{typeof(k)}, σ)
    return MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k, τ)
end

MLBA(;
    n_alternatives = 3,
    ν = fill(0.0, n_alternatives),
    β₀ = 1.0,
    λₚ = 1.0,
    λₙ = 1.0,
    γ = 0.70,
    τ = 0.3,
    A = 0.8,
    k = 0.5,
    σ = fill(1.0, n_alternatives)
) =
    MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k, τ)

function params(d::AbstractMLBA)
    return (d.ν, d.β₀, d.λₚ, d.λₙ, d.γ, d.σ, d.A, d.k, d.τ)
end

rand(d::AbstractMLBA, M::AbstractArray) = rand(Random.default_rng(), d, M)

function rand(rng::AbstractRNG, d::AbstractMLBA, M::AbstractArray)
    compute_drift_rates!(d, M)
    return rand(rng, d)
end

rand(d::AbstractMLBA, n_trials::Int, M::AbstractArray) =
    rand(Random.default_rng(), d, n_trials, M)

function rand(rng::AbstractRNG, d::AbstractMLBA, n_trials::Int, M::AbstractArray)
    compute_drift_rates!(d, M)
    return rand(rng, d, n_trials)
end

function compute_drift_rates!(dist::AbstractMLBA, M::AbstractArray)
    (; ν, β₀) = dist
    n_options = length(ν)
    ν .= β₀
    utilities = map(x -> compute_utility(dist, x), eachrow(M))
    for i ∈ 1:n_options
        for j ∈ 1:n_options
            i == j ? continue : nothing
            ν[i] += compare(dist, utilities[i], utilities[j])
        end
    end
    return nothing
end

function compute_weight(dist::AbstractMLBA, u1, u2)
    (; λₙ, λₚ) = dist
    λ = u1 ≥ u2 ? λₚ : λₙ
    return exp(-λ * abs(u1 - u2))
end

function compute_utility(dist::AbstractMLBA, v)
    (; γ) = dist
    θ = atan(v[2], v[1])
    # x and y intercepts for line passing through v with slope -1
    a = sum(v)
    u = fill(0.0, 2)
    u[1] = a / (tan(θ)^γ + 1)^(1 / γ)
    u[2] = a * (1 - (u[1] / a)^γ)^(1 / γ)
    #u[2] = (a * tan(θ)) / (1 + tan(θ)^γ)^(1/ γ)
    return u
end

function compare(dist::AbstractMLBA, u1, u2)
    v = 0.0
    for i ∈ 1:2
        v += compute_weight(dist, u1[i], u2[i]) * (u1[i] - u2[i])
    end
    return v
end

"""
    simulate(rng::AbstractRNG, model::AbstractMLBA, M::AbstractArray; n_steps = 100)

Returns a matrix containing evidence samples of the MLBA decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractMLBA`: a subtype of AbstractMLBA

# Keywords 

- `n_steps=100`: number of time steps at which evidence is recorded
"""
function simulate(rng::AbstractRNG, model::AbstractMLBA, M::AbstractArray; n_steps = 100)
    compute_drift_rates!(model, M)
    return simulate(rng, model; n_steps)
end

simulate(model::AbstractMLBA, M::AbstractArray; n_steps = 100) =
    simulate(Random.default_rng(), model, M; n_steps)
