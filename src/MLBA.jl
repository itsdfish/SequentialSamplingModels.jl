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
