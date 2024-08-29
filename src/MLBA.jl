"""
    MLBA{T <: Real, T1 <: Union{<: T, Vector{<: T}}} <: AbstractMLBA{T,T1}

# Fields 

- `ν::Vector{T}`: a vector of drift rates, which is a function of β₀, λₚ, λₙ, γ
- `β₀::T`: baseline input for drift rate 
- `λₚ::T`: decay constant for attention weights of positive differences
- `λₙ::T`: decay constant for attention weights of negative differences  
- `γ::T`: risk aversion exponent for subjective values
- `σ::T1`: a scalar or vector of drift rate standard deviation
- `A::T`: max start point
- `k::T`: A + k = b, where b is the decision threshold
- `τ::T`: an encoding-response offset

# Constructors

    MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k, τ)
    
    MLBA(;
        n_alternatives = 3,
        ν = fill(0.0, n_alternatives),
        β₀ = 5.0,
        λₚ = 0.20,
        λₙ = 0.40,
        γ = 5.0,
        τ = 0.3,
        A = 1.0,
        k = 1.0,
        σ = 1.0
    )

# Example 

```julia
using SequentialSamplingModels

dist = MLBA(
    λₚ = 0.20,
    λₙ = 0.40,
    β₀ = 5,
    γ = 5,
    τ = 0.3,
    A = 0.8,
    k = 0.5
)

M = [
    1 4
    2 2
    4 1
]

choice, rt = rand(dist, 1000, M)
like = pdf.(dist, choice, rt, (M,))
loglike = logpdf.(dist, choice, rt, (M,))
```
# References 

Trueblood, J. S., Brown, S. D., & Heathcote, A. (2014). The multiattribute linear ballistic accumulator model of context effects in multialternative choice. Psychological Review, 121(2), 179.
"""
mutable struct MLBA{T <: Real, T1 <: Union{<:T, Vector{<:T}}} <: AbstractMLBA{T, T1}
    ν::Vector{T}
    β₀::T
    λₚ::T
    λₙ::T
    γ::T
    σ::T1
    A::T
    k::T
    τ::T
end

function MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k::T, τ) where {T}
    _, β₀, λₚ, λₙ, γ, _, A, k, τ = promote(ν[1], β₀, λₚ, λₙ, γ, σ[1], A, k, τ)
    ν = convert(Vector{T}, ν)
    σ = isa(σ, Vector) ? convert(Vector{T}, σ) : convert(T, σ)
    return MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k, τ)
end

MLBA(;
    n_alternatives = 3,
    ν = fill(0.0, n_alternatives),
    β₀ = 5.0,
    λₚ = 0.20,
    λₙ = 0.40,
    γ = 5.0,
    τ = 0.3,
    A = 1.0,
    k = 1.0,
    σ = 1.0
) =
    MLBA(ν, β₀, λₚ, λₙ, γ, σ, A, k, τ)

function params(d::AbstractMLBA)
    return (d.ν, d.β₀, d.λₚ, d.λₙ, d.γ, d.σ, d.A, d.k, d.τ)
end

rand(d::AbstractMLBA, M::AbstractArray) = rand(Random.default_rng(), d, M)

"""
    rand(rng::AbstractRNG, d::AbstractMLBA, M::AbstractArray)

Generates a single choice-rt pair of simulated data from the Multi-attribute Linear Ballistic Accumulator.

# Arguments

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 
"""
function rand(rng::AbstractRNG, d::AbstractMLBA, M::AbstractArray)
    compute_drift_rates!(d, M)
    return rand(rng, d)
end

rand(d::AbstractMLBA, n_trials::Int, M::AbstractArray) =
    rand(Random.default_rng(), d, n_trials, M)

"""
    rand(rng::AbstractRNG, d::AbstractMLBA, n_trials::Int, M::AbstractArray)

Generates `n_trials` choice-rt pair of simulated data from the Multi-attribute Linear Ballistic Accumulator.

# Arguments

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `n_trials::Int`: the number of trials to simulate
- `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 
"""
function rand(rng::AbstractRNG, d::AbstractMLBA, n_trials::Int, M::AbstractArray)
    compute_drift_rates!(d, M)
    return rand(rng, d, n_trials)
end

"""
    compute_drift_rates!(dist::AbstractMLBA, M::AbstractArray)


# Arguments

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 
"""
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

"""
    compute_weight(dist::AbstractMLBA, u1, u2)

Computes attention weight between two options. The weight is an inverse function of similarity between the 
options. Similarity between two options decays exponentially as a function of distance and is asymmetrical 
when decay rates λₚ ≠ λₙ. 

# Arguments

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `u1`: utility of the first option 
- `u1`: utility of the second option 
"""
function compute_weight(dist::AbstractMLBA, u1, u2)
    (; λₙ, λₚ) = dist
    λ = u1 ≥ u2 ? λₚ : λₙ
    return exp(-λ * abs(u1 - u2))
end

"""
    compute_utility(dist::AbstractMLBA, v)

Computes the utility of an alternative.  

# Arguments

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `v`: a vector representing the attributes of a given n_alternative
"""
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

"""
    compare(dist::AbstractMLBA, u1, u2)

Computes a weighted difference between two options, where the weight is an inverse function of similarity
between options.   

# Arguments

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `u1`: utility of the first option 
- `u1`: utility of the second option 
"""
function compare(dist::AbstractMLBA, u1, u2)
    v = 0.0
    for i ∈ 1:2
        v += compute_weight(dist, u1[i], u2[i]) * (u1[i] - u2[i])
    end
    return v
end

"""
    pdf(d::AbstractMLBA, c::Int, rt::Real,  M::AbstractArray)

Computes default probability density for multi-alternative linear ballistic accumulator. 

# Arguments 

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `c::Int`: choice index
- `rt::Real`: reaction time in seconds 
- `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 
"""
function pdf(d::AbstractMLBA, c::Int, rt::Real, M::AbstractArray)
    compute_drift_rates!(d, M)
    return pdf(d, c, rt)
end

"""
    logpdf(d::AbstractMLBA, c::Int, rt::Real,  M::AbstractArray)

Computes default log probability density for multi-alternative linear ballistic accumulator. 

# Arguments 

- `dist::AbstractMLBA`: an object for the multi-attribute linear ballistic accumulator
- `c::Int`: choice index
- `rt::Real`: reaction time in seconds 
- `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 
"""
function logpdf(d::AbstractMLBA, c::Int, rt::Real, M::AbstractArray)
    compute_drift_rates!(d, M)
    return logpdf(d, c, rt)
end

loglikelihood(d::AbstractMLBA, data::NamedTuple, M::AbstractArray) =
    sum(logpdf.(d, data..., (M,)))

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
