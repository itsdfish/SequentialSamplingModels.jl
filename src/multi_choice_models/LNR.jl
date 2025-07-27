"""
    LNR{T <: Real, T1 <: Union{<:T, Vector{<:T}}} <: AbstractLNR{T, T1}

# Parameters 

- `ν::Vector{T}`: a vector of means in log-space. ν ∈ ℝⁿ.
- `σ::T1`: a scalar or vector of standard deviation parameter in log-space. σ ∈ ℝ⁺.
- `τ::T`: a encoding-response offset. τ ∈ [0, min_rt].

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    LNR(ν, σ, τ)
    
The second constructor uses keywords with default values, and is not order dependent: 

    LNR(; ν = [-1, -2], σ = 1, τ = 0.20)

# Example

```julia
using SequentialSamplingModels
dist = LNR(ν = [-2,-3], σ = 1, τ = .3)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```
# References

Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). 
The lognormal race: A cognitive-process model of choice and latency with desirable 
psychometric properties. Psychometrika, 80(2), 491-513.
"""
struct LNR{T <: Real, T1 <: Union{<:T, Vector{<:T}}} <: AbstractLNR{T, T1}
    ν::Vector{T}
    σ::T1
    τ::T
    function LNR(ν::Vector{T}, σ::T1, τ::T) where {T <: Real, T1 <: Union{<:T, Vector{<:T}}}
        @argcheck all(σ .≥ 0)
        @argcheck τ ≥ 0
        return new{T,T1}(ν, σ, τ)
    end
end

function LNR(ν, σ, τ::T) where {T}
    _, _, τ = promote(ν[1], σ[1], τ)
    ν = convert(Vector{T}, ν)
    σ = isa(σ, Vector) ? convert(Vector{T}, σ) : convert(T, σ)
    return LNR(ν, σ, τ)
end

function params(d::AbstractLNR)
    return (d.ν, d.σ, d.τ)
end

LNR(; ν = [-1, -2], σ = 1, τ = 0.20) = LNR(ν, σ, τ)

function rand(rng::AbstractRNG, dist::AbstractLNR)
    (; ν, σ, τ) = dist
    x = @. rand(rng, LogNormal(ν, σ)) + τ
    rt, choice = findmin(x)
    return (; choice, rt)
end

function logpdf(d::AbstractLNR{T, T1}, r::Int, t::Float64) where {T, T1 <: Vector{<:Real}}
    (; ν, σ, τ) = d
    LL = 0.0
    for i ∈ 1:length(ν)
        if i == r
            LL += logpdf(LogNormal(ν[i], σ[i]), t - τ)
        else
            LL += logccdf(LogNormal(ν[i], σ[i]), t - τ)
        end
    end
    return LL
end

function logpdf(d::AbstractLNR{T, T1}, r::Int, t::Float64) where {T, T1 <: Real}
    (; ν, σ, τ) = d
    LL = 0.0
    for i ∈ 1:length(ν)
        if i == r
            LL += logpdf(LogNormal(ν[i], σ), t - τ)
        else
            LL += logccdf(LogNormal(ν[i], σ), t - τ)
        end
    end
    return LL
end

function pdf(d::AbstractLNR{T, T1}, r::Int, t::Float64) where {T, T1 <: Vector{<:Real}}
    (; ν, σ, τ) = d
    density = 1.0
    for i ∈ 1:length(ν)
        if i == r
            density *= pdf(LogNormal(ν[i], σ[i]), t - τ)
        else
            density *= (1 - cdf(LogNormal(ν[i], σ[i]), t - τ))
        end
    end
    return density
end

function pdf(d::AbstractLNR{T, T1}, r::Int, t::Float64) where {T, T1 <: Real}
    (; ν, σ, τ) = d
    density = 1.0
    for i ∈ 1:length(ν)
        if i == r
            density *= pdf(LogNormal(ν[i], σ), t - τ)
        else
            density *= (1 - cdf(LogNormal(ν[i], σ), t - τ))
        end
    end
    return density
end

"""
    simulate(model::AbstractLNR; n_steps=100, _...)

Returns a matrix containing evidence samples of the LBA decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractLNR`: a subtype of AbstractLNR

# Keywords 

- `Δt = 0.001`: the time step
"""
function simulate(rng::AbstractRNG, model::AbstractLNR; Δt = 0.001, _...)
    (; ν, σ) = model
    νs = @. rand(rng, Normal(ν, σ))
    βs = @. exp(νs)
    _, choice = findmax(βs)
    t = 1 / βs[choice]
    time_steps = range(0, t, step = Δt)
    evidence = collect.(range.(0, βs * t, length = length(time_steps)))
    return time_steps, hcat(evidence...)
end
