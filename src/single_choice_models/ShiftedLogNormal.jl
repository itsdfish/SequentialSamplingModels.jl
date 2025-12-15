"""
    ShiftedLogNormal{T <: Real} <: AbstractShiftedLogNormal

A special case of the lognormal race (LNR) model for a single response. The first passage time is lognormally distributed.

# Parameters 

- `ν::T`: mean finishing time in log-space. ν ∈ ℝ. 
- `σ::T`: standard deviation parameter in log-space. σ ∈ ℝ⁺.
- `τ::T`: a encoding-response offset. τ ∈ [0, min_rt].

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    ShiftedLogNormal(ν, σ, τ)
    
The second constructor uses keywords with default values, and is not order dependent: 

    ShiftedLogNormal(; ν = -1, σ=.5, τ = .20)

# Example

```julia
using SequentialSamplingModels
dist = ShiftedLogNormal(ν = -1, σ=.5, τ = .20)
rts = rand(dist, 10)
like = pdf.(dist, rts)
loglike = logpdf.(dist, rts)
```
# References

Heathcote, A., & Bohlscheid, E. Analysis and Modeling of Response Time using the Shifted Lognormal Distribution.
"""
struct ShiftedLogNormal{T <: Real} <: AbstractShiftedLogNormal
    ν::T
    σ::T
    τ::T
    function ShiftedLogNormal(ν::T, σ::T, τ::T) where {T <: Real}
        @argcheck σ ≥ 0
        @argcheck τ ≥ 0
        return new{T}(ν, σ, τ)
    end
end

ShiftedLogNormal(ν, σ, τ) = ShiftedLogNormal(promote(ν, σ, τ)...)

ShiftedLogNormal(; ν = -1, σ = 0.5, τ = 0.20) = ShiftedLogNormal(ν, σ, τ)

function logpdf(dist::AbstractShiftedLogNormal, rt)
    (; τ, ν, σ) = dist
    @argcheck τ ≤ rt
    return logpdf(LogNormal(ν, σ), rt - τ)
end

function pdf(dist::AbstractShiftedLogNormal, rt::Real)
    (; τ, ν, σ) = dist
    @argcheck τ ≤ rt
    return pdf(LogNormal(ν, σ), rt - τ)
end

function cdf(dist::AbstractShiftedLogNormal, rt::Real)
    (; τ, ν, σ) = dist
    @argcheck τ ≤ rt
    return cdf(LogNormal(ν, σ), rt - τ)
end

function rand(rng::AbstractRNG, dist::AbstractShiftedLogNormal, n_trials::Int)
    (; τ, ν, σ) = dist
    return rand(rng, LogNormal(ν, σ), n_trials) .+ τ
end

function rand(rng::AbstractRNG, dist::AbstractShiftedLogNormal)
    (; τ, ν, σ) = dist
    return rand(rng, LogNormal(ν, σ)) + τ
end

function params(d::AbstractShiftedLogNormal)
    return (d.ν, d.σ, d.τ)
end

function mean(dist::AbstractShiftedLogNormal)
    return mean(LogNormal(dist.ν, dist.σ)) + dist.τ
end

function std(dist::AbstractShiftedLogNormal)
    return std(LogNormal(dist.ν, dist.σ))
end
