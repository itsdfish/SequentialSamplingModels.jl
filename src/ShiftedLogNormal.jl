"""
    ShiftedLogNormal{T <: Real} <: AbstractShiftedLogNormal

A special case of the lognormal race (LNR) model for a single response. The first passage time is lognormally distributed.

# Parameters 

- `ν`: mean finishing time in log-space
- `σ`: standard deviation parameter in log-space
- `τ`: a encoding-response offset

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
end

ShiftedLogNormal(ν, σ, τ) = ShiftedLogNormal(promote(ν, σ, τ)...)

ShiftedLogNormal(; ν = -1, σ=.5, τ = .20) = ShiftedLogNormal(ν, σ, τ)

function logpdf(dist::AbstractShiftedLogNormal, rt)
    (; τ, ν, σ) = dist 
    return logpdf(LogNormal(ν, σ), rt - τ)
end

function rand(dist::AbstractShiftedLogNormal, n_trials::Int)
    (; τ, ν, σ) = dist 
    return rand(LogNormal(ν, σ), n_trials) .+ τ
end

model = ShiftedLogNormal(ν = 1, σ = 1, τ = .20)