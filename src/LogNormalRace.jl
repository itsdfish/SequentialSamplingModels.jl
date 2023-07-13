"""
    LNR{T<:Real} <: SSM2D

A lognormal race model object. 

# Parameters 

- `ν`: a vector of means in log-space
- `σ`: a standard deviation parameter in log-space
- `τ`: a encoding-response offset

# Constructors

    LNR(ν, σ, τ)

    LNR(;ν, σ, τ)

# Example

```julia
using SequentialSamplingModels
dist = LNR(ν=[-2,-3], σ=1.0, τ=.3)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```
# References

Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). 
The lognormal race: A cognitive-process model of choice and latency with desirable 
psychometric properties. Psychometrika, 80(2), 491-513.
"""
struct LNR{T<:Real} <: SSM2D
    ν::Vector{T}
    σ::T
    τ::T
end

function LNR(ν, σ, τ)
    _, σ, τ = promote(ν[1], σ, τ)
    ν = convert(Vector{typeof(σ)}, ν)
    return LNR(ν, σ, τ)
end

function params(d::LNR)
    return (d.ν, d.σ, d.τ)    
end

LNR(;ν, σ, τ) = LNR(ν, σ, τ)

function rand(rng::AbstractRNG, dist::LNR)
    (;ν,σ,τ) = dist
    x = @. rand(rng, LogNormal(ν, σ)) + τ
    rt,choice = findmin(x)
    return (;choice,rt)
end

function logpdf(d::LNR, r::Int, t::Float64)
    (;ν,σ,τ) = d
    LL = 0.0
    for (i,m) in enumerate(ν)
        if i == r
            LL += logpdf(LogNormal(m, σ), t - τ)
        else
            LL += logccdf(LogNormal(m, σ), t - τ)
        end
    end
    return LL
end

function pdf(d::LNR, r::Int, t::Float64)
    (;ν,σ,τ) = d
    density = 1.0
    for (i,m) in enumerate(ν)
        if i == r
            density *= pdf(LogNormal(m, σ), t - τ)
        else
            density *= (1 - cdf(LogNormal(m, σ), t - τ))
        end
    end
    return density
end
