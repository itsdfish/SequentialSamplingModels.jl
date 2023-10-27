"""
    PoissonRace{T<:Real} <: AbstractPoissonRace

# Parameters 

- `ν`: gamma scale parameter
- `α`: threshold
- `τ`: a encoding-response offset

# Constructors

    PoissonRace(ν, α, τ)

    PoissonRace(;ν=[.05,.06], α=[5,5], τ=.3)

# Example

```julia
using SequentialSamplingModels
dist = PoissonRace(ν=[.05,.06], α=[5,5], τ=.3)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```
# References

LaBerge, D. A. (1962). A recruitment model of simple behavior. Psychometrika, 27, 375-395.
"""
struct PoissonRace{T<:Real} <: AbstractPoissonRace
    ν::Vector{T}
    α::Vector{Int}
    τ::T
end

function PoissonRace(ν, α, τ)
    _, τ = promote(ν[1], τ)
    ν = convert(Vector{typeof(τ)}, ν)
    return PoissonRace(ν, α, τ)
end

function params(d::AbstractPoissonRace)
    return (d.ν, d.α, d.τ)    
end

PoissonRace(;ν=[.05,.06], α=[5,5], τ=.30) = PoissonRace(ν, α, τ)

function rand(rng::AbstractRNG, dist::AbstractPoissonRace)
    (;ν,α,τ) = dist
    x = @. rand(rng, Gamma(α, ν)) + τ
    rt,choice = findmin(x)
    return (;choice,rt)
end

function logpdf(d::AbstractPoissonRace, r::Int, t::Float64)
    (;ν,α,τ) = d
    LL = 0.0
    for i ∈ 1:length(ν)
        if i == r
            LL += logpdf(Gamma(α[i], ν[i]), t - τ)
        else
            LL += logccdf(Gamma(α[i], ν[i]), t - τ)
        end
    end
    return LL
end

function pdf(d::AbstractPoissonRace, r::Int, t::Float64)
    (;ν,α,τ) = d
    density = 1.0
    for i ∈ 1:length(ν)
        if i == r
            density *= pdf(Gamma(α[i], ν[i]), t - τ)
        else
            density *= (1 - cdf(Gamma(α[i], ν[i]), t - τ))
        end
    end
    return density
end