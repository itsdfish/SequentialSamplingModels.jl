"""
    LNR(;μ, σ, ϕ)

A lognormal race model object 

# Fields 

- `μ`: a vector of means in log-space
- `σ`: a standard deviation parameter in log-space
- `ϕ`: a encoding-response offset

# Example

```julia
using SequentialSamplingModels
dist = LNR(μ=[-2,-3], σ=1.0, ϕ=.3)
data = rand(dist, 10)
like = pdf.(dist, data)
loglike = logpdf.(dist, data)
```
# References

Rouder, J. N., Province, J. M., Morey, R. D., Gomez, P., & Heathcote, A. (2015). 
The lognormal race: A cognitive-process model of choice and latency with desirable 
psychometric properties. Psychometrika, 80(2), 491-513.
"""
struct LNR{T<:Real} <: SSM2D
    μ::Vector{T}
    σ::T
    ϕ::T
end

function LNR(μ, σ, ϕ)
    _, σ, ϕ = promote(μ[1], σ, ϕ)
    μ = convert(Vector{typeof(σ)}, μ)
    return LNR(μ, σ, ϕ)
end

function params(d::LNR)
    return (d.μ, d.σ, d.ϕ)    
end

LNR(;μ, σ, ϕ) = LNR(μ, σ, ϕ)

function rand(rng::AbstractRNG, dist::LNR)
    (;μ,σ,ϕ) = dist
    x = @. rand(rng, LogNormal(μ, σ)) + ϕ
    rt,choice = findmin(x)
    return (;choice,rt)
end

function logpdf(d::LNR, r::Int, t::Float64)
    (;μ,σ,ϕ) = d
    LL = 0.0
    for (i,m) in enumerate(μ)
        if i == r
            LL += logpdf(LogNormal(m, σ), t - ϕ)
        else
            LL += logccdf(LogNormal(m, σ), t - ϕ)
        end
    end
    return LL
end

# function logpdf(d::LNR{T1,T2,Vector{T3}}, r::Int, t::Float64) where {T1,T2,T3}
#     (;μ,σ,ϕ) = d
#     LL = 0.0
#     for (i,m) in enumerate(μ)
#         if i == r
#             LL += logpdf(LogNormal(m, σ), t - ϕ[i])
#         else
#             LL += logccdf(LogNormal(m, σ), t - ϕ[i])
#         end
#     end
#     return LL
# end

logpdf(d::LNR, data::Tuple) = logpdf(d, data...)

pdf(d::LNR, data::Tuple) = pdf(d, data...)

function pdf(d::LNR, r::Int, t::Float64)
    (;μ,σ,ϕ) = d
    density = 1.0
    for (i,m) in enumerate(μ)
        if i == r
            density *= pdf(LogNormal(m, σ), t - ϕ)
        else
            density *= (1 - cdf(LogNormal(m, σ), t - ϕ))
        end
    end
    return density
end
