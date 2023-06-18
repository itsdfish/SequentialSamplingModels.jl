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
struct LNR{T1,T2,T3} <: SequentialSamplingModel
    μ::T1
    σ::T2
    ϕ::T3
end

Broadcast.broadcastable(x::LNR) = Ref(x)

function params(d::LNR)
    (d.μ, d.σ, d.ϕ)    
end

loglikelihood(d::LNR, data) = sum(logpdf.(d, data...))

LNR(;μ, σ, ϕ) = LNR(μ, σ, ϕ)

function rand(rng::AbstractRNG, dist::LNR)
    (;μ,σ,ϕ) = dist
    x = @. rand(rng, LogNormal(μ, σ)) + ϕ
    rt,resp = findmin(x)
    return resp,rt
end

function rand(rng::AbstractRNG, d::LNR, N::Int)
    choice = fill(0, N)
    rt = fill(0.0, N)
    for i in 1:N
        choice[i],rt[i] = rand(rng, d)
    end
    return (choice=choice,rt=rt)
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

function logpdf(d::LNR{T1,T2,Vector{T3}}, r::Int, t::Float64) where {T1,T2,T3}
    (;μ,σ,ϕ) = d
    LL = 0.0
    for (i,m) in enumerate(μ)
        if i == r
            LL += logpdf(LogNormal(m, σ), t - ϕ[i])
        else
            LL += logccdf(LogNormal(m, σ), t - ϕ[i])
        end
    end
    return LL
end

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
