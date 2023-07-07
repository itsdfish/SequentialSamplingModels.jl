loglikelihood(d::AbstractWald, data::AbstractArray{T,1}) where {T} = sum(logpdf.(d, data))

"""
    Wald{T<:Real} <: AbstractWald

A model object for the Wald model, also known as the inverse Gaussian model.

# Parameters 

- `ν`: drift rate
- `α`: decision threshold
- `θ`: a encoding-response offset

# Constructors

    Wald(ν, α, θ)

    Wald(;ν, α, θ)

# Example

```julia
using SequentialSamplingModels
dist = Wald(ν=3.0, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
# References
Anders, R., Alario, F., & Van Maanen, L. (2016). The shifted Wald distribution for response time data analysis. Psychological methods, 21(3), 309.

Folks, J. L., & Chhikara, R. S. (1978). The inverse Gaussian distribution and its statistical application—a review. Journal of the Royal Statistical Society Series B: Statistical Methodology, 40(3), 263-275.
"""
struct Wald{T<:Real} <: AbstractWald
    ν::T
    α::T
    θ::T
end

Wald(;ν, α, θ) = Wald(ν, α, θ)

function Wald(ν, α, θ)
    return Wald(promote(ν, α, θ)...)
end

function params(d::Wald)
    return (d.ν, d.α, d.θ)    
end

function pdf(d::AbstractWald, t::AbstractFloat)
    return pdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.θ)
end

function logpdf(d::AbstractWald, t::AbstractFloat)
    return logpdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.θ)
end

function logccdf(d::Wald, t::AbstractFloat)
    return logccdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.θ)
end

function cdf(d::Wald, t::AbstractFloat)
    return cdf(InverseGaussian(d.α/d.ν, d.α^2), t - d.θ)
end

rand(rng::AbstractRNG, d::AbstractWald) = rand(rng, InverseGaussian(d.α/d.ν, d.α^2)) + d.θ

function rand(rng::AbstractRNG, d::AbstractWald, n::Int)
    return rand(rng, InverseGaussian(d.α / d.ν, d.α^2), n) .+ d.θ
end

mean(d::AbstractWald) = mean(InverseGaussian(d.α / d.ν, d.α^2)) + d.θ
std(d::AbstractWald) = std(InverseGaussian(d.α / d.ν, d.α^2))
