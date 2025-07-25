"""
    Wald{T<:Real} <: AbstractWald

A model object for the Wald model, also known as the inverse Gaussian model.

# Parameters 

- `ν`: drift rate. ν ∈ ℝ⁺. 
- `α`: decision threshold. α ∈ ℝ⁺
- `τ`: a encoding-response offset. τ ∈ [0, min_rt]

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    Wald(ν, α, τ)

The second constructor uses keywords with default values, and is not order dependent: 

    Wald(;ν=1.5, α=.50, τ=0.20)

# Example

```julia
using SequentialSamplingModels
dist = Wald(ν=3.0, α=.5, τ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
# References
Anders, R., Alario, F., & Van Maanen, L. (2016). The shifted Wald distribution for response time data analysis. Psychological methods, 21(3), 309.

Folks, J. L., & Chhikara, R. S. (1978). The inverse Gaussian distribution and its statistical application—a review. Journal of the Royal Statistical Society Series B: Statistical Methodology, 40(3), 263-275.
"""
struct Wald{T <: Real} <: AbstractWald
    ν::T
    α::T
    τ::T
end

Wald(; ν = 1.5, α = 0.75, τ = 0.20) = Wald(ν, α, τ)

function Wald(ν, α, τ)
    return Wald(promote(ν, α, τ)...)
end

function params(d::Wald)
    return (d.ν, d.α, d.τ)
end

function pdf(d::AbstractWald, t::AbstractFloat)
    return pdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.τ)
end

function logpdf(d::AbstractWald, t::AbstractFloat)
    return d.ν == 0 ? -Inf : logpdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.τ)
end

function logccdf(d::Wald, t::AbstractFloat)
    return logccdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.τ)
end

function cdf(d::Wald, t::AbstractFloat)
    return cdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.τ)
end

rand(rng::AbstractRNG, d::AbstractWald) = rand(rng, InverseGaussian(d.α / d.ν, d.α^2)) + d.τ

function rand(rng::AbstractRNG, d::AbstractWald, n::Int)
    return rand(rng, InverseGaussian(d.α / d.ν, d.α^2), n) .+ d.τ
end

mean(d::AbstractWald) = mean(InverseGaussian(d.α / d.ν, d.α^2)) + d.τ
std(d::AbstractWald) = std(InverseGaussian(d.α / d.ν, d.α^2))

function simulate(rng::AbstractRNG, model::Wald; Δt = 0.001)
    (; ν, α) = model
    n = length(model.ν)
    x = 0.0
    t = 0.0
    evidence = [0.0]
    time_steps = [t]
    while x .< α
        t += Δt
        x = increment!(rng, model, x, ν; Δt)
        push!(evidence, x)
        push!(time_steps, t)
    end
    return time_steps, evidence
end

function increment!(rnd::AbstractRNG, model::AbstractWald, x, Δμ; Δt = 0.001)
    return x += Δμ * Δt + rand(rnd, Normal(0.0, √(Δt)))
end
