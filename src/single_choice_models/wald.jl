"""
    Wald{T<:Real} <: AbstractWald

A single choice implementation of a drift diffusion model. When `η = 0`, the model reduces to a standard Wald model 
without inter-trial variability in the drift rate `ν`. 

# Parameters

- `υ`: drift rate. ν ∈ ℝ⁺. 
- `η`: standard deviation of drift rate. ν ∈ ℝ⁺.
- `α`: decision threshold. α ∈ ℝ⁺.
- `τ`: a encoding-response offset. τ ∈ [0, min_rt]

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    Wald(ν, η, α, τ)
    
The second constructor uses keywords with default values, and is not order dependent: 

    Wald(;ν=3.0, η=.2, α=.5, τ=.130)
    
# Example

```julia
using SequentialSamplingModels
dist = Wald(;ν=3.0, η=.2, α=.5, τ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
# References

Anders, R., Alario, F., & Van Maanen, L. (2016). The shifted Wald distribution for response time data analysis. Psychological methods, 21(3), 309.

Folks, J. L., & Chhikara, R. S. (1978). The inverse Gaussian distribution and its statistical application—a review. Journal of the Royal Statistical Society Series B: Statistical Methodology, 40(3), 263-275.

Steingroever, H., Wabersich, D., & Wagenmakers, E. J. (2020). 
Modeling across-trial variability in the Wald drift rate parameter. 
Behavior Research Methods, 1-17.
"""
struct Wald{T <: Real} <: AbstractWald
    ν::T
    η::T
    α::T
    τ::T
    function Wald(ν::T, η::T, α::T, τ::T) where {T <: Real}
        @argcheck ν ≥ 0
        @argcheck η ≥ 0
        @argcheck α ≥ 0
        @argcheck τ ≥ 0
        return new{T}(ν, η, α, τ)
    end
end

function Wald(ν, η, α, τ)
    return Wald(promote(ν, η, α, τ)...)
end

Wald(; ν = 3.0, η = 0.2, α = 0.5, τ = 0.130) = Wald(ν, η, α, τ)

function params(d::Wald)
    return (d.ν, d.η, d.α, d.τ)
end

function pdf(d::Wald, rt::AbstractFloat)
    return d.η > 0 ? _pdf_full(d, rt) : _pdf_partial(d, rt)
end

function _pdf_full(d::Wald, rt::AbstractFloat)
    (; ν, η, α, τ) = d
    c1 = α / √((2 * π * (rt - τ)^3) * ((rt - τ) * η^2 + 1))
    c2 = 1 / Φ(ν / η)
    c3 = exp(-(ν * (rt - τ) - α)^2 / (2 * (rt - τ) * ((rt - τ) * η^2 + 1)))
    c4 = (α * η^2 + ν) / √(η^2 * ((rt - τ) * η^2 + 1))
    return c1 * c2 * c3 * Φ(c4)
end

function _pdf_partial(d::Wald, rt::AbstractFloat)
    return pdf(InverseGaussian(d.α / d.ν, d.α^2), rt - d.τ)
end

function logpdf(d::Wald, rt::AbstractFloat)
    @argcheck d.τ ≤ rt
    return d.η > 0 ? _logpdf_full(d, rt) : _logpdf_partial(d, rt)
end

function _logpdf_partial(d::Wald, rt::AbstractFloat)
    return d.ν == 0 ? -Inf : logpdf(InverseGaussian(d.α / d.ν, d.α^2), rt - d.τ)
end

function _logpdf_full(d::Wald, rt::AbstractFloat)
    (; ν, η, α, τ) = d
    c1 = log(α) - log(√((2 * π * (rt - τ)^3) * ((rt - τ) * η^2 + 1)))
    c2 = log(1) - logcdf(Normal(0, 1), ν / η)
    c3 = -(ν * (rt - τ) - α)^2 / (2 * (rt - τ) * ((rt - τ) * η^2 + 1))
    c4 = (α * η^2 + ν) / √(η^2 * ((rt - τ) * η^2 + 1))
    return c1 + c2 + c3 + logcdf(Normal(0, 1), c4)
end

function rand(rng::AbstractRNG, d::Wald)
    x = rand(rng, truncated(Normal(d.ν, d.η), 0, Inf))
    return rand(rng, InverseGaussian(d.α / x, d.α^2)) + d.τ
end

function rand(rng::AbstractRNG, d::Wald, n::Int)
    return map(x -> rand(rng, d), 1:n)
end

function simulate(rng::AbstractRNG, model::Wald; Δt = 0.001)
    (; ν, α, η) = model
    x = 0.0
    t = 0.0
    evidence = [0.0]
    time_steps = [t]
    ν′ = rand(rng, truncated(Normal(ν, η), 0, Inf))
    while x < α
        t += Δt
        x = increment!(rng, model, x, ν′; Δt)
        push!(evidence, x)
        push!(time_steps, t)
    end
    return time_steps, evidence
end

function increment!(rnd::AbstractRNG, model::Wald, x, Δμ; Δt = 0.001)
    return x += Δμ * Δt + rand(rnd, Normal(0.0, √(Δt)))
end
