"""
    WaldMixture{T<:Real} <: AbstractWald

# Parameters

- `υ`: drift rate. ν ∈ ℝ⁺. 
- `η`: standard deviation of drift rate. ν ∈ ℝ⁺.
- `α`: decision threshold. α ∈ ℝ⁺.
- `τ`: a encoding-response offset. τ ∈ [0, min_rt]

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    WaldMixture(ν, η, α, τ)
    
The second constructor uses keywords with default values, and is not order dependent: 

    WaldMixture(;ν=3.0, η=.2, α=.5, τ=.130)
    
# Example

```julia
using SequentialSamplingModels
dist = WaldMixture(;ν=3.0, η=.2, α=.5, τ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
# References
Steingroever, H., Wabersich, D., & Wagenmakers, E. J. (2020). 
Modeling across-trial variability in the Wald drift rate parameter. 
Behavior Research Methods, 1-17.
"""
struct WaldMixture{T <: Real} <: AbstractWald
    ν::T
    η::T
    α::T
    τ::T
    function WaldMixture(ν::T, η::T, α::T, τ::T) where {T <: Real}
        @argcheck η ≥ 0
        @argcheck α ≥ 0
        @argcheck τ ≥ 0
        return new{T}(ν, η, α, τ)
    end
end

function WaldMixture(ν, η, α, τ)
    return WaldMixture(promote(ν, η, α, τ)...)
end

WaldMixture(; ν = 3.0, η = 0.2, α = 0.5, τ = 0.130) = WaldMixture(ν, η, α, τ)

function params(d::WaldMixture)
    return (d.ν, d.η, d.α, d.τ)
end

function pdf(d::WaldMixture, rt::AbstractFloat)
    (; ν, η, α, τ) = d
    @argcheck τ ≤ rt
    c1 = α / √((2 * π * (rt - τ)^3) * ((rt - τ) * η^2 + 1))
    c2 = 1 / Φ(ν / η)
    c3 = exp(-(ν * (rt - τ) - α)^2 / (2 * (rt - τ) * ((rt - τ) * η^2 + 1)))
    c4 = (α * η^2 + ν) / √(η^2 * ((rt - τ) * η^2 + 1))
    return c1 * c2 * c3 * Φ(c4)
end

function logpdf(d::WaldMixture, rt::AbstractFloat)
    (; ν, η, α, τ) = d
    @argcheck τ ≤ rt
    c1 = log(α) - log(√((2 * π * (rt - τ)^3) * ((rt - τ) * η^2 + 1)))
    c2 = log(1) - logcdf(Normal(0, 1), ν / η)
    c3 = -(ν * (rt - τ) - α)^2 / (2 * (rt - τ) * ((rt - τ) * η^2 + 1))
    c4 = (α * η^2 + ν) / √(η^2 * ((rt - τ) * η^2 + 1))
    return c1 + c2 + c3 + logcdf(Normal(0, 1), c4)
end

function rand(rng::AbstractRNG, d::WaldMixture)
    x = rand(rng, truncated(Normal(d.ν, d.η), 0, Inf))
    return rand(rng, InverseGaussian(d.α / x, d.α^2)) + d.τ
end

function rand(rng::AbstractRNG, d::WaldMixture, n::Int)
    return map(x -> rand(rng, d), 1:n)
end

function simulate(rng::AbstractRNG, model::WaldMixture; Δt = 0.001)
    (; ν, α, η) = model
    x = 0.0
    t = 0.0
    evidence = [0.0]
    time_steps = [t]
    ν′ = rand(truncated(Normal(ν, η), 0, Inf))
    while x .< α
        t += Δt
        x = increment!(rng, model, x, ν′; Δt)
        push!(evidence, x)
        push!(time_steps, t)
    end
    return time_steps, evidence
end

function increment!(rnd::AbstractRNG, model::WaldMixture, x, Δμ; Δt = 0.001)
    return x += Δμ * Δt + rand(rnd, Normal(0.0, √(Δt)))
end
