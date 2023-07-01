
"""
    WaldMixture

# Fields

- `υ`: drift rate
- `σ`: standard deviation of drift rate
- `α`: decision threshold
- `θ`: a encoding-response offset

## Example
```julia
using SequentialSamplingModels
dist = WaldMixture(υ=3.0, σ=.2, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
## References
Steingroever, H., Wabersich, D., & Wagenmakers, E. J. (2020). 
Modeling across-trial variability in the Wald drift rate parameter. 
Behavior Research Methods, 1-17.
"""
struct WaldMixture{T<:Real} <: AbstractWald
    ν::T
    σ::T
    α::T
    θ::T
end

function WaldMixture(ν, σ, α, θ)
    return WaldMixture(promote(ν, σ, α, θ)...)
end

WaldMixture(;ν, σ, α, θ) = WaldMixture(ν, σ, α, θ)

function params(d::WaldMixture)
    return (d.ν, d.σ, d.α, d.θ)    
end

function pdf(d::WaldMixture, t::AbstractFloat)
    (;ν, σ, α ,θ) = d
    c1 = α / √(2 * π * (t - θ)^3)
    c2 = 1 / cdf(Normal(0,1), ν / σ)
    c3 = exp(-(ν * (t - θ) - α)^2 / (2 * (t - θ) * ((t - θ) * σ^2 + 1)))
    c4 = (α * σ^2 + ν) / √(σ^2 * ((t - θ)*σ^2 + 1))
    return c1 * c2 * c3 * cdf(Normal(0,1), c4)
end

function logpdf(d::WaldMixture, t::AbstractFloat)
    (;ν, σ, α ,θ) = d
    c1 = log(α) - log(√(2 * π * (t - θ)^3))
    c2 = log(1) - logcdf(Normal(0,1), ν / σ)
    c3 = -(ν * (t - θ) - α)^2 / (2*(t - θ)*((t - θ)*σ^2 + 1))
    c4 = (α * σ^2 + ν) / √(σ^2 * ((t - θ) * σ^2 + 1))
    return c1 + c2 + c3 + logcdf(Normal(0,1), c4)
end

function rand(rng::AbstractRNG, d::WaldMixture) 
    x = rand(rng, truncated(Normal(d.ν, d.σ), 0, Inf))
    return rand(rng, InverseGaussian(d.α / x, d.α^2)) + d.θ
end

function rand(rng::AbstractRNG, d::WaldMixture, n::Int)
    return map(x -> rand(rng, d), 1:n)
end