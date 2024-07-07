"""
    Wald{T<:Real} <: AbstractWald

# Parameters

- `υ`: drift rate
- `η`: standard deviation of drift rate
- `α`: decision threshold
- `τ`: a encoding-response offset

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

Folks, J. L., & Chhikara, R. S. (1978). The inverse Gaussian distribution and its statistical application—a review. 
Journal of the Royal Statistical Society Series B: Statistical Methodology, 40(3), 263-275.

Steingroever, H., Wabersich, D., & Wagenmakers, E. J. (2020). Modeling across-trial variability in the Wald drift rate parameter. 
Behavior Research Methods, 1-17.
"""
struct Wald{T <: Real} <: AbstractWald
    ν::T
    η::T
    α::T
    τ::T
end

function Wald(ν, η, α, τ)
    return Wald(promote(ν, η, α, τ)...)
end

Wald(; ν = 3.0, η = 0.2, α = 0.5, τ = 0.130) = Wald(ν, η, α, τ)

function params(d::Wald)
    return (d.ν, d.η, d.α, d.τ)
end

function pdf(d::AbstractWald, t::Real)
    (; ν, η, α, τ) = d
    return η ≈ 0 ? pdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.τ) : _pdf(d, t)
end

function _pdf(d::AbstractWald, t::Real)
    (; ν, η, α, τ) = d
    c1 = α / √((2 * π * (t - τ)^3) * ((t - τ) * η^2 + 1))
    c2 = 1 / Φ(ν / η)
    c3 = exp(-(ν * (t - τ) - α)^2 / (2 * (t - τ) * ((t - τ) * η^2 + 1)))
    c4 = (α * η^2 + ν) / √(η^2 * ((t - τ) * η^2 + 1))
    return c1 * c2 * c3 * Φ(c4)
end

function logpdf(d::AbstractWald, t::Real)
    (; ν, η, α, τ) = d
    return η ≈ 0 ? logpdf(InverseGaussian(d.α / d.ν, d.α^2), t - d.τ) : _logpdf(d, t)
end

function _logpdf(d::AbstractWald, t::Real)
    (; ν, η, α, τ) = d
    c1 = log(α) - log(√((2 * π * (t - τ)^3) * ((t - τ) * η^2 + 1)))
    c2 = log(1) - logcdf(Normal(0, 1), ν / η)
    c3 = -(ν * (t - τ) - α)^2 / (2 * (t - τ) * ((t - τ) * η^2 + 1))
    c4 = (α * η^2 + ν) / √(η^2 * ((t - τ) * η^2 + 1))
    return c1 + c2 + c3 + logcdf(Normal(0, 1), c4)
end

function rand(rng::AbstractRNG, d::AbstractWald)
    (; ν, η, α, τ) = d
    x = η ≈ 0 ? ν : rand(rng, truncated(Normal(d.ν, d.η), 0, Inf))
    return rand(rng, InverseGaussian(d.α / x, d.α^2)) + d.τ
end

# function rand(rng::AbstractRNG, d::Wald, n::Int)
#     return map(_ -> rand(rng, d), 1:n)
# end

"""
    simulate(model::Wald; Δt=.001)

Returns a matrix containing evidence samples of the Wald mixture decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::Wald`: an Wald mixture model object

# Keywords

- `Δt=.001`: size of time step of decision process in seconds
"""
function simulate(model::Wald; Δt = 0.001)
    (; ν, α, η) = model
    n = length(model.ν)
    x = 0.0
    t = 0.0
    evidence = [0.0]
    time_steps = [t]
    ν′ = rand(truncated(Normal(ν, η), 0, Inf))
    while x .< α
        t += Δt
        x += ν′ * Δt + rand(Normal(0.0, √(Δt)))
        push!(evidence, x)
        push!(time_steps, t)
    end
    return time_steps, evidence
end
