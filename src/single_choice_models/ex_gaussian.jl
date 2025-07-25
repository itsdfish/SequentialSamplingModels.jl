"""
    ExGaussian{T<:Real} <: SSM1D

The Ex-Gaussian is a convolution of the Gaussian and exponential distribution sometimes used 
to model reaction time distributions. Note that this is not technically a sequential sampling model. 

# Parameters 

- `μ::T`: mean of Gaussian component. μ ∈ ℝ⁺.
- `σ::T`: standard deviation of Gaussian component. σ ∈ ℝ⁺
- `τ::T`: mean of exponential component. τ ∈ [0, min_rt]

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    ExGaussian(μ, σ, τ)

The second constructor uses kewords, and is not order dependent: 

    ExGaussian(;μ=.5, σ=.20, τ=.20) 

# Example

```julia
using SequentialSamplingModels
dist = ExGaussian(;μ=.5, σ=.20, τ=.20) 
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
# References

Matzke, D., & Wagenmakers, E. J. (2009). Psychological interpretation of the ex-Gaussian and shifted Wald parameters: 
A diffusion model analysis. Psychonomic Bulletin & Review, 16, 798-817.
"""
struct ExGaussian{T <: Real} <: SSM1D
    μ::T
    σ::T
    τ::T
end

function ExGaussian(μ, σ, τ)
    μ, σ, τ = promote(μ, σ, τ)
    return ExGaussian(μ, σ, τ)
end

function params(d::ExGaussian)
    return (d.μ, d.σ, d.τ)
end

ExGaussian(; μ = 0.5, σ = 0.20, τ = 0.20) = ExGaussian(μ, σ, τ)

function rand(rng::AbstractRNG, dist::ExGaussian)
    (; μ, σ, τ) = dist
    return rand(Normal(μ, σ)) + rand(Exponential(τ))
end

function logpdf(d::ExGaussian, rt::Float64)
    (; μ, σ, τ) = d
    if τ == 0
        return -Inf
    end
    return log(1 / τ) +
           (μ - rt) / τ +
           (σ^2 / 2τ^2) +
           logcdf(Normal(0, 1), (rt - μ) / σ - σ / τ)
end

# exponentiate logpdf for numerical stability
pdf(d::ExGaussian, rt::Float64) = exp(logpdf(d, rt))

mean(d::ExGaussian) = d.μ + d.τ

std(d::ExGaussian) = √(d.σ^2 + d.τ^2)
