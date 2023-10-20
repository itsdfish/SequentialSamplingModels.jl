"""
    ExGaussian{T<:Real} <: SSM1D

The Ex-Gaussian is a convolution of the Gaussian and exponential distribution sometimes used 
to model reaction time distributions. Note that this is not technically a sequential sampling model. 

# Parameters 

- `μ`: mean of Gaussian component
- `σ`: standard deviation of Gaussian component
- `τ`: mean of exponential component

# Constructors

    ExGaussian(μ, σ, τ)

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
A diffusion model analysis. Psychonomic bulletin & review, 16, 798-817.
"""
struct ExGaussian{T<:Real} <: SSM1D
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

ExGaussian(;μ=.5, σ=.20, τ=.20) = ExGaussian(μ, σ, τ)

function rand(rng::AbstractRNG, dist::ExGaussian)
    (;μ,σ,τ) = dist
    return rand(Normal(μ, σ)) + rand(Exponential(τ))
end

function logpdf(d::ExGaussian, rt::Float64)
    (;μ,σ,τ) = d
    return log(1 / τ) + (μ - rt) / τ + (σ^2 / 2τ^2) + 
        logcdf(Normal(0, 1), (rt - μ) / σ - σ / τ)
end

function pdf(d::ExGaussian, rt::Float64)
    (;μ,σ,τ) = d
    return (1 / τ) * exp((μ - rt) / τ + (σ^2 / 2τ^2)) * Φ((rt - μ) / σ - (σ / τ))
end

mean(d::ExGaussian) = d.μ + d.τ

std(d::ExGaussian) = √(d.σ^2 + d.τ^2)