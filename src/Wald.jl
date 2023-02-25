abstract type AbstractWald <: SequentialSamplingModel
end

"""
# Wald Constructor
- `υ`: drift rate
- `α`: decision threshold
- `θ`: a encoding-response offset
## Usage
````julia
using SequentialSamplingModels
dist = Wald(υ=3.0, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
````
"""
struct Wald{T1,T2,T3} <: AbstractWald
    ν::T1
    α::T2
    θ::T3
end

Wald(;ν, α, θ) = Wald(ν, α, θ)

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

rand(d::AbstractWald) = rand(InverseGaussian(d.α/d.ν, d.α^2)) + d.θ

function rand(d::AbstractWald, n::Int)
    return rand(InverseGaussian(d.α / d.ν, d.α^2), n) .+ d.θ
end

mean(d::AbstractWald) = mean(InverseGaussian(d.α / d.ν, d.α^2)) + d.θ
std(d::AbstractWald) = std(InverseGaussian(d.α / d.ν, d.α^2))

"""
# WaldMixture Constructor
- `υ`: drift rate
- `σ`: standard deviation of drift rate
- `α`: decision threshold
- `θ`: a encoding-response offset
## Usage
````julia
using SequentialSamplingModels
dist = WaldMixture(υ=3.0, σ=.2, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
````
## References
Steingroever, H., Wabersich, D., & Wagenmakers, E. J. (2020). 
Modeling across-trial variability in the Wald drift rate parameter. 
Behavior Research Methods, 1-17.
"""
struct WaldMixture{T1,T2,T3,T4} <: AbstractWald
    ν::T1
    σ::T2
    α::T3
    θ::T4
end

WaldMixture(;ν, σ, α, θ) = WaldMixture(ν, σ, α, θ)

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

function rand(d::WaldMixture) 
    x = rand(truncated(Normal(d.ν, d.σ), 0, Inf))
    return rand(InverseGaussian(d.α / x, d.α^2)) + d.θ
end

function rand(d::WaldMixture, n::Int)
    return map(x -> rand(d), 1:n)
end