abstract type AbstractWald <: ContinuousUnivariateDistribution
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
    υ::T1
    α::T2
    θ::T3
end

Wald(;υ, α, θ) = Wald(υ, α, θ)

function pdf(d::AbstractWald, t::Float64)
    return pdf(InverseGaussian(d.α/d.υ, d.α^2), t - d.θ)
end

function logpdf(d::AbstractWald, t::Float64)
    return logpdf(InverseGaussian(d.α/d.υ, d.α^2), t - d.θ)
end

rand(d::AbstractWald) = rand(InverseGaussian(d.α/d.υ, d.α^2)) + d.θ

function rand(d::AbstractWald, n::Int)
    return rand(InverseGaussian(d.α/d.υ, d.α^2), n) .+ d.θ
end

mean(d::AbstractWald) = mean(InverseGaussian(d.α/d.υ, d.α^2)) + d.θ
std(d::AbstractWald) = std(InverseGaussian(d.α/d.υ, d.α^2))

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
    υ::T1
    σ::T2
    α::T3
    θ::T4
end

WaldMixture(;υ, σ, α, θ) = WaldMixture(υ,σ, α, θ)

function pdf(d::WaldMixture, t::Float64)
    @unpack υ, σ, α ,θ = d
    c1 = α/√(2*π*(t - θ)^3)
    c2 = 1/cdf(Normal(0,1), υ/σ)
    c3 = exp(-(υ*(t - θ) - α)^2/(2*(t - θ)*((t - θ)*σ^2 + 1)))
    c4 = (α*σ^2 + υ)/√(σ^2*((t - θ)*σ^2 + 1))
    return c1*c2*c3*cdf(Normal(0,1), c4)
end

function logpdf(d::WaldMixture, t::Float64)
    @unpack υ, σ, α ,θ = d
    c1 = log(α) - log(√(2*π*(t - θ)^3))
    c2 = log(1) - logcdf(Normal(0,1), υ/σ)
    c3 = -(υ*(t - θ) - α)^2/(2*(t - θ)*((t - θ)*σ^2 + 1))
    c4 = (α*σ^2 + υ)/√(σ^2*((t - θ)*σ^2 + 1))
    return c1 + c2 + c3 + logcdf(Normal(0,1), c4)
end

function rand(d::WaldMixture) 
    x = rand(truncated(Normal(d.υ, d.σ), 0, Inf))
    return rand(InverseGaussian(d.α/x, d.α^2)) + d.θ
end

function rand(d::WaldMixture, n::Int)
    return map(x->rand(d), 1:n)
end