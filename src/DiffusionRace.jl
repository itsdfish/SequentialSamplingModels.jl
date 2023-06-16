"""
    WaldA(;ν, k, A, θ)

Constructor for Wald distribution 

# Fields 

- `ν`: drift rate
- `k`: k = b - A where b is the decision threshold, and A is the maximum starting point
- `A`: the maximum starting point diffusion process, sampled from Uniform distribution
- `θ`: a encoding-motor time offset

## Usage

```julia
using SequentialSamplingModels
dist = WaldA(ν=.5, σ=1.0, ϕ=.3)
data = rand(dist, 10)
like = pdf.(dist, data)
loglike = logpdf.(dist, data)
```
# References

Tillman, G., Van Zandt, T., & Logan, G. D. (2020). Sequential sampling models without random between-trial variability: 
The racing diffusion model of speeded decision making. Psychonomic Bulletin & Review, 27, 911-936.
"""
struct WaldA{T1,T2,T3,T4} <: ContinuousUnivariateDistribution
    ν::T1
    k::T2
    A::T3
    θ::T4
end

Broadcast.broadcastable(x::WaldA) = Ref(x)

function params(d::WaldA)
    (d.ν, d.k, d.A, d.θ)    
end

WaldA(;ν, k, A, θ) = WaldA(ν, k, A, θ)

Φ(x) = cdf(Normal(0, 1), x)
ϕ(x) = pdf(Normal(0, 1), x)

function pdf(d::WaldA, rt::Float64)
    (;ν, k, A, θ) = d
    t = rt - θ
    α = (k - t * ν) / √(t)
    β = (A + k - t * ν) / √(t)
    return (-ν * Φ(α) + 1 / √(t) * ϕ(α) +
        ν * Φ(β) - 1 / √(t) * ϕ(β)) / A
end

logpdf(d::WaldA, rt::Float64) = log(pdf(d, rt))

function cdf(d::WaldA, rt::Float64)
    (;ν, k, A, θ) = d
    t = rt - θ
    b = A + k
    α1 = √(2) * ((ν * t - b)/√(2 * t))
    α2 = √(2) * ((ν * t - k)/√(2 * t))
    β1 = √(2) * ((-ν * t - b)/√(2 * t))
    β2 = √(2) * ((-ν * t - k)/√(2 * t))
    v1 = (1 / (2 * ν * A)) * (Φ(α2) - Φ(α1))
    v2 = (√(t) / A) * (α2 * Φ(α2) - α1 * Φ(α1))
    v3 = -(1 / (2 * ν * A)) * (exp(2 * ν * k) * Φ(β2) - exp(2 * ν * b) * Φ(β1))
    v4 = (√(t) / A) * (ϕ(α2) - ϕ(α1))
    return v1 + v2 + v3 + v4
end

logccdf(d::WaldA, rt::Float64) = log(1 - cdf(d, rt))

# function rand(d::WaldA, α)
#     @unpack ν, θ = d
#     return rand(InverseGaussian(α / ν, α^2)) + θ
# end

function rand(d::WaldA)
    (;ν, k, A, θ) = d
    z = rand(Uniform(0, A))
    α = k + A - z 
    return rand(InverseGaussian(α / ν, α^2)) + θ
end

"""
    DiffusionRace(;ν, k, A, θ)

An object for the racing diffusion model. 

# Fields

- `ν`: a vector of drift rates
- `k`: k = b - A where b is the decision threshold, and A is the maximum starting point
- `A`: the maximum starting point diffusion process, sampled from Uniform distribution
- `θ`: a encoding-motor time offset

# Example

```julia
using SequentialSamplingModels
dist = DiffusionRace(;ν=[1,2], k=.3, A=.7, θ=.2)
data = rand(dist, 10)
like = pdf.(dist, data)
loglike = logpdf.(dist, data)
```
# References

Tillman, G., Van Zandt, T., & Logan, G. D. (2020). Sequential sampling models without random between-trial variability: 
The racing diffusion model of speeded decision making. Psychonomic Bulletin & Review, 27, 911-936.
"""
struct DiffusionRace{T1,T2,T3,T4} <: SequentialSamplingModel
    ν::T1
    k::T2
    A::T3
    θ::T4
end

Broadcast.broadcastable(x::DiffusionRace) = Ref(x)

function params(d::DiffusionRace)
    (d.ν, d.k, d.A, d.θ)    
end

loglikelihood(d::DiffusionRace, data) = sum(logpdf.(d, data...))

DiffusionRace(;ν, k, A, θ) = DiffusionRace(ν, k, A, θ)

function rand(dist::DiffusionRace)
    (;ν, A, k, θ) = dist
    z = rand(Uniform(0, A))
    α = k + A - z 
    x = @. rand(WaldA(ν, k, A, θ))
    rt,resp = findmin(x)
    return resp,rt
end

function rand(d::DiffusionRace, N::Int)
    choice = fill(0, N)
    rt = fill(0.0, N)
    for i in 1:N
        choice[i],rt[i] = rand(d)
    end
    return (choice=choice,rt=rt)
end

function logpdf(d::DiffusionRace, r::Int, rt::Float64)
    (;ν, k, A, θ) = d
    LL = 0.0
    for (i,m) in enumerate(ν)
        if i == r
            LL += logpdf(WaldA(m, k, A, θ), rt)
        else
            LL += logccdf(WaldA(m, k, A, θ), rt)
        end
    end
    return LL
end

logpdf(d::DiffusionRace, data::Tuple) = logpdf(d, data...)

pdf(d::DiffusionRace, data::Tuple) = pdf(d, data...)

function pdf(d::DiffusionRace, r::Int, rt::Float64)
    (;ν, k, A, θ) = d
    like = 1.0
    for (i,m) in enumerate(ν)
        if i == r
            like *= pdf(WaldA(m, k, A, θ), rt)
        else
            like *= (1 - cdf(WaldA(m, k, A, θ), rt))
        end
    end
    return like
end













# using Distributions, Parameters
# import Base.rand
# import Distributions: pdf, logpdf, cdf
# struct Race{T1,T2,T3} <: ContinuousUnivariateDistribution
#     ν::T1
#     α::T2
#     θ::T3
# end

# Broadcast.broadcastable(x::Race) = Ref(x)

# Race(;ν, α, θ) = Race(ν, α, θ)

# function rand(dist::Race)
#     @unpack ν, α, θ = dist
#     x = @. rand(Wald(ν, α, θ))
#     rt,resp = findmin(x)
#     return resp,rt
# end

# rand(dist::Race, N::Int) = [rand(dist) for i in 1:N]

# function logpdf(d::Race, r::Int, rt::Float64)
#     @unpack ν, α, θ = d
#     LL = 0.0
#     for (i,m) in enumerate(ν)
#         if i == r
#             LL += logpdf(Wald(;ν, α, θ), rt)
#         else
#             LL += logccdf(Wald(;ν, α, θ), rt)
#         end
#     end
#     return LL
# end

# logpdf(d::Race, data::Tuple) = logpdf(d, data...)

# pdf(d::Race, data::Tuple) = pdf(d, data...)

# function pdf(d::Race, r::Int, rt::Float64)
#     @unpack ν, α, θ = d
#     like = 1.0
#     for (i,m) in enumerate(ν)
#         if i == r
#             like *= pdf(Wald(;ν=m, α, θ), rt)
#         else
#             like *= (1 - cdf(Wald(;ν=m, α, θ), rt))
#         end
#     end
#     return like
# end

