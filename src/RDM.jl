"""
    WaldA(;ν, k, A, τ)

Constructor for Wald distribution 

# Fields 

- `ν`: drift rate
- `k`: k = b - A where b is the decision threshold, and A is the maximum starting point
- `A`: the maximum starting point diffusion process, sampled from Uniform distribution
- `τ`: a encoding-motor time offset

# Example

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
struct WaldA{T<:Real} <: SSM1D
    ν::T
    A::T
    k::T
    τ::T
end

function WaldA(ν, A, k, τ)
    return WaldA(promote(ν, A, k, τ)...)
end

Broadcast.broadcastable(x::WaldA) = Ref(x)

function params(d::WaldA)
    (d.ν, d.A, d.k, d.τ)
end

WaldA(; ν, A, k, τ) = WaldA(ν, A, k, τ)

function pdf(d::WaldA, rt::Float64)
    (; ν, A, k, τ) = d
    t = rt - τ
    α = (k - t * ν) / √(t)
    β = (A + k - t * ν) / √(t)
    dens = (-ν * Φ(α) + 1 / √(t) * ϕ(α) + ν * Φ(β) - 1 / √(t) * ϕ(β)) / A
    return max(eps(), dens)
end

logpdf(d::WaldA, rt::Float64) = log(pdf(d, rt))

function cdf(d::WaldA, rt::Float64)
    (; ν, A, k, τ) = d
    t = rt - τ
    b = A + k
    α1 = √(2) * ((ν * t - b) / √(2 * t))
    α2 = √(2) * ((ν * t - k) / √(2 * t))
    β1 = √(2) * ((-ν * t - b) / √(2 * t))
    β2 = √(2) * ((-ν * t - k) / √(2 * t))
    v1 = (1 / (2 * ν * A)) * (Φ(α2) - Φ(α1))
    v2 = (√(t) / A) * (α2 * Φ(α2) - α1 * Φ(α1))
    v3 = -(1 / (2 * ν * A)) * (exp(2 * ν * k) * Φ(β2) - exp(2 * ν * b) * Φ(β1))
    v4 = (√(t) / A) * (ϕ(α2) - ϕ(α1))
    return v1 + v2 + v3 + v4
end

logccdf(d::WaldA, rt::Float64) = log(1 - cdf(d, rt))

function rand(rng::AbstractRNG, d::WaldA)
    (; ν, A, k, τ) = d
    z = rand(rng, Uniform(0, A))
    α = k + A - z
    return rand(rng, InverseGaussian(α / ν, α^2)) + τ
end

"""
    RDM{T<:Real} <: AbstractRDM

An object for the racing diffusion model.

# Constructors

    RDM(ν, k, A, τ)

    RDM(;ν=[1,2], k=.3, A=.7, τ=.2)

# Parameters

- `ν`: a vector of drift rates
- `A`: the maximum starting point diffusion process, sampled from Uniform distribution
- `k`: k = b - A where b is the decision threshold, and A is the maximum starting point
- `τ`: a encoding-motor time offset

# Example

```julia
using SequentialSamplingModels
dist = RDM(;ν=[1,2], k=.3, A=.7, τ=.2)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```
# References

Tillman, G., Van Zandt, T., & Logan, G. D. (2020). Sequential sampling models without random between-trial variability: 
The racing diffusion model of speeded decision making. Psychonomic Bulletin & Review, 27, 911-936.
"""
struct RDM{T<:Real} <: AbstractRDM
    ν::Vector{T}
    A::T
    k::T
    τ::T
end

function RDM(ν, k, A, τ)
    _, A, k, τ = promote(ν[1], k, A, τ)
    ν = convert(Vector{typeof(k)}, ν)
    return RDM(ν, A, k, τ)
end

function params(d::AbstractRDM)
    (d.ν, d.A, d.k, d.τ)
end

RDM(; ν = [1, 2], k = 0.3, A = 0.7, τ = 0.2) = RDM(ν, k, A, τ)

function rand(rng::AbstractRNG, dist::AbstractRDM)
    (; ν, A, k, τ) = dist
    x = @. rand(rng, WaldA(ν, k, A, τ))
    rt, choice = findmin(x)
    return (; choice, rt)
end

function logpdf(d::AbstractRDM, r::Int, rt::Float64)
    (; ν, A, k, τ) = d
    LL = 0.0
    for (i, m) in enumerate(ν)
        if i == r
            LL += logpdf(WaldA(m, k, A, τ), rt)
        else
            LL += logccdf(WaldA(m, k, A, τ), rt)
        end
    end
    return LL
end

function pdf(d::AbstractRDM, r::Int, rt::Float64)
    (; ν, A, k, τ) = d
    like = 1.0
    for (i, m) in enumerate(ν)
        if i == r
            like *= pdf(WaldA(m, k, A, τ), rt)
        else
            like *= (1 - cdf(WaldA(m, k, A, τ), rt))
        end
    end
    return like
end


"""
    simulate(model::AbstractRDM)

Returns a matrix containing evidence samples of the racing diffusion model decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractRDM`: an racing diffusion model object

# Keywords

- `Δt=.001`: size of time step of decision process in seconds
"""
function simulate(rng::AbstractRNG, model::AbstractRDM; Δt = 0.001)
    (; ν, A, k) = model
    n = length(model.ν)
    t = 0.0
    z = rand(rng, Uniform(0, A), n)
    α = k + A
    x = z
    ϵ = fill(0.0, n)
    evidence = [deepcopy(x)]
    time_steps = [t]
    while all(x .< α)
        t += Δt
        increment!(rng, model, x, ϵ, ν, Δt)
        push!(evidence, deepcopy(x))
        push!(time_steps, t)
    end
    return time_steps, reduce(vcat, transpose.(evidence))
end

function increment!(rng::AbstractRNG, model::AbstractRDM, x, ϵ, ν, Δt)
    ϵ .= rand(rng, Normal(0.0, 1.0), length(ν))
    x .+= ν * Δt + ϵ * √(Δt)
    return nothing
end
