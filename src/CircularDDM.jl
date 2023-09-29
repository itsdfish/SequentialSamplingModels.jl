abstract type AbstractCDDM <: ContinuousSSM2D end 
"""
    CDDM{T<:Real} <: AbstractCDDM

A circular drift diffusion model (CDDM) for continous responding. CCDM is typically applied to continous report of color in visual
working memory tasks. Currently supports the 2D case. 

# Parameters 
ν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.300, zτ=0.100
- `ν`: a vector drift rates. ν₁ is the mean drift rate along the x-axis; ν₂ is the mean drift rate along the y-axis.
- `η`: a vector across-trial standard deviations of  drift rates. η₁ is the standard deviation of drift rate along the x-axis; 
    ν₂ is the standard deviation of drift rate along the y-axis
- `σ`: intra-trial drift rate variability 
- `α`: response boundary as measured by the radious of a circle 
- `τ`: mean non-decision time 
- `zτ`: range of non-decision time 

# Constructors

    CDDM(ν, η, σ, α, τ, zτ)

    CDDMν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.300, zτ=0.100) 

# Example

```julia
using SequentialSamplingModels
dist = CDDM(;ν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.300, zτ=0.100)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```

# References

Smith, P. L. (2016). Diffusion theory of decision making in continuous report. Psychological Review, 123(4), 425.

Smith, P. L., Garrett, P. M., & Zhou, J. (2023). Obtaining Stable Predicted Distributions of Response Times and Decision Outcomes for the Circular Diffusion Model. 
Computational Brain & Behavior, 1-13.
"""
struct CDDM{T<:Real} <: AbstractCDDM
    ν::Vector{T}
    η::Vector{T}
    σ::T
    α::T
    τ::T
    zτ::T
end

function CDDM(ν, η, σ, α, τ, zτ)
    _, _, τ = promote(ν[1], η[1], τ)
    ν = convert(Vector{typeof(τ)}, ν)
    η = convert(Vector{typeof(τ)}, η)
    return CDDM(ν, η, σ, α, τ, zτ)
end

function params(d::AbstractCDDM)
    return (d.ν, d.η, d.σ, d.α, d.τ, d.zτ)    
end

CDDM(;ν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.300, zτ=0.100) = CDDM(ν, η, σ, α, τ, zτ)

function rand(rng::AbstractRNG, d::AbstractCDDM; scale=.15)
    (;ν,η,σ,α,τ) = d
    ν₁ = rand(rng, Normal(ν[1], η[1]))
    ν₂ = rand(rng, Normal(ν[2], η[2]))
    μ = atan(ν₂, ν₁)
    κ = √(sum(ν.^2)) / σ
    x,y,r = zeros(3)
    iter = 0
    dist = VonMises(μ, κ)
    while r < α
        θstep = rand(rng, dist)
        x += cos(θstep)
        y += sin(θstep)
        r = √(x^2 + y^2)
        iter += 1
    end
    θpos = atan(y, x)
    rt = rand(rng, Gamma(iter, scale)) + τ
    θ = mod(θpos + 2π, 2π)
    return [θ,rt]
end

function rand(rng::AbstractRNG, d::AbstractCDDM, n::Int; scale = .15)
    sim_data = zeros(n, 2)
    for r ∈ 1:n 
        sim_data[r,:] = rand(rng, d; scale)
    end 
    return sim_data 
end

function rand1(d::AbstractCDDM, n::Int; Δt=.001)
    sim_data = zeros(n, 2)
    for r ∈ 1:n 
        sim_data[r,:] = rand1(d; Δt=.001)
    end 
    return sim_data 
end

function rand1(model::AbstractCDDM; Δt=.001)
    (;ν,η,σ,α,τ,zτ) = model
    # ν mean drift rate (x, y)
    # σ: diffusion parameter 
    # α: theshold (i.e., radius of circular threshold)
    # τ: non-decision time 

    # start position, distance, and time at 0
    x,y,r,t = zeros(4)
    𝒩 = Normal(0, σ)
    sqΔt = √(Δt)
    while r < α
        #step in x direction 
        x += ν[1] * Δt + rand(𝒩) * sqΔt
        # step in y direction 
        y += ν[2] * Δt + rand(𝒩) * sqΔt
        # distiance from starting point
        r = √(x^2 + y^2)
        # increment time 
        t += Δt
    end
    θ = atan(y, x)
    return [θ,t + τ]
end

function logpdf(d::AbstractCDDM, r::Int, t::Float64)
    (;ν,η,σ,α,τ,zτ) = d

    return LL
end

function pdf(d::AbstractCDDM, r::Int, t::Float64)
    (;ν,η,σ,τ,zτ) = d

    return density
end

"""
    simulate(model::AbstractCDDM; Δt=.001)

Returns a matrix containing evidence samples of the racing diffusion model decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractCDDM;`: a circular drift diffusion model object

# Keywords

- `Δt=.001`: size of time step of decision process in seconds
"""
function simulate(model::AbstractCDDM; Δt=.001)
    (;ν,η,σ,α,τ,zτ) = model
    x,y,r,t = zeros(4)
    evidence = [zeros(2)]
    time_steps = [t]
    𝒩 = Normal(0, σ)
    sqΔt = √(Δt)
    while r < α
        x += ν[1] * Δt + rand(𝒩) * sqΔt
        y += ν[2] * Δt + rand(𝒩) * sqΔt
        r = √(x^2 + y^2)
        t += Δt
        push!(time_steps, t)
        push!(evidence, [x,y])
    end
    return time_steps,reduce(vcat, transpose.(evidence))
end

# function increment!(model::AbstractRDM, x, ϵ, ν, Δt)
#     ϵ .= rand(Normal(0.0, 1.0), length(ν))
#     x .+= ν * Δt + ϵ * √(Δt)
#     return nothing 
# end

function logpdf(d::AbstractCDDM, r::Int, t::Float64)
    (;ν,η,σ,α,τ,zτ) = d
end

function bessel_hm(d::AbstractCDDM, rt ;k_max = 50)
    rt == 0 ? (return 0.0) : nothing 
    (;σ,α) = d
    x = 0.0
    α² = α^2
    σ² = σ^2
    s = σ² / (2 * π * α²)

    for k ∈ 1:k_max
        j0k = besselj_zero(0, k)
        x += (j0k / besselj(1, j0k)) * exp(-((j0k^2 * σ²) / (2 * α²)) * rt)
    end
    return s * x
end

function bessel_s(d::AbstractCDDM, rt; h = 2.5 / 300, v = 0, ϵ = 1e-12)
    rt == 0 ? (return 0.0) : nothing 
    (;σ,α) = d
    x = 0.0
    j0 = besselj_zero(0, 1)
    s = (α / σ)^2
    t = round(rt / h) * (h / s)
    x1 = ((1 - ϵ) * (1 + t)^(v + 2)) / ((ϵ + t)^(v + 0.5) * t^(3/2))
    x2 = exp(-((1 - ϵ)^2) / (2 * t) - .50 * j0[1]^2 * t)
    return x1 * x2 / s
end

function pdf_angle(d::AbstractCDDM, θ, rt)
    (;ν,η,σ,α,τ,zτ) = d
    t = rt - τ
    σ² = σ^2
    η₁²,η₂² = η.^2
    ν₁²,ν₂² = ν.^2
    G11 = (ν[1] * σ² + α * η₁² * cos(θ))^2
    G21 = (ν[2] * σ² + α * η₂² * sin(θ))^2
      
    Multiplier = σ²/(√(σ² + η₁² * t) * √(σ²+ η₂² * t))
    G12 = 2 * (η₁² * σ²) * (σ² + η₁² * t)
    G22 = 2 * (η₂² * σ²) * (σ² + η₂² * t)
    Girs1 = exp(G11 / G12 - ν[1]^2/(2 * η₁²))
    Girs2 = exp(G21 / G22 - ν[2]^2/(2 * η₂²))
    return Multiplier * Girs1 * Girs2
end