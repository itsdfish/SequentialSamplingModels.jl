abstract type AbstractCDDM <: ContinuousSSM2D end 
"""
    CDDM{T<:Real} <: AbstractCDDM

A circular drift diffusion model (CDDM) for continous responding. CCDM is typically applied to continous report of color in visual
working memory tasks. Currently supports the 2D case. 

# Parameters 
- `ν`: a vector drift rates. ν₁ is the mean drift rate along the x-axis; ν₂ is the mean drift rate along the y-axis.
- `η`: a vector across-trial standard deviations of  drift rates. η₁ is the standard deviation of drift rate along the x-axis; 
    ν₂ is the standard deviation of drift rate along the y-axis
- `σ`: intra-trial drift rate variability 
- `α`: response boundary as measured by the radious of a circle 
- `τ`: mean non-decision time 

# Constructors

    CDDM(ν, η, σ, α, τ)

    CDDMν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.300) 

# Example

```julia
using SequentialSamplingModels
dist = CDDM(;ν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.30)
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
end

function CDDM(ν, η, σ, α, τ)
    _, _, σ, α, τ = promote(ν[1], η[1], σ, α, τ)
    ν = convert(Vector{typeof(τ)}, ν)
    η = convert(Vector{typeof(τ)}, η)
    return CDDM(ν, η, σ, α, τ)
end

function params(d::AbstractCDDM)
    return (d.ν, d.η, d.σ, d.α, d.τ)    
end

function CDDM(;ν=[1,.5], η=[1,1], σ=1, α=1.5, τ=0.300) 
    return CDDM(ν, η, σ, α, τ)
end

function rand(model::AbstractCDDM; Δt=.001)
    (;ν,η,σ,α,τ) = model
    # start position, distance, and time at 0
    x,y,r,t = zeros(4)
    _ν = @. rand(Normal(ν, η))
    𝒩 = Normal(0, σ)
    sqΔt = √(Δt)
    while r < α
        #step in x direction 
        x += _ν[1] * Δt + rand(𝒩) * sqΔt
        # step in y direction 
        y += _ν[2] * Δt + rand(𝒩) * sqΔt
        # distiance from starting point
        r = √(x^2 + y^2)
        # increment time 
        t += Δt
    end
    θ = atan(y, x)
    return [θ,t + τ]
end

function rand(d::AbstractCDDM, n::Int; Δt=.001)
    sim_data = zeros(n, 2)
    for r ∈ 1:n 
        sim_data[r,:] = rand(d; Δt=.001)
    end 
    return sim_data 
end

function logpdf(d::AbstractCDDM, r::Int, t::Float64)
    (;ν,η,σ,α,τ) = d

    return LL
end

function pdf(d::AbstractCDDM, data::Vector{<:Real}; k_max = 50)
    θ,rt = data 
    return max(0.0, pdf_term1(d, θ, rt) * pdf_term2(d, rt; k_max))
end

function pdf_term1(d::AbstractCDDM, θ::Real, rt::Real)
    (;ν,η,σ,α,τ) = d
    pos = (α * cos(θ), α * sin(θ))
    val = 1.0
    t = rt - τ
    _η = set_min(η)
    for i ∈ 1:length(ν)
        x0 = (_η[i] / σ)^2 
        x1 = 1 / √(t * x0 + 1)
        x2 = (-ν[i]^2) / (2 * _η[i]^2)
        x3 = (pos[i] * x0 + ν[i])^2
        x4 = (2 * _η[i]^2) * (x0 * t + 1)
        val *= x1 * exp(x2 + x3 / x4)
    end
    return val
end

function set_min(η)
    _η = similar(η)
    for i ∈ 1:length(η)
        _η[i] = η[i] == 0 ? .01 : η[i]
    end
    return _η
end

function pdf_term2(d::AbstractCDDM, rt::Real; k_max = 50)
    return bessel_hm(d, rt; k_max)
end

function pdf_rt(d::AbstractCDDM, rt::Real; n_steps = 50, kwargs...)
    Δθ = 2π / n_steps
    val = 0.0 
    for θ ∈ range(-2π, 2π, length=n_steps)
        val += pdf(d, [θ, rt]; kwargs...)
    end
    return val * Δθ
end

function pdf_angle(d::AbstractCDDM, θ::Real; n_steps = 50, kwargs...)
    Δt = (3 - d.τ) / n_steps
    val = 0.0 
    for t ∈ range(d.τ, 3, length=n_steps)
        val += pdf(d, [θ, t]; kwargs...)
    end
    return val * Δt
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
    (;ν,η,σ,α,τ) = model
    x,y,r,t = zeros(4)
    evidence = [zeros(2)]
    time_steps = [t]
    𝒩 = Normal(0, σ)
    _ν = @. rand(Normal(ν, η))
    sqΔt = √(Δt)
    while r < α
        x += _ν[1] * Δt + rand(𝒩) * sqΔt
        y += _ν[2] * Δt + rand(𝒩) * sqΔt
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
    (;ν,η,σ,α,τ) = d
end

function bessel_hm(d::AbstractCDDM, rt ;k_max = 50)
    rt == 0 ? (return 0.0) : nothing 
    (;σ,α,τ) = d
    x = 0.0
    t = rt - τ
    α² = α^2
    σ² = σ^2
    s = σ² / (2 * π * α²)

    for k ∈ 1:k_max
        j0k = besselj_zero(0, k)
        x += (j0k / besselj(1, j0k)) * exp(-((j0k^2 * σ²) / (2 * α²)) * t)
    end
    return s * x
end

function bessel_s(d::AbstractCDDM, rt; h = 2.5 / 300, v = 0, ϵ = 1e-12)
    rt == 0 ? (return 0.0) : nothing 
    (;σ,α) = d
    x = 0.0
    #  t = rt - τ
    j0 = besselj_zero(0, 1)
    s = (α / σ)^2
    t = round(rt / h) * (h / s)
    x1 = ((1 - ϵ) * (1 + t)^(v + 2)) / ((ϵ + t)^(v + 0.5) * t^(3/2))
    x2 = exp(-((1 - ϵ)^2) / (2 * t) - .50 * j0[1]^2 * t)
    return x1 * x2 / s
end