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

function rand(rng::AbstractRNG, dist::AbstractCDDM)
    (;ν,η,σ,α,τ,zτ) = dist

    return (;choice,rt)
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
    solve_zeros(order, n_zeros; ϵ = 1e-12, max_iter = 100)

Finds the input of a bassel function which evaluate to zero. Assumes the 
bassel function is of the first kind.

# Arguments 

- `order`: order of the bassel function  
- `n_zeros`: number of solutions to return 

# Keywords 

- `ϵ = 1e-12`: error tolerance 
- `max_iter`: maximum iterations for finding solution 
"""
function solve_zeros(order, n_zeros; ϵ = 1e-12, max_iter = 100)
    k3 = n_zeros * 3
    solutions = fill(0.0, k3)
    for j ∈ 1:k3
        x0 = 1 + √(2) + (j-1) * π + order + order^0.4
        solutions[j] = find_zero(order, x0; ϵ, max_iter)
    end 
    sort!(solutions)
    diffs = pushfirst!(succ_diffs(solutions), 1.0)
    solutions = solutions[diffs .> 1e-8]
    return solutions[1:n_zeros]
end


function find_zero(n, x0; ϵ = 1e-12, max_iter=100)
    n1 = n + 1
    n2 = n^2
    
    err = 1.0
    i = 1
    x = 0.0

    while (abs(err) > ϵ) && (i < max_iter)
        a = besselj(n, x0)
        b = besselj(n1, x0)
        x02 = x0^2
        err = 2 * a * x0 * (n * a - b * x0) / 
            (2 * b^2 * x02 - a * b * x0 * (4 * n + 1) + (n * n1 + x02) * a^2)
        x = x0 - err
        x0 = x
        i += 1
    end
    return x
end

function succ_diffs(x)
    n = length(x)
    y = zeros(n-1)
    for i ∈ 2:n
        y[i-1] = x[i] - x[i-1]
    end
    return y
end

function logpdf(d::AbstractCDDM, r::Int, t::Float64)
    (;ν,η,σ,α,τ,zτ) = d
end

function bassel_hm(d::AbstractCDDM, rt ;k_max = 50)
    rt == 0 ? (return 0.0) : nothing 
    (;σ,α) = d
    x = 0.0
    j0 = solve_zeros(0, k_max)
    α² = α^2
    σ² = σ^2
    s = σ² / (2 * π * α²)

    for k ∈ 1:k_max
        x +=  (j0[k] / besselj(1, j0[k])) * exp(-((j0[k]^2 * σ²) / (2 * α²)) * rt)
    end
    return s * x
end

function bassel_s(d::AbstractCDDM, rt; h = 2.5 / 300, v = 0, ϵ = 1e-12)
    rt == 0 ? (return 0.0) : nothing 
    (;σ,α) = d
    x = 0.0
    j0 = solve_zeros(0, 1)
    s = (α / σ)^2
    t = round(rt / h) * (h / s)
    # println("t $t")
    x1 = ((1 - ϵ) * (1 + t)^(v + 2)) / ((ϵ + t)^(v + 0.5) * t^(3/2))
    x2 = exp(-((1 - ϵ)^2) / (2 * t) - .50 * j0[1]^2 * t)
    # println("x1 $x1")
    # println("x2 $x2")
    return x1 * x2 / s
end