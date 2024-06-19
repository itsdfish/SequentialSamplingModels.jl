"""
    MDFT{T<:Real} <: Simulator

A model type for Multialternative Decision Field Theory. 
    
# Parameters 

- `ν = [2.5,2.0]`: drift rates 
- `α = 1.5`: evidence threshold 
- `β = .20`: lateral inhabition 
- `λ = .10`: leak rate
- `τ = .30`: non-decision time 
- `σ = 1.0`: diffusion noise 
- `Δt = .001`: time step 

# Constructors 

    MDFT(ν, α, β, λ, τ, σ, Δt)

    MDFT(;ν = [2.5,2.0], 
        α = 1.5, 
        β = .20, 
        λ = .10, 
        τ = .30, 
        σ = 1.0, 
        Δt = .001)
        
# Example 

```julia 
using SequentialSamplingModels 
ν = [2.5,2.0]
α = 1.5
β = 0.20
λ = 0.10 
σ = 1.0
τ = 0.30
Δt = .001

dist = MDFT(; ν, α, β, λ, τ, σ, Δt)
choices,rts = rand(dist, 500)
```
# References

Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. Psychological Review, 108 3, 550–592. https://doi.org/10.1037/0033-295X.108.3.550
"""
mutable struct MDFT{T <: Real, T1} <: AbstractMDFT
    ν::Vector{T}
    S::Array{T, 2}
    α::T
    τ::T
    σ::T
    Δt::T
    M::T1
end

function MDFT(ν, S, α, τ, σ, Δt, M)
    _, _, α, τ, σ, Δt, = promote(ν[1], S[1], α, τ, σ, Δt, Δt)
    S = convert(Array{typeof(τ), 2}, S)
    ν = convert(Array{typeof(τ), 1}, ν)
    return MDFT(ν, S, α, τ, σ, Δt, M)
end

function MDFT(; M, W, C = default_contrast_matrix(M), S, α, τ = 0.30, σ = 1.0, Δt = 0.001)
    ν = C * M * W
    return MDFT(ν, S, α, τ, σ, Δt, M)
end

function default_contrast_matrix(m)
    n = size(m, 1)
    C = fill(-1 / (n - 1), n, n)
    for r ∈ 1:n
        C[r, r] = 1.0
    end
    return C
end

increment!(dist::AbstractMDFT, x, Δμ, ϵ) = increment!(Random.default_rng(), dist, x, Δμ, ϵ)

function increment!(rng::AbstractRNG, dist::AbstractMDFT, x, Δμ, ϵ)
    (; ν, S, σ, Δt, M) = dist
    n = length(ν)
    #W = fill(.5, 2)
    idx = rand(1:2)
    weight = idx == 1 ? [1.0, 0.0] : [0.0, 1.0]
    C = default_contrast_matrix(M)
    v = C * M * weight
    ϵ .= rand(rng, Normal(0, σ), n)
    # add mean change in evidence plus noise 
    x .+= (S * x .+ v) * Δt .+ ϵ * sqrt(Δt)
    return x
end

# function increment!(rng::AbstractRNG, dist::AbstractMDFT, x, Δμ, ϵ)
#     (;ν, S, σ, Δt) = dist
#     n = length(ν)
#     # compute change of mean evidence
#     compute_mean_evidence!(ν, S, x, Δμ)
#     # sample noise 
#     ϵ .= rand(rng, Normal(0, σ), n)
#     # add mean change in evidence plus noise 
#     x .+= Δμ * Δt .+ ϵ * √(Δt)
#     return x
# end

function compute_mean_evidence!(ν, S, x, Δμ)
    Δμ .= ν .+ S * x
end
