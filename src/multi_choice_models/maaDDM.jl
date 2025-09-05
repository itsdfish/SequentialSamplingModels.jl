"""
    maaDDM{T<:Real} <: AbstractaDDM

An object for the multi-attribute attentional drift diffusion model. 

# Constructors

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    maaDDM(ν, σ, Δ, θ, ϕ, ω, α, z, τ)

The second constructor uses keywords with default values, and is not order dependent: 

    maaDDM(;
        ν = [4.0 5.0; 5.0 4.0],
        α = 1.0,
        z = 0.0,
        θ = 0.3,
        ϕ = 0.50,
        ω = 0.70,
        σ = 0.02,
        Δ = 0.0004,
        τ = 0.0,
    )
            
In this version of the model, the non-attended attribute of the non-attended alternative is doubly discounted. For example,
the mean drift rate for the attribute 1 of alternative 1 is given by:

```julia
    Δ * (ω * (ν[1,1] - θ * ν[2,1]) + (1 - ω) * ϕ * (ν[1,2] - θ * ν[2,2]))
```

# Keywords 

- `ν::T`: drift rates where rows are alternatives and columns are attributes. ν ∈ ℝⁿᵐ.
- `σ::T`: standard deviation of noise in evidence accumulation. σ ∈ ℝ⁺.
- `Δ::T`: constant of evidence accumulation speed (evidence per ms). Δ ∈ ℝ⁺.
- `θ::T`: bias away from unattended alternative (lower indicates more bias). θ ∈ [0,1].
- `ϕ::T`: bias away from unattended attribute. ϕ ∈ [0,1]. 
- `ω::T`: attribute weight. ω ∈ [0,1].
- `α::T`: evidence threshold. α ∈ ℝ⁺. 
- `z::T`: initial evidence. z ∈ [0, α]
- `τ::T`: non-decision time. τ ∈ [0, min_rt].

# Example 

```julia 
using SequentialSamplingModels
using StatsBase

mutable struct Transition
    state::Int 
    n::Int
    mat::Array{Float64,2} 
 end

 function Transition(mat)
    n = size(mat,1)
    state = rand(1:n)
    return Transition(state, n, mat)
 end
 
 function attend(transition)
     (;mat,n,state) = transition
     w = @view mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end

ν = [4.0 5.0; 5.0 4.0]
α = 1.0 
z = 0.0
θ = .3
ϕ = .50
ω = .70
σ = .02
Δ = .0004
τ = 0.0

dist = maaDDM(; ν, σ, Δ, θ, ϕ, ω, α, z, τ)

tmat = Transition([.98 .015 .0025 .0025;
                .015 .98 .0025 .0025;
                .0025 .0025 .98 .015;
                .0025 .0025 .015 .98])

 choices,rts = rand(dist, 100, attend, tmat)
```

# References 

Yang, X., & Krajbich, I. (2023). A dynamic computational model of gaze and choice in multi-attribute decisions. 
Psychological Review, 130(1), 52.
"""
struct maaDDM{T <: Real} <: AbstractaDDM
    ν::Array{T, 2}
    σ::T
    Δ::T
    θ::T
    ϕ::T
    ω::T
    α::T
    z::T
    τ::T
    function maaDDM(
        ν::Array{T, 2},
        σ::T,
        Δ::T,
        θ::T,
        ϕ::T,
        ω::T,
        α::T,
        z::T,
        τ::T
    ) where {T <: Real}
        @argcheck σ ≥ 0
        @argcheck Δ ≥ 0
        @argcheck (θ ≥ 0) && (θ ≤ 1)
        @argcheck (ϕ ≥ 0) && (ϕ ≤ 1)
        @argcheck (ω ≥ 0) && (ω ≤ 1)
        @argcheck α ≥ 0
        @argcheck (z ≥ 0) && (z ≤ α)
        @argcheck τ ≥ 0
        return new{T}(ν, σ, Δ, θ, ϕ, ω, α, z, τ)
    end
end

function maaDDM(ν, σ, Δ, θ, ϕ, ω, α, z, τ)
    _, σ, Δ, θ, ϕ, ω, α, z, τ = promote(ν[1], σ, Δ, θ, ϕ, ω, α, z, τ)
    ν = convert(Array{typeof(z), 2}, ν)
    return maaDDM(ν, σ, Δ, θ, ϕ, ω, α, z, τ)
end

function maaDDM(;
    ν = [4.0 5.0; 5.0 4.0],
    α = 1.0,
    z = 0.0,
    θ = 0.3,
    ϕ = 0.50,
    ω = 0.70,
    σ = 0.02,
    Δ = 0.0004,
    τ = 0.0
)
    return maaDDM(ν, σ, Δ, θ, ϕ, ω, α, z, τ)
end

params(d::maaDDM) = (d.ν, d.σ, d.Δ, d.θ, d.ϕ, d.ω, d.α, d.z, d.τ)

get_pdf_type(d::maaDDM) = Approximate

n_options(d::maaDDM) = size(d.ν, 1)

"""
    increment!(rng, dist::maaDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `dist::maaDDM`: a model object for the multiattribute attentional drift diffusion model
- `location`: an index for fixation location 
"""
function increment!(rng, dist::maaDDM, location)
    (; ν, θ, ϕ, ω, Δ, σ) = dist
    # option 1, attribute 1
    if location == 1
        return Δ * (ω * (ν[1, 1] - θ * ν[2, 1]) + (1 - ω) * ϕ * (ν[1, 2] - θ * ν[2, 2])) +
               noise(rng, σ)
        # option 1, attribute 2
    elseif location == 2
        return Δ * (ϕ * ω * (ν[1, 1] - θ * ν[2, 1]) + (1 - ω) * (ν[1, 2] - θ * ν[2, 2])) +
               noise(rng, σ)
        # option 2, attribute 1
    elseif location == 3
        return Δ * (ω * (θ * ν[1, 1] - ν[2, 1]) + (1 - ω) * ϕ * (θ * ν[1, 2] - ν[2, 2])) +
               noise(rng, σ)
        # option 2, attribute 2
    elseif location == 4
        return Δ * (ϕ * ω * (θ * ν[1, 1] - ν[2, 1]) + (1 - ω) * (θ * ν[1, 2] - ν[2, 2])) +
               noise(rng, σ)
    end
    @argcheck location ∈ [1, 2, 3, 4]
    return -100.0
end
