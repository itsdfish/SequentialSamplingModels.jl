"""
    MDFT{T <: Real} <: AbstractMDFT

A model type for Multialternative Decision Field Theory. 
    
# Parameters 
- `σ = 1.0`: diffusion noise 
- `α = 1.5`: evidence threshold 
- `τ = .30`: non-decision time
- `w::Vector{T}`: attention weights vector where each element corresponds to the attention given to the corresponding dimension
- `S::Array{T, 2}`: feedback matrix
- `C::Array{T, 2}`: contrast matrix

# Constructors 

    MDFT(σ, α, τ, w, S, C)

    MDFT(σ, α, τ, w, S, C = default_contrast_matrix(S))
        
# Example 

```julia 
using SequentialSamplingModels 
ν = [2.5,2.0]
α = 1.5
β = 0.20
λ = 0.10 
σ = 1.0
τ = 0.30

dist = MDFT(; ν, α, β, λ, τ, σ)
choices,rts = rand(dist, 500)
```
# References

Roe, Robert M., Jermone R. Busemeyer, and James T. Townsend. "Multialternative decision field theory: A dynamic connectionst model of decision making." Psychological review 108.2 (2001): 370.
"""
mutable struct MDFT{T <: Real} <: AbstractMDFT
    σ::T
    α::T
    τ::T
    w::Vector{T}
    S::Array{T, 2}
    C::Array{T, 2}
end

function MDFT(σ, α, τ, w, S, C)
    σ, α, τ, _, _, _ = promote(σ, α, τ, w[1], S[1], C[1])
    w = convert(Vector{typeof(τ)}, w)
    S = convert(Array{typeof(τ), 2}, S)
    C = convert(Array{typeof(τ), 2}, C)
    return MDFT(σ, α, τ, w, S, C)
end

function MDFT(; σ = 1.0, α, τ, w, S, C = default_contrast_matrix(S))
    return MDFT(σ, α, τ, w, S, C)
end

function params(d::AbstractMDFT)
    return (d.σ, d.α, d.τ)
end

get_pdf_type(d::AbstractMDFT) = Approximate

"""
    rand(dist::AbstractMDFT, n_sim::Int; Δt = 0.001)

Generate `n_sim` random choice-rt pairs for the Multiattribute Decision Field Theory (MDFT).

# Arguments

- `dist`: model object for the Multiattribute Decision Field Theory (MDFT).
- `n_sim::Int`: the number of simulated choice-rt pairs 

# Keywords
- `Δt = 0.001`: time step size
"""
function rand(
    rng::AbstractRNG,
    dist::AbstractMDFT,
    n_sim::Int,
    M::AbstractArray;
    Δt = 0.001
)
    n_options = size(M, 1)
    x = fill(0.0, n_options)
    Δμ = fill(0.0, n_options)
    ϵ = fill(0.0, n_options)
    choices = fill(0, n_sim)
    rts = fill(0.0, n_sim)
    CM = dist.C * M
    for i ∈ 1:n_sim
        choices[i], rts[i] = _rand(rng, dist, x, Δμ, ϵ, CM; Δt)
        x .= 0.0
    end
    return (; choices, rts)
end

rand(dist::AbstractMDFT, n_sim::Int, M::AbstractArray; Δt = 0.001) =
    rand(Random.default_rng(), dist, n_sim, M; Δt)

"""
    rand(dist::AbstractMDFT; Δt = 0.001)

Generate a random choice-rt pair for the Multiattribute Decision Field Theory (MDFT).

# Arguments
- `dist`: model object for the Multiattribute Decision Field Theory (MDFT). 
- `Δt = 0.001`: time step size 
"""
function rand(rng::AbstractRNG, dist::AbstractMDFT, M::AbstractArray; Δt = 0.001)
    n_options = size(M, 1)
    # evidence for each alternative
    x = fill(0.0, n_options)
    # mean change in evidence for each alternative
    Δμ = fill(0.0, n_options)
    # noise for each alternative 
    ϵ = fill(0.0, n_options)
    # precompute matric multiplication
    CM = dist.C * M
    return _rand(rng, dist, x, Δμ, ϵ, CM; Δt)
end

rand(dist::AbstractMDFT, M::AbstractArray; Δt = 0.001) =
    rand(Random.default_rng(), dist, M; Δt)

function _rand(rng::AbstractRNG, dist::AbstractMDFT, x, Δμ, ϵ, CM; Δt = 0.001)
    (; α, τ) = dist
    t = 0.0
    iter = 1
    while all(x .< α)
        increment!(rng, dist, x, Δμ, ϵ, CM; Δt)
        iter += 1
        t += Δt
    end
    _, choice = findmax(x)
    rt = t + τ
    return (; choice, rt)
end

increment!(dist::AbstractMDFT, x, Δμ, ϵ, CM) =
    increment!(Random.default_rng(), dist, x, Δμ, ϵ, CM)

function increment!(rng::AbstractRNG, dist::AbstractMDFT, x, Δμ, ϵ, CM; Δt)
    (; σ, w, S, C) = dist
    n_options, n_attributes = size(CM)
    att_idx = sample(1:n_attributes, Weights(w))
    v = @view CM[:, att_idx]
    compute_mean_evidence!(dist, x, Δμ, v)
    ϵ .= rand(rng, Normal(0, σ), n_options)
    # mean evidence plus noise 
    x .= Δμ * Δt .+ C * ϵ * sqrt(Δt)
    return x
end

function compute_mean_evidence!(dist::AbstractMDFT, x, Δμ, v)
    (; S) = dist
    Δμ .= S * x .+ v
    return nothing
end

"""
    simulate(model::AbstractMDFT; _...)

Returns a matrix containing evidence samples of the Multialternative Decision Field Theory (MDFT) decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::AbstractMDFT`: an MDFT model object
- `M`: value matrix 
"""
function simulate(model::AbstractMDFT, M; Δt = 0.001, _...)
    (; α) = model
    n_options = size(M, 1)
    CM = model.C * M
    x = fill(0.0, n_options)
    μΔ = fill(0.0, n_options)
    ϵ = fill(0.0, n_options)
    t = 0.0
    evidence = [fill(0.0, n_options)]
    time_steps = [t]
    while all(x .< α)
        t += Δt
        increment!(model, x, μΔ, ϵ, CM; Δt)
        push!(evidence, copy(x))
        push!(time_steps, t)
    end
    return time_steps, reduce(vcat, transpose.(evidence))
end

function default_contrast_matrix(S::AbstractArray{T}) where {T}
    n = size(S, 1)
    C = Array{T, 2}(undef, n, n)
    C .= -1 / (n - 1)
    for r ∈ 1:n
        C[r, r] = 1.0
    end
    return C
end
