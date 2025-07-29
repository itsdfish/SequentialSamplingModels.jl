
"""
    ClassicMDFT{T <: Real} <: AbstractMDFT

A model type for Multiattribute Decision Field Theory. 
    
# Parameters 
- `σ::T = 1.0`: diffusion noise. σ ∈ ℝ⁺. 
- `C::Array{T, 2}`: contrast weight matrix where c_ij is the contrast weight when comparing options i and j.
- `S::Array{T, 2}`: feedback matrix allowing self-connections and interconnections between alternatives. Self-connections range from zero to 1, where s_ij < 1 represents decay. Interconnections 
     between options i and j  where i ≠ j are inhibatory if s_ij < 0.
- `w::Vector{T}`: attention weights vector where each element corresponds to the attention given to the corresponding dimension. wᵢ ∈ [0,1], ∑wᵢ = 1.
- `α::T = 15.0`: evidence threshold. α  ∈ ℝ⁺.
- `τ::T = .30`: non-decision time. τ ∈ [0, min_rt].

# Constructors 

    ClassicMDFT(σ, C, S, w, α, τ)

    ClassicMDFT(;
        σ = 1.0,
        α = 15.0,
        τ = 10.0,
        w,
        S,
        C = make_default_contrast(size(S, 1))
    )
        
# Example 

An example of the similarity effect. When choosing between options 1 and 2, the model predicts equal preference 
because the options fall along the diagonal of attribute space, signifying a 1 to 1 trade-off of equally weighted 
attributes. Option 3 is introduced to the choice set, which is similar to (and competitive with) option 1 and disimilar to option 2.
In this case, the model predicts an increase the choice probability for option 2 relative to option 1.
```julia 
# value matrix where rows correspond to alternatives, and columns correspond to attributes
M = [
    1.0 3.0
    3.0 1.0
    0.9 3.1
]

model = ClassicMDFT(;
    # diffusion noise 
    σ = 1.0,
    # attribute attention weights 
    w = [0.5, 0.5],
    # feedback matrix 
    S = [
        0.9500000 -0.0122316 -0.04999996
        -0.0122316 0.9500000 -0.00903030
        -0.0499996 -0.0090303 0.95000000
    ],
    # decision threshold
    α = 17.5,
    # non-decision time 
    τ = 0.300,
)
choices, rts = rand(model, 10_000, M; Δt = 1.0)
map(c -> mean(choices .== c), 1:3)
```
# References

Roe, Robert M., Jermone R. Busemeyer, and James T. Townsend. "Multiattribute Decision Field Theory: A dynamic connectionst model of decision making." Psychological review 108.2 (2001): 370.
"""
mutable struct ClassicMDFT{T <: Real} <: AbstractMDFT
    σ::T
    C::Array{T, 2}
    S::Array{T, 2}
    w::Vector{T}
    α::T
    τ::T
    function ClassicMDFT(
        σ::T,
        C::Array{T, 2},
        S::Array{T, 2},
        w::Vector{T},
        α::T,
        τ::T
    ) where {T <: Real}
        @argcheck σ ≥ 0
        @argcheck α ≥ 0
        @argcheck τ ≥ 0
        @argcheck all(w .≥ 0) && (sum(w) == 1)
        return new{T}(σ, C, S, w, α, τ)
    end
end

function ClassicMDFT(σ, C, S, w, α, τ)
    σ, _, _, _, α, τ, = promote(σ, C[1], S[1], w[1], α, τ)
    C = convert(Array{typeof(τ), 2}, C)
    S = convert(Array{typeof(τ), 2}, S)
    w = convert(Vector{typeof(τ)}, w)
    return ClassicMDFT(σ, C, S, w, α, τ)
end

function ClassicMDFT(;
    σ = 1.0,
    α = 15.0,
    τ = 10.0,
    w,
    S,
    C = make_default_contrast(size(S, 1))
)
    return ClassicMDFT(σ, C, S, w, α, τ)
end

get_pdf_type(::AbstractMDFT) = Approximate

params(d::ClassicMDFT) = (d.σ, d.C, d.S, d.w, d.α, d.τ)

"""
    rand(
        rng::AbstractRNG,
        dist::AbstractMDFT,
        n_sim::Int,
        M::AbstractArray;
        Δt = 0.001
    )

Generate `n_sim` random choice-rt pairs for the Multiattribute Decision Field Theory (MDFT).

# Arguments

- `rng::AbstractRNG`: a random number generator which is a subtype of `AbstractRNG`
- `dist::AbstractMDFT`: model object for the Multiattribute Decision Field Theory (MDFT).
- `n_sim::Int`: the number of simulated choice-rt pairs
- `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 

# Keywords

- `Δt = 0.001`: time step size
"""
function rand(
    rng::AbstractRNG,
    dist::ClassicMDFT,
    n_sim::Int,
    M::AbstractArray;
)
    n_alternatives = size(M, 1)
    x = fill(0.0, n_alternatives)
    Δμ = fill(0.0, n_alternatives)
    choice = fill(0, n_sim)
    rt = fill(0.0, n_sim)
    CM = dist.C * M
    for i ∈ 1:n_sim
        choice[i], rt[i] = _rand(rng, dist, x, Δμ, CM)
        x .= 0.0
    end
    return (; choice, rt)
end

rand(dist::ClassicMDFT, M::AbstractArray;) = rand(Random.default_rng(), dist, M)

rand(dist::ClassicMDFT, n_sim::Int, M::AbstractArray) =
    rand(Random.default_rng(), dist, n_sim, M)

function rand(rng::AbstractRNG, dist::ClassicMDFT, M::AbstractArray)
    n_alternatives = size(M, 1)
    # evidence for each alternative
    x = fill(0.0, n_alternatives)
    # mean change in evidence for each alternative
    Δμ = fill(0.0, n_alternatives)
    # noise for each alternative 
    ϵ = fill(0.0, n_alternatives)
    # precompute matric multiplication
    CM = dist.C * M
    return _rand(rng, dist, x, Δμ, CM)
end

function _rand(rng::AbstractRNG, dist::ClassicMDFT, x, Δμ, CM)
    (; α, τ) = dist
    t = 0.0
    while all(x .< α)
        increment!(rng, dist, x, Δμ, CM)
        t += 1
    end
    _, choice = findmax(x)
    rt = t + τ
    return (; choice, rt)
end

function increment!(rng::AbstractRNG, dist::ClassicMDFT, x, Δμ, CM)
    (; σ, w, S, C) = dist
    n_alternatives, n_attributes = size(CM)
    att_idx = sample(1:n_attributes, Weights(w))
    v = @view CM[:, att_idx]
    compute_mean_evidence!(dist, x, Δμ, v)
    x .= Δμ .+ C * rand(rng, Normal(0, σ), n_alternatives)
    return nothing
end

function compute_mean_evidence!(dist::AbstractMDFT, x, Δμ, v)
    (; S) = dist
    Δμ .= S * x .+ v
    return nothing
end
