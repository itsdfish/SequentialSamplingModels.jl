
"""
    ClassicMDFT{T <: Real} <: AbstractMDFT

A model type for Multiattribute Decision Field Theory. 
    
# Parameters 
- `σ = 1.0`: diffusion noise 
- `α = 15.0`: evidence threshold 
- `τ = .30`: non-decision time
- `w::Vector{T}`: attention weights vector where each element corresponds to the attention given to the corresponding dimension
- `S::Array{T, 2}`: feedback matrix allowing self-connections and interconnections between alternatives. Self-connections range from zero to 1, where s_ij < 1 represents decay. Interconnections 
     between options i and j  where i ≠ j are inhibatory if s_ij < 0.
- `C::Array{T, 2}`: contrast weight matrix where c_ij is the contrast weight when comparing options i and j.

# Constructors 

    ClassicMDFT(σ, α, τ, w, S, C)

    ClassicMDFT(σ, α, τ, w, S, C = make_default_contrast(S))
        
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
    # non-decision time 
    τ = 0.300,
    # diffusion noise 
    σ = 1.0,
    # decision threshold
    α = 17.5,
    # attribute attention weights 
    w = [0.5, 0.5],
    # feedback matrix 
    S = [
        0.9500000 -0.0122316 -0.04999996
        -0.0122316 0.9500000 -0.00903030
        -0.0499996 -0.0090303 0.95000000
    ],
)
choices, rts = rand(model, 10_000, M; Δt = 1.0)
map(c -> mean(choices .== c), 1:3)
```
# References

Roe, Robert M., Jermone R. Busemeyer, and James T. Townsend. "Multiattribute Decision Field Theory: A dynamic connectionst model of decision making." Psychological review 108.2 (2001): 370.
"""
mutable struct ClassicMDFT{T <: Real} <: AbstractMDFT
    σ::T
    α::T
    τ::T
    w::Vector{T}
    S::Array{T, 2}
    C::Array{T, 2}
end

function ClassicMDFT(σ, α, τ, w, S, C)
    σ, α, τ, _, _, _ = promote(σ, α, τ, w[1], S[1], C[1])
    w = convert(Vector{typeof(τ)}, w)
    S = convert(Array{typeof(τ), 2}, S)
    C = convert(Array{typeof(τ), 2}, C)
    return ClassicMDFT(σ, α, τ, w, S, C)
end

function ClassicMDFT(;
    σ = 1.0,
    α = 15.0,
    τ = 10.0,
    w,
    S,
    C = make_default_contrast(size(S, 1))
)
    return ClassicMDFT(σ, α, τ, w, S, C)
end

get_pdf_type(d::AbstractMDFT) = Approximate

function params(d::ClassicMDFT)
    return (d.σ, d.α, d.τ, d.w, d.S, d.C)
end

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
    ϵ = fill(0.0, n_alternatives)
    choices = fill(0, n_sim)
    rts = fill(0.0, n_sim)
    CM = dist.C * M
    for i ∈ 1:n_sim
        choices[i], rts[i] = _rand(rng, dist, x, Δμ, ϵ, CM)
        x .= 0.0
    end
    return (; choices, rts)
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
    return _rand(rng, dist, x, Δμ, ϵ, CM)
end

function _rand(rng::AbstractRNG, dist::ClassicMDFT, x, Δμ, ϵ, CM)
    (; α, τ) = dist
    t = 0.0
    while all(x .< α)
        increment!(rng, dist, x, Δμ, ϵ, CM)
        t += 1
    end
    _, choice = findmax(x)
    rt = t + τ
    return (; choice, rt)
end

function increment!(rng::AbstractRNG, dist::ClassicMDFT, x, Δμ, ϵ, CM)
    (; σ, w, S, C) = dist
    n_alternatives, n_attributes = size(CM)
    att_idx = sample(1:n_attributes, Weights(w))
    v = @view CM[:, att_idx]
    compute_mean_evidence!(dist, x, Δμ, v)
    ϵ .= rand(rng, Normal(0, σ), n_alternatives)
    x .= Δμ .+ C * ϵ
    return nothing
end

function compute_mean_evidence!(dist::AbstractMDFT, x, Δμ, v)
    (; S) = dist
    Δμ .= S * x .+ v
    return nothing
end
