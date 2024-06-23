
"""
    MDFT{T <: Real} <: AbstractMDFT

A model type for simulating Multiattribute Decision Field Theory (MDFT) as an Stochastic Differential Equation (SDE). 
    
# Parameters 
- `σ = 1.0`: diffusion noise 
- `α = 15.0`: evidence threshold 
- `τ = .30`: non-decision time
- `w::Vector{T}`: attention weights vector where each element corresponds to the attention given to the corresponding dimension
- `S::Array{T, 2}`: feedback matrix allowing self-connections and interconnections between alternatives. Self-connections range from zero to 1, where s_ij < 1 represents decay. Interconnections 
     between options i and j  where i ≠ j are inhibatory if s_ij < 0.
- `C::Array{T, 2}`: contrast weight matrix where c_ij is the contrast weight when comparing options i and j.

# Constructors 

    MDFT(σ, α, τ, w, S, C)

    MDFT(σ, α, τ, w, S, C = make_default_contrast(S))
        
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

model = MDFT(;
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

Evans, N. J., Holmes, W. R., & Trueblood, J. S. (2019). Response-time data provide critical constraints on dynamic models of multi-alternative, multi-attribute choice. Psychonomic Bulletin & Review, 26, 901-933.

Hotaling, J. M., Busemeyer, J. R., & Li, J. (2010). Theoretical developments in decision
field theory: Comment on tsetsos, usher, and chater (2010). Psychological Review, 117 , 1294-1298.

Roe, Robert M., Jermone R. Busemeyer, and James T. Townsend. "Multiattribute Decision Field Theory: A dynamic connectionst model of decision making." Psychological review 108.2 (2001): 370.
"""
mutable struct MDFT{T <: Real} <: AbstractMDFT
    σ::T
    α::T
    τ::T
    γ::T
    κ::Vector{T}
    ϕ1::T
    ϕ2::T
    β::T
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

function MDFT(; σ = 1.0, α = 15.0, τ = 10.0, w, S, C = make_default_contrast(S))
    return MDFT(σ, α, τ, w, S, C)
end

function params(d::MDFT)
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
    dist::MDFT,
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

rand(dist::MDFT, M::AbstractArray; Δt = 0.001) = rand(Random.default_rng(), dist, M; Δt)

rand(dist::MDFT, n_sim::Int, M::AbstractArray; Δt = 0.001) =
    rand(Random.default_rng(), dist, n_sim, M; Δt)

function rand(rng::AbstractRNG, dist::MDFT, M::AbstractArray; Δt = 0.001)
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

function _rand(rng::AbstractRNG, dist::MDFT, x, Δμ, ϵ, CM; Δt = 0.001)
    (; α, τ) = dist
    t = 0.0
    att_idx = rand(1:2)
    while all(x .< α)
        increment!(rng, dist, x, Δμ, ϵ, CM, att_idx; Δt)
        t += Δt
    end
    _, choice = findmax(x)
    rt = t + τ
    return (; choice, rt)
end

function _rand(rng::AbstractRNG, dist::MDFT, x, Δμ, ϵ, CM, att_idx; Δt = 0.001)
    (; α, τ) = dist
    t = 0.0
    while all(x .< α)
        increment!(rng, dist, x, Δμ, ϵ, CM, att_idx; Δt)
        t += 1
    end
    _, choice = findmax(x)
    rt = t + τ
    return (; choice, rt)
end

function increment!(rng::AbstractRNG, dist::MDFT, x, Δμ, ϵ, CM, att_idx; Δt)
    (; σ) = dist
    n_options, n_attributes = size(CM)
    att_idx = update_attention(dist, att_idx; Δt)
    v = @view CM[:, att_idx]
    compute_mean_evidence!(dist, x, Δμ, v)
    ϵ .= rand(rng, Normal(0, σ), n_options)
    x .+= Δμ * Δt .+ ϵ * √Δt
    return x
end

function make_default_contrast(S::AbstractArray{T}) where {T}
    n = size(S, 1)
    C = Array{T, 2}(undef, n, n)
    C .= -1 / (n - 1)
    for r ∈ 1:n
        C[r, r] = 1.0
    end
    return C
end

function update_attention(dist::MDFT, att_idx; Δt)
    (;κ) = model 
    if rand() ≤ prob_switch(κ[att_idx], Δt)
        return att_idx == 1 ? 2 : 1
    end 
    return att_idx
end

"""

# References 

Hotaling, J. M., Busemeyer, J. R., & Li, J. (2010). Theoretical developments in decision
field theory: Comment on tsetsos, usher, and chater (2010). Psychological Review, 117 , 1294-1298.
"""
function compute_distances(dist::MDFT, M)
    (; β) = model 
    # number of alternatives
    n = size(M, 1)
    D = fill(0.0, n, n)
    for i ∈ 1:n
        for j ∈ (i+1):n
             Δ1, Δ2 = (M[i,1] - M[j,1]) / √(2), (M[i,2] - M[j,2]) / √(2)
             D[i,j] = D[j,i] = (Δ2 - Δ1)^2 + β * (Δ2 + Δ1)^2
        end
    end
    return D
end

"""

# References 

Hotaling, J. M., Busemeyer, J. R., & Li, J. (2010). Theoretical developments in decision
field theory: Comment on tsetsos, usher, and chater (2010). Psychological Review, 117 , 1294-1298.
"""
function compute_feedback_matrix(dist::MDFT, D)
    (;ϕ1,ϕ2) = dist 
    n = size(D, 1)
    S = fill(0.0, n, n)
    for i ∈ 1:n
        for j ∈ (i+1):n
            δ = i == j ? 1 : 0
            S[i,j] = S[j,i] = δ - ϕ2 * exp(-ϕ1 * D[i,j]^2)
        end
    end
    return S
end

prob_switch(κ, Δt) = 1 - exp(-κ * Δt)
# prob_switch(1, 0.01)

# function prob_switch(Q::Matrix, t)
#     return exp(Q * t)
# end
