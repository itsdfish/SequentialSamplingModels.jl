"""
    MDFT{T <: Real} <: AbstractMDFT

A model type for simulating Multi-attribute Decision Field Theory (MDFT) as an Stochastic Differential Equation (SDE). 
    
# Parameters 
- `σ = 1.0`: diffusion noise 
- `α = 15.0`: evidence threshold 
- `τ = .30`: non-decision time
- `γ::T`: scales the valance, `CMW`, functioning like a drift rate
- `κ::Vector{T}`: exponential rate parameters for switching attention between attributes. Currently, limited to two 
    attributes 
- `ϕ1`: controls evidence leakage in the distance function for creating the feedback matrix, `S`
- `ϕ2`: controls the sensitivity of lateral inhibition to distance in the distance function for creating the feedback matrix, `S`
- `β`: controls the weight of the dominance dimension in the feedback matrix distance function. If `β` < 0, the indifference dimension 
    recieves more where. If `β` > 0, the dominance dimension recieves more weight
- `S::Array{T, 2}`: feedback matrix allowing self-connections and interconnections between alternatives. Self-connections range from zero to 1, where s_ij < 1 represents decay. Interconnections 
     between options i and j  where i ≠ j are inhibitory if s_ij < 0.
- `C::Array{T, 2}`: contrast weight matrix where c_ij is the contrast weight when comparing options i and j.

# Constructors 

    MDFT(σ, α, τ, γ, κ, ϕ1, ϕ2, β, C)

    MDFT(;
        n_alternatives,
        σ,
        α,
        τ,
        γ,
        κ,
        ϕ1,
        ϕ2,
        β,
        C = make_default_contrast(n_alternatives)
    )
        
# Example 

An example of the similarity effect. When choosing between options 1 and 2, the model predicts equal preference 
because the options fall along the diagonal of attribute space, signifying a 1 to 1 trade-off of equally weighted 
attributes. Option 3 is introduced to the choice set, which is similar to (and competitive with) option 1 and disimilar to option 2.
In this case, the model predicts an increase the choice probability for option 2 relative to option 1.
```julia 
using SequentialSamplingModels

model = MDFT(;
    n_alternatives = 3,
    σ = 0.1,
    α = .50,
    τ = 0.0,
    γ = 1.0,
    κ = [5.0, 5.0],
    ϕ1 = 0.01,
    ϕ2 = 0.10,
    β = 10.0
)
# value matrix where rows correspond to alternatives, and columns correspond to attributes
M = [
    1.0 3.0
    3.0 1.0
    0.9 3.1
]

choices, rts = rand(model, 10_000, M)
probs = map(c -> mean(choices .== c), 1:3)
```
# References

Evans, N. J., Holmes, W. R., & Trueblood, J. S. (2019). Response-time data provide critical constraints on dynamic models of multi-alternative, multi-attribute choice. Psychonomic Bulletin & Review, 26, 901-933.

Hotaling, J. M., Busemeyer, J. R., & Li, J. (2010). Theoretical developments in decision
field theory: Comment on tsetsos, usher, and chater (2010). Psychological Review, 117 , 1294-1298.

Roe, Robert M., Jermone R. Busemeyer, and James T. Townsend. "Multi-attribute Decision Field Theory: A dynamic connectionst model of decision making." Psychological review 108.2 (2001): 370.
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
    S::Array{T, 2}
    C::Array{T, 2}
    _CM::Array{T, 2}
    _att_idx::Int
end

function MDFT(σ, α, τ, γ, κ, ϕ1, ϕ2, β, C)
    σ, α, τ, γ, _, ϕ1, ϕ2, β, = promote(σ, α, τ, γ, κ[1], ϕ1, ϕ2, β)
    κ = convert(Vector{typeof(τ)}, κ)
    C = convert(Array{typeof(τ), 2}, C)
    _CM = zeros(size(C, 1), length(κ))
    S = similar(C)
    return MDFT(σ, α, τ, γ, κ, ϕ1, ϕ2, β, S, C, _CM, 0)
end

function MDFT(;
    n_alternatives,
    σ,
    α,
    τ,
    γ,
    κ,
    ϕ1,
    ϕ2,
    β,
    C = make_default_contrast(n_alternatives)
)
    return MDFT(σ, α, τ, γ, κ, ϕ1, ϕ2, β, C)
end

function params(d::MDFT)
    return (d.σ, d.α, d.τ, d.γ, d.κ, d.ϕ1, d.ϕ2, d.β, d.C)
end

n_options(d::AbstractMDFT) = size(d.C, 1)

"""
    rand(
        rng::AbstractRNG,
        dist::MDFT,
        n_sim::Int,
        M::AbstractArray;
        Δt = 0.001
    )

Generate `n_sim` random choice-rt pairs for the Multi-attribute Decision Field Theory (MDFT).

# Arguments

- `rng::AbstractRNG`: a random number generator which is a subtype of `AbstractRNG`
- `dist::MDFT`: model object for the Multi-attribute Decision Field Theory (MDFT).
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
    dist._CM = dist.C * M * dist.γ
    distances = compute_distances(dist, M)
    dist.S = compute_feedback_matrix(dist, distances)
    for i ∈ 1:n_sim
        choices[i], rts[i] = _rand(rng, dist, x, Δμ; Δt)
        x .= 0.0
    end
    return (; choices, rts)
end

rand(dist::MDFT, M::AbstractArray; Δt = 0.001) = rand(Random.default_rng(), dist, M; Δt)

rand(dist::MDFT, n_sim::Int, M::AbstractArray; Δt = 0.001) =
    rand(Random.default_rng(), dist, n_sim, M; Δt)

function rand(rng::AbstractRNG, dist::MDFT, M::AbstractArray; Δt = 0.001)
    (; _CM, C, γ) = dist
    n_options = size(M, 1)
    # evidence for each alternative
    x = fill(0.0, n_options)
    # mean change in evidence for each alternative
    Δμ = fill(0.0, n_options)
    # precompute matric multiplication
    _CM .= C * M * γ
    return _rand(rng, dist, x, Δμ; Δt)
end

function _rand(rng::AbstractRNG, dist::MDFT, x, Δμ; Δt = 0.001)
    (; α, τ) = dist
    t = 0.0
    dist._att_idx = rand(1:2)
    while all(x .< α)
        increment!(rng, dist, x, Δμ; Δt)
        t += Δt
    end
    _, choice = findmax(x)
    rt = t + τ
    return (; choice, rt)
end

"""
    increment!(rng::AbstractRNG, dist::MDFT, x, Δμ, ϵ; Δt)

Increments the preference states `x` on each time step. 

# Arguments

- `rng::AbstractRNG`: a random number generator which is a subtype of `AbstractRNG`
- `dist::AbstractMDFT`: model object for the Multi-attribute Decision Field Theory (MDFT).
- `x`: a vector of preference states 
- `Δμ`: a vector of mean change in the preference states 
- `ϵ`: a vector of normally distributed noise added to the preference states 

# Keywords

- `Δt = 0.001`: time step size
"""
function increment!(rng::AbstractRNG, dist::MDFT, x, Δμ; Δt)
    (; σ, _CM) = dist
    n_options = size(_CM, 1)
    att_idx = update_attention(dist; Δt)
    dist._att_idx = att_idx
    v = @view _CM[:, att_idx]
    compute_mean_evidence!(dist, x, Δμ, v)
    x .+= Δμ * Δt .+ rand(rng, Normal(0, dist.σ * √Δt), n_options)
    return nothing
end

increment!(dist::MDFT, x, Δμ; Δt) = increment!(Random.default_rng(), dist, x, Δμ; Δt)

"""
    make_default_contrast(n)

Creates an alternative × alternative contrast matrix representing comparisions between alternatives. 
The contrast has the following properties: 

1. The value of the diagonals are 1
2. The rows sum to 0 
3. The off diagonal values are equal 

# Arguments

- `n`: the number of alternatives in the `M` matrix 

# Example 

```julia
make_default_contrast(3)
3×3 Matrix{Float64}:
  1.0  -0.5  -0.5
 -0.5   1.0  -0.5
 -0.5  -0.5   1.0
```
"""
function make_default_contrast(n)
    C = fill(0.0, n, n)
    C .= -1 / (n - 1)
    for r ∈ 1:n
        C[r, r] = 1.0
    end
    return C
end

"""
    update_attention(dist::MDFT; Δt)

Switch attention to different attribute based on exponential waiting time.  

# Arguments

- `dist::MDFT`: a model object for simulating MDFT

# Keywords

- `Δt`: duration of time step
"""
function update_attention(dist::MDFT; Δt)
    (; κ, _att_idx) = dist
    if rand() ≤ prob_switch(κ[_att_idx], Δt)
        return _att_idx == 1 ? 2 : 1
    end
    return _att_idx
end

"""
    compute_distances(dist::MDFT, M)

Computes distance of stimuli in terms of the dominance and indifference dimensions on a unit plane. 

# Arguments

- `dist::MDFT`: a model object for simulating MDFT
- `D`: alternative × alternative distance matrix computed by `compute_distances`

# References 

Hotaling, J. M., Busemeyer, J. R., & Li, J. (2010). Theoretical developments in decision
field theory: Comment on tsetsos, usher, and chater (2010). Psychological Review, 117, 1294-1298.
"""
function compute_distances(dist::MDFT, M)
    (; β) = dist
    # number of alternatives
    n = size(M, 1)
    D = fill(0.0, n, n)
    for i ∈ 1:n
        for j ∈ (i + 1):n
            Δ1, Δ2 = (M[i, 1] - M[j, 1]) / √(2), (M[i, 2] - M[j, 2]) / √(2)
            D[i, j] = D[j, i] = (Δ2 - Δ1)^2 + β * (Δ2 + Δ1)^2
        end
    end
    return D
end

"""
    compute_feedback_matrix(dist::MDFT, D)

Computes feedback matrix `S` for Multi-attribute decision field theory. The magnitude of self-connections and inhibitory 
connections are inversely proportional to distance between alternatives in attribute space. 

# Arguments

- `dist::MDFT`: a model object for simulating MDFT
- `D`: alternative × alternative distance matrix computed by `compute_distances`

# References 

Hotaling, J. M., Busemeyer, J. R., & Li, J. (2010). Theoretical developments in decision
field theory: Comment on tsetsos, usher, and chater (2010). Psychological Review, 117, 1294-1298.
"""
function compute_feedback_matrix(dist::MDFT, D)
    (; ϕ1, ϕ2) = dist
    n = size(D, 1)
    S = fill(0.0, n, n)
    for i ∈ 1:n
        for j ∈ i:n
            δ = i == j ? 1 : 0
            S[i, j] = S[j, i] = δ - ϕ2 * exp(-ϕ1 * D[i, j]^2)
        end
    end
    return S
end

prob_switch(κ, Δt) = 1 - exp(-κ * Δt)

"""
    simulate(model::MDFT, M::AbstractArray; Δt = 0.001, _...)

Returns a matrix containing evidence samples of the AbstractMDFT decision process. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::MDFT`: an MDFT model object
- `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 
"""
function simulate(model::MDFT, M::AbstractArray; Δt = 0.001, _...)
    (; α, C, γ, _CM) = model
    n = size(M, 1)
    x = fill(0.0, n)
    μΔ = fill(0.0, n)
    t = 0.0
    _CM .= C * M * γ
    model._att_idx = rand(1:2)
    distances = compute_distances(model, M)
    model.S = compute_feedback_matrix(model, distances)
    evidence = [fill(0.0, n)]
    time_steps = [t]
    while all(x .< α)
        t += Δt
        increment!(model, x, μΔ; Δt)
        push!(evidence, copy(x))
        push!(time_steps, t)
    end
    return time_steps, reduce(vcat, transpose.(evidence))
end
