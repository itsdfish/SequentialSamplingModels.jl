"""
    stDDM{T<:Real} <: AbstractstDDM

An object for the starting-time diffusion decision model.

# Parameters 

- `ν`:  vector of drift rates for attribute one and two
- `α`:  evidence threshold
- `τ`:  non-decision time 
- `s`:  initial latency bias (positive for attribute two, negative for attribute one)
- `z`:  initial evidence 
- `η`:  vector of variability in drift rate for attribute one and two
- `σ`:  diffusion noise 
- `Δt`: time step 

# Constructors 

    stDDM(ν, α, τ, s, z, η, σ, Δt)

    stDDM(;ν = [0.5,0.6], 
        α = 1.0, 
        τ = .300, 
        s = 0.50, 
        z = 0.50, 
        η = [1.0,1.0], 
        σ = 1,
        Δt = .001)
        
# Example 

```julia 
using SequentialSamplingModels

ν = [0.5, 0.6]
α = 1.0
τ = 0.300
s = 0.50
z = 0.50
η = [1.0, 1.0]
σ = 1
Δt = 0.001

# Create stDDM model instance
dist = stDDM(;ν, α, τ, s, z, η, σ, Δt)

choices,rts = rand(dist, 500)
```

# References

Amasino, D.R., Sullivan, N.J., Kranton, R.E. et al. Amount and time exert independent influences on intertemporal choice. Nat Hum Behav 3, 383–392 (2019). https://doi.org/10.1038/s41562-019-0537-2
Barakchian, Z., Beharelle, A.R. & Hare, T.A. Healthy decisions in the cued-attribute food choice paradigm have high test-retest reliability. Sci Rep, (2021). https://doi.org/10.1038/s41598-021-91933-6
Chen, HY., Lombardi, G., Li, SC. et al. Older adults process the probability of winning sooner but weigh it less during lottery decisions. Sci Rep, (2022). https://doi.org/10.1038/s41598-022-15432-y
Lombardi, G., & Hare, T. Piecewise constant averaging methods allow for fast and accurate hierarchical Bayesian estimation of drift diffusion models with time-varying evidence accumulation rates. PsyArXiv, (2021). https://doi.org/10.31234/osf.io/5azyx
Sullivan, N.J., Huettel, S.A. Healthful choices depend on the latency and rate of information accumulation. Nat Hum Behav 5, 1698–1706 (2021). https://doi.org/10.1038/s41562-021-01154-0
"""

mutable struct stDDM{T<:Real} <: AbstractstDDM
    ν::Vector{T}
    α::T
    τ::T
    s::T
    z::T
    η::Vector{T}
    σ::T
    Δt::T
end

function stDDM(ν, α, τ, s, z, η, σ, Δt)
    _, α, τ, s, z,_ , σ, Δt = promote(ν[1], α, τ, s, z, η[1], σ, Δt)
    ν = convert(Vector{typeof(τ)}, ν)
    η = convert(Vector{typeof(τ)}, η)
    return stDDM(ν, α, τ, s, z, η, σ, Δt)
end

function stDDM(;ν = [0.5,0.6], 
    α = 1.0, 
    τ = .300, 
    s = 0.50, 
    z = 0.50, 
    η = fill(1.0, length(ν)), 
    σ = 1,
    Δt = .001)

    return stDDM(ν, α, τ, s, z, η, σ, Δt)
end

function params(d::AbstractstDDM)
    (d.ν, d.α, d.τ, d.s, d.z, d.η, d.σ, d.Δt)    
end

get_pdf_type(d::AbstractstDDM) = Approximate

"""
    rand(dist::AbstractstDDM)

Generate a random choice-rt pair for starting-time diffusion decision model.

# Arguments
- `dist`: model object for the starting-time diffusion decision model. 
"""
function rand(rng::AbstractRNG, dist::AbstractstDDM)
    return simulate_trial(rng, dist)
end

"""
    rand(dist::AbstractstDDM, n_sim::Int)

Generate `n_sim` random choice-rt pairs for the starting-time diffusion decision model.

# Arguments
- `dist`: model object for the starting-time diffusion decision model.
- `n_sim::Int`: the number of simulated choice-rt pairs  
"""
function rand(rng::AbstractRNG, dist::AbstractstDDM, n_sim::Int)
    choices = fill(0, n_sim)
    rts = fill(0.0, n_sim)
    for i in 1:n_sim
        choices[i],rts[i] = simulate_trial(rng, dist)
    end
    return (;choices,rts) 
end

# noise(rng, σ) = rand(rng, Normal(0, σ))

"""
simulate_trial(rng::AbstractRNG, dist::AbstractstDDM, TMax)

Generate a single simulated trial from the starting-time diffusion decision model.

# Arguments

- `rng`: a random number generator
- `model::AbstractstDDM`: a starting-time diffusion decision model object
- `TMax`: total/max time for simulation
"""

function simulate_trial(rng::AbstractRNG, dist; TMax=6)
    (;ν, α, τ, s, z, η, σ, Δt) = dist

    lt = Int(TMax / Δt)
    vec_tν1 = ones(Int, lt)
    vec_tν2 = ones(Int, lt)
    aux = abs(Int(s / Δt))
    
    if s > 0
        vec_tν1[1:aux] .= 0
    elseif s < 0
        vec_tν2[1:aux] .= 0
    end
    
    t = TMax
    choice = 0  # Initialize choice with a default value

    X = z * α
    flag = false
    cont = 1
    while !flag && cont <= lt
        # noise = noise(rng, σ) * sqrt(Δt)
        noise = rand(rng, Normal(0, σ)) * sqrt(Δt)

        X += (ν[1] * η[1] * vec_tν1[cont] + ν[2] * η[2] * vec_tν2[cont]) * Δt + noise
        
        if X > α
            # t = τ + cont * Δt
            choice = 1
            flag = true
        elseif X < 0
            # t = -τ - cont * Δt
            choice = 2
            flag = true
        end
        t = τ + cont * Δt

        cont += 1
    end
    
    return (;choice,rt=t)

end


# """
#     simulate(model::AbstractstDDM; _...)

# Returns a matrix containing evidence samples of the stDDM decision process. In the matrix, rows 
# represent samples of evidence per time step and columns represent different accumulators.

# # Arguments

# - `model::AbstractstDDM`: a starting-time diffusion decision model diffusion model object
# """
# function simulate(rng::AbstractRNG, model::AbstractstDDM, _...)
#     (;ν,α,z,Δt) = model
#     x = α * z
#     t = 0.0
#     evidence = [x]
#     time_steps = [t]
#     while (x < α) && (x > 0)
#         t += Δt
#         x += ν * Δt + rand(rng, Normal(0.0, 1.0)) * √(Δt)
#         push!(evidence, x)
#         push!(time_steps, t)
#     end
#     return time_steps,evidence
# end
