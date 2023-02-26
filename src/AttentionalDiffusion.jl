abstract type AbstractaDDM <: SequentialSamplingModel end

@concrete struct aDDM <: AbstractaDDM
    ν1
    ν2
    α
    z
    θ
    σ
    Δ
end

"""
    aDDM(;ν1=5.0, ν2=4.0, α=1.0, z=α*.5, θ=.3, σ=.02, Δ=.0004)

Constructor for attentional diffusion model object. 

# Keywords 

- `ν1=5.0`: relative decision value for alternative 1
- `ν2=4.0`: relative decision value for alternative 2
- `α=1.0`: evidence threshold 
- `z=0.0`: initial evidence 
- `θ=.3`: bias towards attended alternative (lower indicates more bias)
- `σ=.02`: standard deviation of noise in evidence accumulation
- `Δ=.0004`: constant of evidence accumulation speed (evidence per ms)

# References 

Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of 
value in simple choice. Nature neuroscience, 13(10), 1292-1298.
"""
function aDDM(;ν1=5.0, ν2=4.0, α=1.0, z=0.0, θ=.3, σ=.02, Δ=.0004)
    return aDDM(ν1, ν2, α, z, θ, σ, Δ)
end

Broadcast.broadcastable(x::aDDM) = Ref(x)

"""
    rand(dist::aDDM, n_sim::Int, fixation, args...; kwargs...)

Generate `n_sim` simulated trials from the attention diffusion model.

# Arguments

- `dist`: an attentional diffusion model object
- `n_sim::Int`: the number of simulated trials
- `fixation`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args...`: optional positional arguments for the `fixation` function

# Keywords

- `kwargs...`: optional keyword arguments for the `fixation` function
"""
function rand(dist::AbstractaDDM, n_sim::Int, fixation, args...; rand_state! = _rand_state!, kwargs...)
    rts = [Vector{Float64}() for _ in 1:2]
    for sim in 1:n_sim 
        rand_state!(args...; kwargs...)
        choice,rt = _rand(dist, () -> fixation(args...; kwargs...))
        push!(rts[choice], rt)
    end
    return rts
end

function _rand_state!(tmat)
    tmat.state = rand(1:tmat.n)
    return nothing
 end
 

"""
    rand(dist::aDDM, fixation, args...; kwargs...)

Generate a single simulated trial from the attention diffusion model.

# Arguments

- `dist`: an attentional diffusion model object
- `fixation`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args...`: optional positional arguments for the `fixation` function

# Keywords

- `kwargs...`: optional keyword arguments for the `fixation` function
"""
function rand(dist::AbstractaDDM, fixation, args...; kwargs...)
    return _rand(dist, () -> fixation(args...; kwargs...))
end

function _rand(dist::AbstractaDDM, fixation)
    (;α,z) = dist
    t = 0.0
    Δt = .001
    v = z
    while abs(v) < α
        t += Δt
        location = fixation()
        v += update(dist, location)
    end
    choice = (v < α) + 1
    return choice,t
end

"""
    update(dist::aDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `dist::aDDM`: a model object for the attentional drift diffusion model
- `location`: an index for fixation location 
"""
function update(dist::aDDM, location)
    (;σ,ν1,ν2,θ,Δ) = dist
    # option 1
    if location == 1
        return Δ * (ν1 - θ * ν2) + noise(σ)
    # option 2
    elseif location == 2
        return -Δ * (ν2 - θ * ν1) + noise(σ)
    else
        return noise(σ)
    end
    return -100.0
end 

noise(σ) = rand(Normal(0, σ))

@concrete struct maaDDM <: AbstractaDDM
    ν₁₁
    ν₁₂
    ν₂₁
    ν₂₂
    α
    z
    θ
    ϕ
    ω
    σ
    Δ
end

"""
    maaDDM(; ν₁₁ = 4.0, 
            ν₁₂ = 5.0, 
            ν₂₁ = 5.0, 
            ν₂₂ = 4.0, 
            α = 1.0, 
            z = 0.0, 
            θ = .3, 
            ϕ = .50, 
            ω = .70, 
            σ = .02, 
            Δ = .0004)
            
Constructor for multialternative attentional diffusion model object. 

In this version of the model, the non-attended attribute of the non-attended alternative is doubly discounted. For example,
the mean drift rate for the attribute 1 of alternative 1 is given by:

```julia
    Δ * (ω * (ν₁₁ - θ * ν₂₁) + (1 - ω) * ϕ * (ν₁₂ - θ * ν₂₂))
```

# Keywords 

- `ν₁₁=5.0`: relative decision value for alternative 1, attribute 1
- `ν₁₂=4.0`: relative decision value for alternative 1, attribute 2
- `ν₂₁`=5.0: relative decision value for alternative 2, attribute 1
- `ν₂₂`4.0:  relative decision value for alternative 2, attribute 2
- `α=1.0`: evidence threshold 
- `z=0.0`: initial evidence 
- `θ=.3`: bias away from unattended alternative (lower indicates more bias)
- `ϕ=.50`: bias away from unattended attribute 
- `ω=.70`: attribute weight
- `σ=.02`: standard deviation of noise in evidence accumulation
- `Δ=.0004`: constant of evidence accumulation speed (evidence per ms)

# References 

Yang, X., & Krajbich, I. (2023). A dynamic computational model of gaze and choice in multi-attribute decisions. 
Psychological Review, 130(1), 52.
"""
function maaDDM(; ν₁₁ = 4.0, 
                    ν₁₂ = 5.0, 
                    ν₂₁ = 5.0, 
                    ν₂₂ = 4.0, 
                    α = 1.0, 
                    z = 0.0, 
                    θ = .3, 
                    ϕ = .50, 
                    ω = .70, 
                    σ = .02, 
                    Δ = .0004)

    return maaDDM(ν₁₁, ν₁₂, ν₂₁, ν₂₂, α, z, θ, ϕ, ω, σ, Δ)
end

Broadcast.broadcastable(x::maaDDM) = Ref(x)

"""
    update(dist::maaDDM, location)

Returns the change evidence for a single iteration. 

# Arguments

- `dist::maaDDM`: a model object for the multiattribute attentional drift diffusion model
- `location`: an index for fixation location 
"""
function update(dist::maaDDM, location)
    (;ν₁₁,ν₁₂,ν₂₁,ν₂₂,θ,ϕ,ω,Δ,σ) = dist
    # option 1, attribute 1
    if location == 1
        return Δ * (ω * (ν₁₁ - θ * ν₂₁) + (1 - ω) * ϕ * (ν₁₂ - θ * ν₂₂)) + noise(σ)
    # option 1, attribute 2
    elseif location == 2
        return Δ * (ϕ * ω * (ν₁₁ - θ * ν₂₁) + (1 - ω) * (ν₁₂ - θ * ν₂₂)) + noise(σ)
    # option 2, attribute 1
    elseif location == 3
        return Δ * (ω * (θ * ν₁₁ - ν₂₁) + (1 - ω) * ϕ * (θ * ν₁₂ - ν₂₂)) + noise(σ)
    # option 2, attribute 2
    else
        return Δ * (ϕ * ω * (θ * ν₁₁ - ν₂₁) + (1 - ω) * (θ * ν₁₂ - ν₂₂)) + noise(σ)
    end
    return -100.0
end 
