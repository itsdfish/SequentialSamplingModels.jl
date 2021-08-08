"""
    AttentionalDiffusion(;ν1=5.0, ν2=4.0, α=1.0, z=α*.5, θ=.6, σ=.05, Δ=.005)

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
@concrete struct AttentionalDiffusion <: SequentialSamplingModel
    ν1
    ν2
    α
    z
    θ
    σ
    Δ
end

function AttentionalDiffusion(;ν1=5.0, ν2=4.0, α=1.0, z=0.0, θ=.3, σ=.02, Δ=.0004)
    return AttentionalDiffusion(ν1, ν2, α, z, θ, σ, Δ)
end

Broadcast.broadcastable(x::AttentionalDiffusion) = Ref(x)

"""
    rand(dist::AttentionalDiffusion, n_sim::Int, fixation, args...; kwargs...)

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
function rand(dist::AttentionalDiffusion, n_sim::Int, fixation, args...; kwargs...)
    rts = [Vector{Int}() for _ in 1:2]
    for sim in 1:n_sim 
       choice,rt = _rand(dist, ()->fixation(args...; kwargs...))
       push!(rts[choice], rt)
    end
    return rts
end

"""
    rand(dist::AttentionalDiffusion, fixation, args...; kwargs...)

Generate a single simulated trial from the attention diffusion model.

# Arguments

- `dist`: an attentional diffusion model object
- `fixation`: a function of the visual fixation process which returns 1 for alternative 
    and 2 for alternative 2
- `args...`: optional positional arguments for the `fixation` function

# Keywords

- `kwargs...`: optional keyword arguments for the `fixation` function
"""
function rand(dist::AttentionalDiffusion, fixation, args...; kwargs...)
    return _rand(dist, ()->fixation(args...; kwargs...))
end

function _rand(dist::AttentionalDiffusion, fixation)
    @unpack α,z,σ,ν1,ν2,θ,Δ = dist
    t = 0
    v = z
    while abs(v) < α
        t += 1
        option = fixation()
        if option == 1
            v += Δ * (ν1 - θ * ν2) + noise(σ)
        elseif option == 2
            v -= Δ * (ν2 - θ * ν1) + noise(σ)
        else
            v += noise(σ)
        end
    end
    choice = (v < α) + 1
    return choice,t
end

noise(σ) = rand(Normal(0, σ))