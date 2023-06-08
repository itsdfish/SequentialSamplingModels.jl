"""
    DDM 

Model object for the Drift Diffusion Model. 

# Fields

- `ν`: drift rate
- `η`: across-trial drift rate standard deviation
- `α`: evidence threshold 
- `z`: mean starting point
- `sz`: range of starting point variability
- `τ`: non-decision time 
- `st`: range of non-decision time
- `σ`: diffusion noise 
- `Δt`: time step 
"""
@concrete mutable struct DDM <: SequentialSamplingModel
    ν
    η
    α
    z
    sz
    τ
    st
    σ
    Δt
end

"""
DDM(; ν = 0.50,
    η = 0.10,
    α = 0.08,
    z = 0.04,
    sz = 0.02,
    τ = 0.30,
    st = .02,
    σ = 0.10,
    Δt = 0.001)

Constructor for Drift Diffusion Model model. 
    
# Keywords 
- `ν=.50`: drift rates 
- `η=0.10`: across-trial drift rate standard deviation
- `α=0.08`: evidence threshold 
- `z=0.04`: mean starting point
- `sz=0.02`: range of starting point variability
- `τ=0.3`: non-decision time 
- `st=0.02`: range of non-decision time
- `σ=0.10`: diffusion noise 
- `Δt=.001`: time step 
"""
function DDM(; ν = 0.50,
    η = 0.10,
    α = 0.08,
    z = 0.04,
    sz = 0.02,
    τ = 0.30,
    st = .02,
    σ = 0.10,
    Δt = 0.001)

    return DDM(ν, η, α, z, sz, τ, st, σ, Δt)
end

"""
    rand(dist::DDM)

Generate a random choice-rt pair for the Drift Diffusion Model.

# Arguments
- `dist`: model object for the Drift Diffusion Model. 
"""
function rand(dist::DDM)
    (;ν,α,z,σ,τ, Δt,η,st,sz) = dist
    t = 0.0
    srΔt = √Δt
    x = rand(Uniform(z - sz / 2, z + sz / 2))
    ν′ = rand(Normal(ν, η))
    while (x < α) && (x > 0)
        ϵ = rand(Normal(0, σ))
        x += ν′* Δt + ϵ * srΔt
        t += Δt
    end
    choice = (x < α) + 1
    t += rand(Uniform(τ - st / 2, τ + st / 2))
    return choice,t
end

"""
    rand(dist::DDM, n_sim::Int)

Generate `n_sim` random choice-rt pairs for the Drift Diffusion Model.

# Arguments
- `dist`: model object for the Drift Diffusion Model.
- `n_sim::Int`: the number of simulated choice-rt pairs  
"""
function rand(dist::DDM, n_sim::Int)
    choices = fill(0, n_sim)
    rts = fill(0.0, n_sim)
    for i in 1:n_sim
        choices[i],rts[i] = rand(dist)
    end
    return choices,rts 
end