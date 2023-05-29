# SequentialSamplingModels

This package is a collection of sequential sampling models and is based on the Distributions.jl API.
The examples below show basic usage. Addition information can be found in the REPL with the help function, e.i., ```? LNR```. 

## Installation

You can install a stable version of Turing by running the following in the Julia REPL:

```julia
julia> ] add SequentialSamplingModels
```

The package can then be loaded with:

```julia
using SequentialSamplingModels
```

## Models

*SequentialSamplingModels* implements a variety of sequential sampling models that can be used to model specific outcomes (e.g., reaction time data).

### Lognormal Race Model (LNR)

See [Heathcote and Love (2012)](http://www.frontiersin.org/Cognitive_Science/10.3389/fpsyg.2012.0029) and [Rouder et al. (2015)](https://link.springer.com/article/10.1007/s11336-013-9396-3).

```julia
dist = LNR(μ=[-2,-3], σ=1.0, ϕ=.3)
choice,rts = rand(dist, 10)
like = pdf.(dist, choice, rts)
loglike = logpdf.(dist, choice, rts)
```

### Linear Ballistic Accumulator (LBA)

See [Brown and Heathcote (2008)](https://www.sciencedirect.com/science/article/abs/pii/S0010028507000722?via%3Dihub).

```julia
dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```

### Shifted Wald

```julia
dist = Wald(ν=3.0, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```

### Wald Mixture

```julia
dist = WaldMixture(ν=3.0, σ=.2, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
### Racing Diffusion Model

```julia
dist = DiffusionRace(;ν=[1.0,.5], k=0.5, A=1.0, θ=.2)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rts)
loglike = logpdf.(dist, choice, rts)
```

### Attentional Diffusion Model

```julia
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
     w = mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end

 model = AttentionalDiffusion()
 
 tmat = Transition([.98 .015 .005;
                    .015 .98 .005;
                    .45 .45 .1])

 choice,rts = rand(model, 1000, attend, tmat)
 ```

 ### Multiattribute Attentional Drift Diffusion

 ```julia 
using StatsBase

model = maaDDM()

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
     w = mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end

 # transition within alternatives.
 # columns 
 # 1. alternative 1, attribute 1
 # 2. alternative 1, attribute 2
 # 3. alternative 2, attribute 1
 # 4. alternative 2, attribute 2
 tmat = Transition([.98 .015 .0025 .0025;
                    .015 .98 .0025 .0025;
                    .0025 .0025 .98 .015;
                    .0025 .0025 .015 .98])

 choice,rts = rand(model, 1000, attend, tmat)
 ```
