# SequentialSamplingModels

This package is a collection of sequential sampling models and is based on the Distributions.jl API.
The examples below show basic usage. Addition information can be found in the REPL with the help function, e.i., ```? LNR```. 

## Lognormal Race Model

```julia
using SequentialSamplingModels
dist = LNR(μ=[-2,-3], σ=1.0, ϕ=.3)
data = rand(dist, 10)
like = pdf.(dist, data)
loglike = logpdf.(dist, data)
```

## Linear Ballistic Accumulator

```julia
using SequentialSamplingModels
dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```

## Shifted Wald

```julia
using SequentialSamplingModels
dist = Wald(ν=3.0, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```

## Wald Mixture

```julia
using SequentialSamplingModels
dist = WaldMixture(ν=3.0, σ=.2, α=.5, θ=.130)
rt = rand(dist, 10)
like = pdf.(dist, rt)
loglike = logpdf.(dist, rt)
```
## Racing Diffusion Model

```julia
using SequentialSamplingModels
dist = DiffusionRace(;ν=[1.0,.5], k=0.5, A=1.0, θ=.2)
data = rand(dist, 10)
like = pdf.(dist, data)
loglike = logpdf.(dist, data)
```

## Attentional Diffusion Model

```julia
using StatsBase, Parameters, SequentialSamplingModels

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
     @unpack mat,n,state = transition
     w = mat[state,:]
     next_state = sample(1:n, Weights(w))
     transition.state = next_state
     return next_state
 end

 model = AttentionalDiffusion()
 
 tmat = Transition(
     [.98 .015 .005;
     .015 .98 .005;
     .45 .45 .1]
    )

 rts = rand(model, 1000, x->attend(x), tmat)
 ```