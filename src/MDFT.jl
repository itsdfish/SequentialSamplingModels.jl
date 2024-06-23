
# get_pdf_type(d::AbstractMDFT) = Approximate

# """
#     rand(
#         rng::AbstractRNG,
#         dist::AbstractMDFT,
#         n_sim::Int,
#         M::AbstractArray;
#         Δt = 0.001
#     )

# Generate `n_sim` random choice-rt pairs for the Multiattribute Decision Field Theory (MDFT).

# # Arguments

# - `rng::AbstractRNG`: a random number generator which is a subtype of `AbstractRNG`
# - `dist::AbstractMDFT`: model object for the Multiattribute Decision Field Theory (MDFT).
# - `n_sim::Int`: the number of simulated choice-rt pairs
# - `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 

# # Keywords

# - `Δt = 0.001`: time step size
# """
# function rand(
#     rng::AbstractRNG,
#     dist::AbstractMDFT,
#     n_sim::Int,
#     M::AbstractArray;
#     Δt = 0.001
# )
#     n_options = size(M, 1)
#     x = fill(0.0, n_options)
#     Δμ = fill(0.0, n_options)
#     ϵ = fill(0.0, n_options)
#     choices = fill(0, n_sim)
#     rts = fill(0.0, n_sim)
#     CM = dist.C * M
#     for i ∈ 1:n_sim
#         choices[i], rts[i] = _rand(rng, dist, x, Δμ, ϵ, CM; Δt)
#         x .= 0.0
#     end
#     return (; choices, rts)
# end

# """
#     rand(dist::AbstractMDFT, n_sim::Int, M::AbstractArray; Δt = 0.001)

# Generate multiple random choice-rt pairs for the Multiattribute Decision Field Theory (MDFT).

# # Arguments

# - `dist::AbstractMDFT`: a subtype of `AbstractMDFT` for simulating Multiattribute Decision Field Theory (MDFT). 
# - `n_sim::Int`: the number of simulated choice-rt pairs to generate 
# - `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 

# # Keywords 

# - `Δt = 0.001`: time step size 
# """
# rand(dist::AbstractMDFT, n_sim::Int, M::AbstractArray; Δt = 0.001) =
#     rand(Random.default_rng(), dist, n_sim, M; Δt)

# function rand(rng::AbstractRNG, dist::AbstractMDFT, M::AbstractArray; Δt = 0.001)
#     n_options = size(M, 1)
#     # evidence for each alternative
#     x = fill(0.0, n_options)
#     # mean change in evidence for each alternative
#     Δμ = fill(0.0, n_options)
#     # noise for each alternative 
#     ϵ = fill(0.0, n_options)
#     # precompute matric multiplication
#     CM = dist.C * M
#     return _rand(rng, dist, x, Δμ, ϵ, CM; Δt)
# end

# """
#     rand(dist::AbstractMDFT, M::AbstractArray; Δt = 0.001)

# Generate a random choice-rt pair for the Multiattribute Decision Field Theory (MDFT).

# # Arguments

# - `dist::AbstractMDFT`: a subtype of `AbstractMDFT` for simulating Multiattribute Decision Field Theory (MDFT). 
# - `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 

# # Keywords 

# - `Δt = 0.001`: time step size 
# """
# rand(dist::AbstractMDFT, M::AbstractArray; Δt = 0.001) =
#     rand(Random.default_rng(), dist, M; Δt)

# function _rand(rng::AbstractRNG, dist::AbstractMDFT, x, Δμ, ϵ, CM; Δt = 0.001)
#     (; α, τ) = dist
#     t = 0.0
#     while all(x .< α)
#         increment!(rng, dist, x, Δμ, ϵ, CM; Δt)
#         t += Δt
#     end
#     _, choice = findmax(x)
#     rt = t + τ
#     return (; choice, rt)
# end

# """
#     increment!(dist::AbstractMDFT, x, Δμ, ϵ, CM::AbstractArray; Δt = 0.001)

# Increments the preference state `x` on each time step of the simulation for Multiattribute Decision Field Theory (MDFT). 

# # Arguments

# - `dist::AbstractMDFT`: a subtype of `AbstractMDFT` for simulating MDFT. 
# - `x`: a vector of the current preference state of the alternatives 
# - `Δμ`: a pre-allocated vector for the mean updated preference state
# - `ϵ`: a pre-allocated vector for noise in the preferences of the alternatives 
# - `CM::AbstractArray`: a pre-allocated array alternative × attribute value matrix representing the product of 
#     contrast matrix and the stimulus value matrix

# # Keywords 

# - `Δt = 0.001`: time step size 
# """
# increment!(dist::AbstractMDFT, x, Δμ, ϵ, CM::AbstractArray; Δt = 0.001) =
#     increment!(Random.default_rng(), dist, x, Δμ, ϵ, CM; Δt)

# """
#     simulate(rng::AbstractRNG, model::AbstractMDFT, M::AbstractArray; Δt = 0.001, _...)

# Returns a matrix containing evidence samples of the Multiattribute Decision Field Theory (MDFT) decision process. In the matrix, rows 
# represent samples of evidence per time step and columns represent different accumulators.

# # Arguments

# - `dist::AbstractMDFT`: a subtype of `AbstractMDFT` for simulating MDFT. 
# - `M::AbstractArray`: an alternative × attribute value matrix representing the value of the stimuli 

# # Keywords 

# - `Δt = 0.001`: time step size (not applicable to `ClassicMDFT`)
# """
# function simulate(rng::AbstractRNG, model::AbstractMDFT, M::AbstractArray; _...)
#     (; α, C) = model
#     n_options = size(M, 1)
#     CM = C * M
#     x = fill(0.0, n_options)
#     μΔ = fill(0.0, n_options)
#     ϵ = fill(0.0, n_options)
#     t = 0.0
#     evidence = [fill(0.0, n_options)]
#     time_steps = [t]
#     while all(x .< α)
#         t += Δt
#         increment!(model, x, μΔ, ϵ, CM; Δt)
#         push!(evidence, copy(x))
#         push!(time_steps, t)
#     end
#     return time_steps, reduce(vcat, transpose.(evidence))
# end

# function make_default_contrast(S::AbstractArray{T}) where {T}
#     n = size(S, 1)
#     C = Array{T, 2}(undef, n, n)
#     C .= -1 / (n - 1)
#     for r ∈ 1:n
#         C[r, r] = 1.0
#     end
#     return C
# end
