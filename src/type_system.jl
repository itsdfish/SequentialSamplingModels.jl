abstract type Mixed <: ValueSupport end

"""
    SSM2D = Distribution{Multivariate, Mixed}

An abstract type for sequential sampling models characterized by a multivariate choice-reaction time distribution.
Sub-types of `SSM2D` output a `NamedTuple` consisting of a vector of choices and reaction times. 
"""
const SSM2D = Distribution{Multivariate, Mixed}

"""
    ContinuousMultivariateSSM <: ContinuousMultivariateDistribution

An abstract type for continuous multivariate sequential sampling models e.g., a circular drift diffusion model.
"""
abstract type ContinuousMultivariateSSM <: ContinuousMultivariateDistribution end

"""
    SSM1D <: ContinuousUnivariateDistribution

An abstract type for sequential sampling models characterized by a single choice reaction time distribution.
Sub-types of `SSM1D` output a vector of reaction times.
"""
abstract type SSM1D <: ContinuousUnivariateDistribution end

"""
    AbstractDDM <: SSM2D

An abstract type for the drift diffusion model.  
"""
abstract type AbstractDDM <: SSM2D end

"""
    AbstractaDDM <: SSM2D

An abstract type for the attentional drift diffusion model.  
"""
abstract type AbstractaDDM <: AbstractDDM end

"""
    AbstractLBA <: SSM2D

An abstract type for the linear ballistic accumulator model.  
"""
abstract type AbstractLBA <: SSM2D end

"""
    AbstractWald <: SSM1D

An abstract type for the Wald model.  
"""
abstract type AbstractWald <: SSM1D end

"""
    AbstractLNR <: SSM2D

An abstract type for the lognormal race model
"""
abstract type AbstractLNR <: SSM2D end

"""
    AbstractMLBA <: AbstractLBA

An abstract type for the multi-attribute linear ballistic accumulator
"""
abstract type AbstractMLBA <: AbstractLBA end

"""
    AbstractLCA <: SSM2D

An abstract type for the leaky competing accumulator model
"""
abstract type AbstractLCA <: SSM2D end

"""
    AbstractMDFT <: SSM2D 
"""
abstract type AbstractMDFT <: SSM2D end

"""
    AbstractPoissonRace <: SSM2D

An abstract type for the Poisson race model.
"""
abstract type AbstractPoissonRace <: SSM2D end

"""
    AbstractstDDM <: SSM2D

An abstract type for the starting-time diffusion decision model.
"""
abstract type AbstractstDDM <: SSM2D end

"""
    AbstractRDM <: SSM2D

An abstract type for the racing diffusion model.
"""
abstract type AbstractRDM <: SSM2D end

"""
    AbstractShiftedLogNormal <: SSM1D

An abstract type for the shifted lognormal model.
"""
abstract type AbstractShiftedLogNormal <: SSM1D end

abstract type PDFType end

"""
    Exact <: PDFType

Has closed-form PDF. 
"""
struct Exact <: PDFType end

"""
    Approximate <: PDFType

Has approximate PDF based on kernel density estimator. 
"""
struct Approximate <: PDFType end

get_simulator_type(d::SSM1D) = Exact
get_simulator_type(d::SSM2D) = Exact
get_simulator_type(d::ContinuousMultivariateSSM) = Exact

get_pdf_type(d::SSM1D) = Exact
get_pdf_type(d::SSM2D) = Exact
get_pdf_type(d::ContinuousMultivariateSSM) = Exact

minimum(d::SSM1D) = 0.0
maximum(d::SSM1D) = Inf

minimum(d::SSM2D) = 0.0
maximum(d::SSM2D) = Inf

insupport(d::SSM1D, rt::Real) = rt ≥ minimum(d) && rt ≤ maximum(d)
insupport(d::SSM2D, data) = data.rt ≥ minimum(d) && data.rt ≤ maximum(d)

Base.broadcastable(x::SSM1D) = Ref(x)
Base.broadcastable(x::SSM2D) = Ref(x)
Base.broadcastable(x::ContinuousMultivariateSSM) = Ref(x)

Base.length(d::SSM2D) = 2

rand(d::SSM2D; kwargs...) = rand(Random.default_rng(), d; kwargs...)
rand(d::ContinuousMultivariateSSM; kwargs...) = rand(Random.default_rng(), d; kwargs...)
rand(d::ContinuousMultivariateSSM, n_trials::Int; kwargs...) =
    rand(Random.default_rng(), d, n_trials; kwargs...)

"""
    rand(rng::AbstractRNG, d::SSM2D, N::Int; kwargs...)

Default method for Generating `n_sim` random choice-rt pairs from a sequential sampling model 
with more than one choice option.

# Arguments

- `d::SSM2D`: a 2D sequential sampling model.
- `n_trials::Int`: the number of simulated choices and rts  

# Keywords

- `kwargs...`: optional keyword arguments 
"""
function rand(rng::AbstractRNG, d::SSM2D, n_trials::Int; kwargs...)
    choice = fill(0, n_trials)
    rt = fill(0.0, n_trials)
    for i ∈ 1:n_trials
        choice[i], rt[i] = rand(rng, d; kwargs...)
    end
    return (; choice, rt)
end

rand(d::SSM2D, n_trials::Int; kwargs...) =
    rand(Random.default_rng(), d, n_trials; kwargs...)

"""
    logpdf(d::SSM2D, data::NamedTuple) 

Computes the likelihood for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
logpdf(d::SSM2D, data::NamedTuple) = logpdf.(d, data.choice, data.rt)
logpdf(d::SSM2D, data::AbstractVector{<:Real}) = logpdf(d, Int(data[1]), data[2])

"""
    loglikelihood(d::SSM1D, data::AbstractArray{T, 1})

Computes the summed log likelihood for a 1D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::AbstractVector{<:Real}`: a vector of reaction times
"""
loglikelihood(d::SSM1D, data::AbstractVector{<:Real}) = sum(logpdf.(d, data))

"""
    loglikelihood(d::SSM2D, data::NamedTuple) 

Computes the summed log likelihood for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
loglikelihood(d::SSM2D, data::NamedTuple) = sum(logpdf.(d, data...))

loglikelihood(d::SSM2D, data::AbstractArray{<:Real, 2}) =
    sum(logpdf.(d, Int.(data[:, 1]), data[:, 2]))

"""
    pdf(d::SSM2D, data::NamedTuple) 

Computes the probability density for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
pdf(d::SSM2D, data::NamedTuple, args...; kwargs...) =
    pdf.(d, data.choice, data.rt, args...; kwargs...)

pdf(d::SSM2D, data::AbstractArray{Real, 2}) = pdf(d, Int(data[1]), data[2])

"""
    cdf(d::SSM2D, choice::Int, ub=10)

Computes the cumulative density for a given choice. The cumulative density is based on 
an analytic formula, a numeric integration of `pdf`, or Monte Carlo simulation, depending on which is 
available for a given model. 

# Arguments
- `d::SSM2D`: a 2D sequential sampling model.
- `choice::Int`: the number of simulated choices and rts  
- `ub::Real`: upper bound of integration
- `args...`: optional arguments passed to `rand`
"""
function cdf(d::SSM2D, choice::Int, ub::Real, args...)
    return cdf(get_pdf_type(d), d, choice, ub, args...)
end

function cdf(::Type{<:Exact}, d::SSM2D, choice::Int, ub::Real, args...)
    return hcubature(t -> pdf(d, choice, t[1], args...), [d.τ], [ub])[1]::Float64
end

function cdf(
    ::Type{<:Approximate},
    d::SSM2D,
    choice::Int,
    ub::Real,
    args...;
    n_sim = 10_000
)
    c, rt = rand(d, n_sim, args...)
    return mean(c .== choice .&& rt .≤ ub)
end

function survivor(d::SSM2D, choice::Int, ub::Real, args...)
    return 1 - cdf(d, choice, ub, args...)
end

"""
    cdf(d::SSM1D, choice::Int, ub=10)

Computes the cumulative density for a given choice. The cumulative density is based on 
an analytic formula, a numeric integration of `pdf`, or Monte Carlo simulation, depending on which is 
available for a given model. 

# Arguments
- `d::SSM1D`: a 1D sequential sampling model.
- `ub`: upper bound of integration
"""
function cdf(d::SSM1D, ub::Real)
    return cdf(get_pdf_type(d), d, ub)
end

function cdf(::Type{<:Exact}, d::SSM1D, ub)
    return hcubature(t -> pdf(d, t[1]), [d.τ], [ub])[1]::Float64
end

function cdf(::Type{<:Approximate}, d::SSM1D, ub; n_sim = 10_000)
    rt = rand(d, n_sim)
    return mean(rt .≤ ub)
end

function survivor(d::SSM1D, ub)
    return 1 - cdf(d, ub)
end

"""
    n_options(dist::SSM2D)

Returns the number of choice options based on the length of the drift rate vector `ν`.

# Arguments

- `d::SSM2D`: a sub-type of `SSM2D`
"""
n_options(d::SSM2D) = length(d.ν)

"""
    n_options(dist::SSM1D)

Returns 1 for the number of choice options

# Arguments

- `d::SSM1D`: a sub-type of `SSM1D`
"""
n_options(d::SSM1D) = 1

n_options(d::ContinuousMultivariateSSM) = length(d.ν)

"""
    simulate(model::SSM2D, args...; Δt = .001, kwargs...)

Returns a matrix containing evidence samples from a 2D SSM. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::SSM2D`: a subtype of a 2D SSM 
- `args...`: optional positional arguments 

# Keywords

- `Δt = .001`: size of time step of decision process in seconds
- `kwargs...`: optional keyword arguments
"""
simulate(model::SSM2D, args...; Δt = 0.001, kwargs...) =
    simulate(Random.default_rng(), model, args...; Δt, kwargs...)

"""
    simulate(model::SSM1D, args...; Δt = .001, kwargs...)

Returns a matrix containing evidence samples from a 2D SSM. In the matrix, rows 
represent samples of evidence per time step and columns represent different accumulators.

# Arguments

- `model::SSM1D`: a subtype of a 2D SSM 
- `args...`: optional positional arguments 

# Keywords

- `Δt = .001`: size of time step of decision process in seconds
- `kwargs...`: optional keyword arguments
"""
simulate(model::SSM1D, args...; Δt = 0.001, kwargs...) =
    simulate(Random.default_rng(), model, args...; Δt, kwargs...)

"""
    increment!(model::SSM2D, x, μΔ; Δt = 0.001) 

Increments the evidence states `x` on each time step. 

# Arguments

- `dist::SSM2D`: a subtype of 2D SSM.
- `x`: a vector of preference states 
- `μΔ`: a vector of mean change in evidence (i.e. drift rates)

# Keywords

- `Δt = 0.001`: time step size
"""
increment!(model::SSM2D, x, μΔ; Δt = 0.001) =
    increment!(Random.default_rng(), model, x, μΔ; Δt)

"""
    increment!(model::SSM1D, x, μΔ; Δt = 0.001) 

Increments the evidence states `x` on each time step. 

# Arguments

- `dist::SSM1D`: a subtype of 1D SSM.
- `x`: a vector of preference states 
- `μΔ`: a vector of mean change in evidence (i.e. drift rates)

# Keywords

- `Δt = 0.001`: time step size
"""
increment!(model::SSM1D, x, μΔ; Δt = 0.001) =
    increment!(Random.default_rng(), model, x, μΔ; Δt)
