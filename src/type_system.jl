abstract type Mixed <: ValueSupport end 

"""
    SSM2D = Distribution{Multivariate, Mixed}

An abstract type for sequential sampling models characterized by a multivariate choice-reaction time distribution.
Sub-types of `SSM2D` output a `NamedTuple` consisting of a vector of choices and reaction times. 
"""
const SSM2D = Distribution{Multivariate, Mixed}

"""
    SSM1D <: ContinuousUnivariateDistribution

An abstract type for sequential sampling models characterized by a single choice reaction time distribution.
Sub-types of `SSM1D` output a vector of reaction times.
"""
abstract type SSM1D <: ContinuousUnivariateDistribution end 

"""
    AbstractaDDM <: SSM2D

An abstract type for the drift diffusion model.  
"""
abstract type AbstractaDDM <: SSM2D end

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
    AbstractLCA <: SSM2D

An abstract type for the leaky competing accumulator model
"""
abstract type AbstractLCA <: SSM2D end

"""
    AbstractRDM <: SSM2D

An abstract type for the racing diffusion model.
"""
abstract type AbstractRDM <: SSM2D end

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

get_pdf_type(d::SSM1D) = Exact
get_pdf_type(d::SSM2D) = Exact

minimum(d::SSM1D) = 0.0
maximum(d::SSM1D) = Inf

minimum(d::SSM2D) = 0.0
maximum(d::SSM2D) = Inf

insupport(d::SSM2D, data) = data.rt ≥ minimum(d) && data.rt ≤ maximum(d)

Base.broadcastable(x::SSM1D) = Ref(x)
Base.broadcastable(x::SSM2D) = Ref(x)

vectorize(d::SSM2D, r::NamedTuple) = [r...]

Base.length(d::SSM2D) = 2

rand(d::SSM2D) = rand(Random.default_rng(), d)
rand(d::SSM2D, n::Int) = rand(Random.default_rng(), d, n)

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
    loglikelihood(d::SSM2D, data::NamedTuple) 

Computes the summed log likelihood for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
loglikelihood(d::SSM2D, data::NamedTuple) = sum(logpdf.(d, data...))

loglikelihood(d::SSM2D, data::AbstractArray{<:Real,2}) = sum(logpdf.(d, Int.(data[:,1]), data[:,2]))

"""
    pdf(d::SSM2D, data::NamedTuple) 

Computes the probability density for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
pdf(d::SSM2D, data::NamedTuple) = pdf.(d, data.choice, data.rt)

pdf(d::SSM2D, data::AbstractArray{Real,2}) = pdf(d, Int(data[1]), data[2])

"""
    rand(rng::AbstractRNG, d::SSM2D, N::Int)

Default method for Generating `n_sim` random choice-rt pairs from a sequential sampling model 
with more than one choice option.

# Arguments
- `d::SSM2D`: a 2D sequential sampling model.
- `n_sim::Int`: the number of simulated choices and rts  
"""
function rand(rng::AbstractRNG, d::SSM2D, N::Int)
    choice = fill(0, N)
    rt = fill(0.0, N)
    for i in 1:N
        choice[i],rt[i] = rand(rng, d)
    end
    return (;choice,rt)
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