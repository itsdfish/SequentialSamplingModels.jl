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

abstract type AbstractWald <: SSM1D end

abstract type AbstractaDDM <: SSM2D end

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
logpdf(d::SSM2D, data::NamedTuple; kwargs...) = logpdf.(d, data.choice, data.rt; kwargs...)
logpdf(d::SSM2D, data::Vector{Real}; kwargs...) = logpdf(d, Int(data[1]), data[2]; kwargs...) 

"""
    loglikelihood(d::SSM2D, data::NamedTuple) 

Computes the summed log likelihood for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
loglikelihood(d::SSM2D, data::NamedTuple) = sum(logpdf.(d, data...))

"""
    pdf(d::SSM2D, data::NamedTuple) 

Computes the probability density for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
pdf(d::SSM2D, data::NamedTuple) = pdf.(d, data.choice, data.rt)


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

abstract type AbstractaDDM <: SSM2D end

abstract type AbstractLBA <: SSM2D end 

abstract type AbstractWald <: SSM1D end

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
logpdf(d::SSM2D, data::NamedTuple; kwargs...) = logpdf.(d, data.choice, data.rt; kwargs...)
logpdf(d::SSM2D, data::Vector{Real}; kwargs...) = logpdf(d, Int(data[1]), data[2]; kwargs...) 

"""
    loglikelihood(d::SSM2D, data::NamedTuple) 

Computes the summed log likelihood for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
loglikelihood(d::SSM2D, data::NamedTuple) = sum(logpdf.(d, data...))

"""
    pdf(d::SSM2D, data::NamedTuple) 

Computes the probability density for a 2D sequential sampling model. 

# Arguments

- `d::SSM2D`: an object for a 2D sequential sampling model 
- `data::NamedTuple`: a NamedTuple of data containing choice and reaction time 
"""
pdf(d::SSM2D, data::NamedTuple) = pdf.(d, data.choice, data.rt)


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

function Base.show(io::IO, ::MIME"text/plain", model::SSM1D)
    return _show(io::IO, model)
end

function Base.show(io::IO, ::MIME"text/plain", model::SSM2D)
    return _show(io::IO, model)
end

function Base.show(io::IO, ::MIME"text/plain", model::ContinuousMultivariateSSM)
    return _show(io::IO, model)
end

function _show(io::IO, model)
    values = [getfield(model, f) for f in fieldnames(typeof(model))]
    values = map(x -> typeof(x) == Bool ? string(x) : x, values)
    T = typeof(model)
    model_name = string(T.name.name)
    return pretty_table(io,
        values;
        title=model_name,
        row_label_column_title="Parameter",
        compact_printing=false,
        header=["Value"],
        row_label_alignment=:l,
        row_labels=[fieldnames(typeof(model))...],
        formatters=ft_printf("%5.2f"),
        alignment=:l,
    )
end

"""
    compute_quantiles(data::Vector{<:Real}; percentiles=.1:.1:.90)

Returns the quantiles associated with a vector of reaction times for a single choice SSM.

- `data::Vector{<:Real}`: a vector of reaction times     
# Keywords 

- `percentiles=.1:.1:.90`: percentiles at which to evaluate the quantiles 
"""
function compute_quantiles(data::Vector{<:Real}; percentiles=.1:.1:.90)
    return quantile(data, percentiles)
end

"""
    compute_quantiles(data::NamedTuple; choice_set=unique(data.choice), percentiles=.1:.1:.90)

Returns the quantiles for each choice of a 2D SSM. Note there is a chance that a given choice will 
have no observations, and thus no quantiles. Such cases will need to be removed or handled in post processing.

# Arguments

- `data::NamedTuple`: a data structure containing discrete choices in the key `choice` and corresponding 
reaction times in key `rt`

# Keywords 

- `percentiles=.1:.1:.90`: percentiles at which to evaluate the quantiles 
- `choice_set=unique(choice)`: a vector of possible choices. 
"""
function compute_quantiles(data::NamedTuple; choice_set=unique(data.choice), percentiles=.1:.1:.90)
    (;choice, rt) = data
    n_choices = length(choice_set)
    quantiles = Vector{typeof(rt)}(undef,n_choices)
    for c ∈ 1:n_choices
        temp_rts = rt[choice .== choice_set[c]]
        isempty(temp_rts) ? (quantiles[c] = temp_rts; continue) : nothing 
        quantiles[c] = quantile(temp_rts, percentiles)
    end
    return quantiles
end

"""
    compute_quantiles(data::Array{<:Real,2}; percentiles=.1:.1:.90)

Returns the marginal quantiles for a continous multivariate SSM. 

- `data::Array{<:Real,2}`: an array of continous observations

# Keywords 

- `percentiles=.1:.1:.90`: percentiles at which to evaluate the quantiles 
"""
function compute_quantiles(data::Array{<:Real,2}; percentiles=.1:.1:.90)
    return map(c -> quantile(data[:,c], percentiles), 1:size(data, 2))
end

"""
    compute_choice_probs(data::NamedTuple; choice_set=unique(data.choice))

Returns the choice probabilities for a 2D SSM. 

# Arguments

- `data::NamedTuple`: a data structure containing discrete choices in the key `choice` and corresponding 
reaction times in key `rt`

# Keywords 

- `choice_set: a vector of possible choices. 
"""
function compute_choice_probs(data::NamedTuple; choice_set)
    (;choice,) = data
    n_choices = length(choice_set)
    probs = fill(0.0, n_choices)
    for c ∈ 1:n_choices
        probs[c] = mean(choice .== choice_set[c])
    end
    return probs
end

Φ(x) = cdf(Normal(0, 1), x)
ϕ(x) = pdf(Normal(0, 1), x)