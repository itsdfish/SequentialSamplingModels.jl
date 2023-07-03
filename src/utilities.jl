abstract type Mixed <: ValueSupport end 

"""
    MixedMultivariateDistribution   

An abstract distribution type for multivariate distributions with mixed value support. 
"""
const MixedMultivariateDistribution = Distribution{Multivariate, Mixed}

"""
    SSM1D <: ContinuousUnivariateDistribution

An abstract type for sequential sampling models characterized by a single choice reaction time distribution.
Sub-types of `SSM1D` output a vector of reaction times.
"""
abstract type SSM1D <: ContinuousUnivariateDistribution end 

"""
    SSM2D <: MixedMultivariateDistribution

An abstract type for sequential sampling models characterized by a multivariate choice-reaction time distribution.
Sub-types of `SSM2D` output a `NamedTuple` consisting of a vector of choices and reaction times. 
"""
abstract type SSM2D <: MixedMultivariateDistribution end 

abstract type AbstractWald <: SSM1D end

abstract type AbstractaDDM <: SSM2D end

minimum(d::SSM1D) = 0.0
maximum(d::SSM1D) = Inf

minimum(d::SSM2D) = 0.0
maximum(d::SSM2D) = Inf

insupport(d::SSM2D, data) = data.rt ≥ minimum(d) && data.rt ≤ maximum(d)

Base.broadcastable(x::SSM1D) = Ref(x)
Base.broadcastable(x::MixedMultivariateDistribution) = Ref(x)

vectorize(d::MixedMultivariateDistribution, r::NamedTuple) = [r...]

Base.length(d::MixedMultivariateDistribution) = 2

rand(d::MixedMultivariateDistribution) = rand(Random.default_rng(), d)
rand(d::MixedMultivariateDistribution, n::Int) = rand(Random.default_rng(), d, n)

logpdf(d::MixedMultivariateDistribution, data::NamedTuple) = logpdf(d, data.choice, data.rt)

loglikelihood(d::MixedMultivariateDistribution, data::NamedTuple) = sum(logpdf.(d, data...))

logpdf(d::SSM2D, data::Vector{Real}) = logpdf(d, Int(data[1]), data[2]) 

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

function Base.show(io::IO, ::MIME"text/plain", model::SSM1D)
    values = [getfield(model, f) for f in fieldnames(typeof(model))]
    values = map(x -> typeof(x) == Bool ? string(x) : x, values)
    T = typeof(model)
    model_name = string(T.name.name)
    return pretty_table(io,
        values;
        title=model_name,
        row_name_column_title="Parameter",
        compact_printing=false,
        header=["Value"],
        row_name_alignment=:l,
        row_names=[fieldnames(typeof(model))...],
        formatters=ft_printf("%5.2f"),
        alignment=:l,
    )
end

function Base.show(io::IO, ::MIME"text/plain", model::SSM2D)
    values = [getfield(model, f) for f in fieldnames(typeof(model))]
    values = map(x -> typeof(x) == Bool ? string(x) : x, values)
    T = typeof(model)
    model_name = string(T.name.name)
    return pretty_table(io,
        values;
        title=model_name,
        row_name_column_title="Parameter",
        compact_printing=false,
        header=["Value"],
        row_name_alignment=:l,
        row_names=[fieldnames(typeof(model))...],
        formatters=ft_printf("%5.2f"),
        alignment=:l,
    )
end