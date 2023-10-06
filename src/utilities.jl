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
        row_name_column_title="Parameter",
        compact_printing=false,
        header=["Value"],
        row_name_alignment=:l,
        row_names=[fieldnames(typeof(model))...],
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
        isempty(temp_rts) ? (continue) : nothing 
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