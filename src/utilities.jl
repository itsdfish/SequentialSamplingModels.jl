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
    return pretty_table(
        io,
        values;
        title = model_name,
        row_label_column_title = "Parameter",
        compact_printing = false,
        header = ["Value"],
        row_label_alignment = :l,
        row_labels = [fieldnames(typeof(model))...],
        formatters = ft_printf("%5.2f"),
        alignment = :l,
    )
end

"""
    compute_quantiles(data::Vector{<:Real}; percentiles=.1:.1:.90)

Returns the quantiles associated with a vector of reaction times for a single choice SSM.

- `data::Vector{<:Real}`: a vector of reaction times     
# Keywords 

- `percentiles=.1:.1:.90`: percentiles at which to evaluate the quantiles 
"""
function compute_quantiles(data::Vector{<:Real}; percentiles = 0.1:0.1:0.90)
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
function compute_quantiles(
    data::NamedTuple;
    choice_set = unique(data.choice),
    percentiles = 0.1:0.1:0.90,
)
    (; choice, rt) = data
    n_choices = length(choice_set)
    quantiles = Vector{typeof(rt)}(undef, n_choices)
    for c ∈ 1:n_choices
        temp_rts = rt[choice.==choice_set[c]]
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
function compute_quantiles(data::Array{<:Real,2}; percentiles = 0.1:0.1:0.90)
    return map(c -> quantile(data[:, c], percentiles), 1:size(data, 2))
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
    (; choice,) = data
    n_choices = length(choice_set)
    probs = fill(0.0, n_choices)
    for c ∈ 1:n_choices
        probs[c] = mean(choice .== choice_set[c])
    end
    return probs
end

Φ(x) = cdf(Normal(0, 1), x)
ϕ(x) = pdf(Normal(0, 1), x)
function sample_condition_parms(c_dists, parm_names, g_values, cond_idx)
    parm_vals = map(d -> rand(d[cond_idx]), values(c_dists))
    parm_vals = (parm_vals..., g_values...)
    return NamedTuple{parm_names}(parm_vals)
end

function sample_parms(dists)
    return parm_vals = map(d -> rand(d), values(dists))
end

function sample_parms(dists, parm_names)
    parm_vals = map(d -> rand(d), values(dists))
    return NamedTuple{parm_names}(parm_vals)
end

function sample_parms(c_dists, g_dists, parm_names)
    g_values = sample_parms(g_dists)
    return [
        sample_condition_parms(c_dists, parm_names, g_values, c) for c ∈ CartesianIndices(c_dists[1])
    ]
end

function sample_subject_parms(
    model::Type{<:SSM2D}, 
    dists, 
    cond_dists, 
    n_subjects, 
)
    parm_names = (keys(cond_dists)..., keys(dists)...,)

    return [sample_parms(cond_dists, dists, parm_names) for s ∈ 1:n_subjects]
end

function sample_subject_parms(
    model::Type{<:SSM2D}, 
    dists, 
    n_subjects, 
)
    parm_names = keys(dists)

    return [sample_parms(dists, parm_names) for s ∈ 1:n_subjects]
end

function simulate_multilevel(
    model::Type{<:SSM2D}, 
    subject_parms::AbstractArray{<:AbstractArray}, 
    n_subjects,
    n_trials; 
    factor_names = Tuple(map(s -> Symbol("factor_$s"), 1:ndims(subject_parms[1]))),
    subj_start_index = 1
)
    n_total = sum(n_subjects .* n_trials)
    parm_names = keys(dists)
    choice = fill(0, n_total)
    condition = fill(0, n_total, ndims(subject_parms[1])) 
    subject_id = fill(0, n_total)
    condition_indices = CartesianIndices(subject_parms[1])
    rt = fill(0.0, n_total)
    cnt = 1
    subj_end_index = n_subjects + subj_start_index - 1
    for s ∈ subj_start_index:subj_end_index
        cond_idx = 1
        for c ∈ condition_indices
            parms = subject_parms[s][cond_idx]
            dist = model(; parms...)
            for t ∈ 1:n_trials[cond_idx] 
                choice[cnt], rt[cnt] = rand(dist)
                condition[cnt,:] .= c.I
                subject_id[cnt] = s
                cnt += 1
            end
            cond_idx += 1
        end
    end
    n_factors = length(factor_names) + 1
    factors = (factor_names[i] => condition[:,n_factors - i] for i in 1:length(factor_names))
    return (;factors..., subject_id, choice, rt)
end

function simulate_multilevel(
    model::Type{<:SSM2D}, 
    subject_parms::Vector{<:NamedTuple}, 
    n_subjects,
    n_trials; 
    subj_start_index = 1
)
    n_total = sum(n_subjects .* n_trials)
    choice = fill(0, n_total)
    subject_id = fill(0, n_total)
    rt = fill(0.0, n_total)
    cnt = 1
    subj_end_index = n_subjects + subj_start_index - 1
    for s ∈ subj_start_index:subj_end_index
        parms = subject_parms[s]
        dist = model(; parms...)
        for t ∈ 1:n_trials 
            choice[cnt], rt[cnt] = rand(dist)
            subject_id[cnt] = s
            cnt += 1
        end
    end
    return (;subject_id, choice, rt)
end

function simulate_multilevel(
    model::Type{<:SSM2D}, 
    dists::NamedTuple, 
    n_subjects,
    n_trials; 
    subj_start_index = 1
)
    subject_parms = sample_subject_parms(model, dists, n_subjects) 
    return simulate_multilevel(model, subject_parms, n_subjects, n_trials; subj_start_index)
end

function simulate_multilevel(
    model::Type{<:SSM2D}, 
    dists::NamedTuple, 
    cond_dists,
    n_subjects,
    n_trials; 
    subj_start_index = 1
)
    subject_parms = sample_subject_parms(model, dists, cond_dists, n_subjects) 
    return simulate_multilevel(model, subject_parms, n_subjects, n_trials; subj_start_index)
end