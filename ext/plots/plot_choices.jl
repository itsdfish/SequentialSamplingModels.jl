"""
    plot_choices(
        data::AbstractVector{<:Real},
        preds::AbstractArray{<:AbstractVector};
        kwargs...
    )

Plots choice probability distributions for multi-choice SSMs.

# Arguments

- `data::AbstractVector{<:Real}`: a vector of observed choice proportions
- `preds::AbstractArray{<:AbstractVector}`: an array containing vectors of choice probabilities

# Keywords

- `kwargs...`: optional keyword arguments to configure the plot 
"""
function plot_choices(
    data::AbstractVector{<:Real},
    preds::AbstractArray{<:AbstractVector};
    kwargs...
)
    n_choices = length(preds[1])
    return plot_choices!(plot(layout = (n_choices, 1); kwargs...), data, preds; kwargs...)
end

"""
    plot_choices!(
        cur_plot::Plots.Plot,
        data::AbstractVector{<:Real},
        preds::AbstractArray{<:AbstractVector};
        kwargs...
    )

Adds to a current plot choice probability distributions for multi-choice SSMs.

# Arguments

- `data::AbstractVector{<:Real}`: a vector of observed choice proportions
- `preds::AbstractArray{<:AbstractVector}`: an array containing vectors of choice probabilities

# Keywords

- `kwargs...`: optional keyword arguments to configure the plot 
"""
function plot_choices!(
    cur_plot::Plots.Plot,
    data::AbstractVector{<:Real},
    preds::AbstractArray{<:AbstractVector};
    kwargs...
)
    n_choices = length(preds[1])
    title = ["Choice $c" for _ ∈ 1:1, c ∈ 1:n_choices]
    _preds = reduce(vcat, transpose.(preds))
    histogram!(
        cur_plot,
        _preds,
        xlims = (0, 1),
        leg = false,
        title = title,
        xlabel = "Choice Probability",
        ylabel = "Density",
        grid = false,
        norm = true;
        kwargs...
    )
    vline!(data', color = :black, linestyle = :dash)
    return cur_plot
end

function plot_choices!(
    data::AbstractVector{<:Real},
    preds::AbstractArray{<:AbstractArray};
    kwargs...
)
    return plot_choices!(Plots.current(), data, preds; kwargs...)
end
