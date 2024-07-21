"""
    plot_quantiles(
        q_data::AbstractVector{<:AbstractVector},
        q_preds::Matrix{<:AbstractVector};
        kwargs...
    )
        
Plots the predictive quantile distribution against the quantiles of the data for multi-choice SSMs.

# Arguments

- `q_data::AbstractVector{<:AbstractVector}`: a vector containing sub-vectors of quantiles for each choice distribution 
- `q_preds::Matrix{<:AbstractVector}`: a matrix containing predictive samples of quantiles 

# Keywords

- `kwargs...`: optional keyword arguments to configure the plot 
"""
function plot_quantiles(
    q_data::AbstractVector{<:AbstractVector},
    q_preds::Matrix{<:AbstractVector};
    kwargs...
)
    n_choices = length(q_preds[1])
    return plot_quantiles!(
        plot(layout = (n_choices, 1); kwargs...),
        q_data,
        q_preds;
        kwargs...
    )
end

"""
    plot_quantiles!(
        cur_plot::Plots.Plot,
        q_data::AbstractVector{<:AbstractVector},
        q_preds::Matrix{<:AbstractVector};
        kwargs...
    )

Adds to an existing plot the predictive quantile distribution against the quantiles of the data for multi-choice SSMs.

# Arguments

- `cur_plot::Plots.Plot`: a plot 
- `q_data::AbstractVector{<:AbstractVector}`: a vector containing sub-vectors of quantiles for each choice distribution 
- `q_preds::Matrix{<:AbstractVector}`: a matrix containing predictive samples of quantiles 

# Keywords

- `kwargs...`: optional keyword arguments to configure the plot 
"""
function plot_quantiles!(
    cur_plot::Plots.Plot,
    q_data::AbstractVector{<:AbstractVector},
    q_preds::Matrix{<:AbstractVector};
    kwargs...
)
    n_choices = length(q_preds[1])
    _q_preds = map(r -> map(i -> q_preds[i][r], 1:length(q_preds)), 1:n_choices)
    for c ∈ 1:n_choices
        filter!(!isempty, _q_preds[c])
        plot_quantiles!(
            cur_plot,
            q_data[c],
            _q_preds[c];
            title = "choice $c",
            subplot = c,
            kwargs...
        )
    end
    return cur_plot
end

"""
    plot_quantiles(
        q_data::AbstractVector,
        q_preds::AbstractArray{<:AbstractVector};
        kwargs...
    )

Plots the predictive quantile distribution against the quantiles of the data for single choice SSMs.

# Arguments

- `q_data::AbstractVector`: a vector containing quantiles
- `q_preds::Matrix{<:AbstractVector}`: a matrix containing predictive samples of quantiles 

# Keywords

- `kwargs...`: optional keyword arguments to configure the plot 
"""
function plot_quantiles(
    q_data::AbstractVector,
    q_preds::AbstractArray{<:AbstractVector};
    kwargs...
)
    return plot_quantiles!(plot(), q_data, q_preds; kwargs...)
end

"""
    plot_quantiles!(
        cur_plot::Plots.Plot,
        q_data::AbstractVector,
        q_preds::AbstractArray{<:AbstractVector};
        kwargs...
    )
        
Adds to an existing plot the predictive quantile distribution against the quantiles of the data for single choice SSMs.

# Arguments

- `q_data::AbstractVector`: a vector containing quantiles
- `q_preds::Matrix{<:AbstractVector}`: a matrix containing predictive samples of quantiles 

# Keywords

- `kwargs...`: optional keyword arguments to configure the plot 
"""
function plot_quantiles!(
    cur_plot::Plots.Plot,
    q_data::AbstractVector,
    q_preds::AbstractArray{<:AbstractVector};
    kwargs...
)
    q_mat = reduce(vcat, transpose.(q_preds))
    lb = map(c -> quantile(q_mat[:, c], 0.025), 1:size(q_mat, 2))
    ub = map(c -> quantile(q_mat[:, c], 0.975), 1:size(q_mat, 2))
    plot!(
        cur_plot,
        q_data,
        q_data,
        color = :black,
        line_style = :dash,
        leg = false,
        grid = false,
        yerror = (q_data .- lb, ub .- q_data),
        xlabel = "Quantile Data",
        ylabel = "Quantile Model";
        kwargs...
    )
    return cur_plot
end

function plot_quantiles!(
    q_data::AbstractVector,
    q_preds::Matrix{<:AbstractVector};
    kwargs...
)
    return plot_quantiles!(Plots.current(), q_preds, q_data; kwargs...)
end
