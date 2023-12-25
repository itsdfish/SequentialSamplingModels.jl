"""
    histogram(d::SSM2D;  kwargs...)

Plots the histogram of a multi-alternative sequential sampling model.

# Arguments

- `d::SSM2D`: a model object for a mult-alternative sequential sampling model 

# Keywords 

- `norm=true`: histogram scaled according to density if true 
- `m_args=()`: optional positional arguments passed to `rand`
- `n_sim=2000`: the number of data points used in the histogram
- `kwargs...`: optional keyword arguments for configuring  plot options
"""
function histogram(d::SSM2D; norm=true, n_sim=2000, kwargs...)
    return ssm_histogram(d; norm, n_sim, kwargs...)
end

function histogram(d::ContinuousMultivariateSSM; norm=true, n_sim=2000, kwargs...)
    return ssm_histogram(d; norm, n_sim, kwargs...)
end

function ssm_histogram(d::ContinuousMultivariateSSM; norm, m_args=(), n_sim, kwargs...)
    n_subplots = n_options(d)
    data = rand(d, n_sim, m_args...)
    data_vecs = map(c -> data[:, c], 1:n_subplots)
    yaxis = norm ? "density" : "frequency"
    defaults = get_histogram_defaults(d)
    hist = histogram(data_vecs; norm, defaults..., kwargs...)
    #ymax = get_y_max(hist, n_subplots) 
    plot!(hist; kwargs...)
    #norm ? scale_density!(hist, choice_probs, 1, n_subplots) : nothing 
    return hist
end

function ssm_histogram(d::SSM2D; norm, m_args=(), n_sim, kwargs...)
    n_subplots = n_options(d)
    choices, rts = rand(d, n_sim, m_args...)
    qq = quantile(rts, .99)
    idx = rts .≤ qq
    choices = choices[idx]
    rts = rts[idx]
    choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
    rt_vecs = map(c -> rts[choices .== c], 1:n_subplots)
    yaxis = norm ? "density" : "frequency"
    defaults = get_histogram_defaults(d)
    hist = histogram(rt_vecs; norm, defaults..., kwargs...)
    ymax = get_y_max(hist, n_subplots) 
    plot!(hist, ylims=(0, ymax); kwargs...)
    norm ? scale_density!(hist, choice_probs, 1, n_subplots) : nothing 
    return hist
end

"""
    histogram(d::SSM1D; norm=true, n_sim=2000, kwargs...)

Plots the histogram of a single alternative sequential sampling model.

# Arguments

- `d::SSM1D`: a model object for a single alternative sequential sampling model 

# Keywords 

- `norm=true`: histogram scaled according to density if true 
- `m_args=()`: optional positional arguments passed to `rand`
- `n_sim=2000`: the number of data points used in the histogram
- `kwargs...`: optional keyword arguments for configuring  plot options
"""
function histogram(d::SSM1D; norm=true, n_sim=2000, kwargs...)
    return ssm_histogram(d; norm, n_sim, kwargs...)
end

function ssm_histogram(d::SSM1D; m_args=(), norm, n_sim, kwargs...)
    n_subplots = n_options(d)
    rts = rand(d, n_sim, m_args...)
    qq = quantile(rts, .99)
    filter!(x -> x < qq, rts)
    yaxis = norm ? "density" : "frequency"
    defaults = get_histogram_defaults(d)
    hist = histogram(rts; norm, defaults..., kwargs...)
    return hist
end

histogram!(d::SSM2D; norm=true, n_sim=2000, kwargs...) = histogram!(Plots.current(), d; norm, n_sim, kwargs...)

histogram!(d::SSM1D; norm=true, n_sim=2000, kwargs...) = histogram!(Plots.current(), d; norm, n_sim, kwargs...)

histogram!(d::ContinuousMultivariateSSM; norm=true, n_sim=2000, kwargs...) = histogram!(Plots.current(), d; norm, n_sim, kwargs...)

"""
    histogram!([cur_plot], d::SSM2D;  kwargs...)

Adds histogram of a multi-alternative sequential sampling model to an existing plot

# Arguments
- `cur_plot`: optional current plot
- `d::SSM2D`: a model object for a mult-alternative sequential sampling model 

# Keywords 

- `norm=true`: histogram scaled according to density if true 
- `m_args=()`: optional positional arguments passed to `rand`
- `n_sim=2000`: the number of data points used in the histogram
- `kwargs...`: optional keyword arguments for configuring  plot options
"""
function histogram!(cur_plot::Plots.Plot, d::SSM2D; norm=true, n_sim=2000, kwargs...)
    return ssm_histogram!(d, cur_plot; norm, n_sim, kwargs...)
end

"""
    histogram!([cur_plot], d::ContinuousMultivariateSSM;  kwargs...)

Adds histogram of a multi-alternative sequential sampling model to an existing plot

# Arguments

- `cur_plot`: optional current plot
- `d::SSM2D`: a model object for a mult-alternative sequential sampling model 

# Keywords 

- `norm=true`: histogram scaled according to density if true 
- `m_args=()`: optional positional arguments passed to `rand`
- `n_sim=2000`: the number of data points used in the histogram
- `kwargs...`: optional keyword arguments for configuring  plot options
"""
function histogram!(cur_plot::Plots.Plot, d::ContinuousMultivariateSSM; norm=true, n_sim=2000, kwargs...)
    return ssm_histogram!(d, cur_plot; norm, n_sim, kwargs...)
end

function ssm_histogram!(d::SSM2D, cur_plot; m_args=(), norm, n_sim, kwargs...)
    n_subplots = n_options(d)
    choices, rts = rand(d, n_sim, m_args...)
    qq = quantile(rts, .99)
    idx = rts .≤ qq
    choices = choices[idx]
    rts = rts[idx]
    choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
    rt_vecs = map(c -> rts[choices .== c], 1:n_subplots)
    yaxis = norm ? "density" : "frequency"
    defaults = get_histogram_defaults(d)
    hist = histogram!(cur_plot, rt_vecs; norm, yaxis, defaults..., kwargs...)
    norm ? scale_density!(hist, choice_probs, 2, n_subplots) : nothing 
    histogram!(hist; kwargs...)
    return hist
end

function ssm_histogram!(d::ContinuousMultivariateSSM, cur_plot; m_args=(), norm, n_sim, kwargs...)
    n_subplots = n_options(d)
    data = rand(d, n_sim, m_args...)
    data_vecs = map(c -> data[:, c], 1:n_subplots)
    yaxis = norm ? "density" : "frequency"
    defaults = get_histogram_defaults(d)
    hist = histogram(cur_plot, data_vecs; norm, defaults..., kwargs...)
    #norm ? scale_density!(hist, choice_probs, 2, n_subplots) : nothing 
    histogram!(hist; kwargs...)
    return hist
end

"""
    histogram([cur_plot], d::SSM1D; norm=true, n_sim=2000, kwargs...)

Adds histogram of a single alternative sequential sampling model to an existing plot

# Arguments

- `cur_plot`: optional current plot
- `d::SSM1D`: a model object for a single alternative sequential sampling model 

# Keywords 

- `norm=true`: histogram scaled according to density if true 
- `m_args=()`: optional positional arguments passed to `rand`
- `n_sim=2000`: the number of data points used in the histogram
- `kwargs...`: optional keyword arguments for configuring  plot options
"""
function histogram!(cur_plot::Plots.Plot, d::SSM1D; norm=true, n_sim=2000, kwargs...)
    return ssm_histogram!(d, cur_plot; norm, n_sim, kwargs...)
end

function ssm_histogram!(d::SSM1D, cur_plot; norm, n_sim, m_args=(), kwargs...)
    n_subplots = n_options(d)
    rts = rand(d, n_sim, m_args...)
    qq = quantile(rts, .99)
    filter!(x -> x ≤ qq, rts)
    yaxis = norm ? "density" : "frequency"
    defaults = get_histogram_defaults(d)
    return histogram!(cur_plot, rts; norm, yaxis, defaults..., kwargs...)
end

function get_y_max(hist, n_options)
    dens = mapreduce(i -> hist[i][1][:y], vcat, 1:n_options)
    dens = filter(!isnan, dens)
    return maximum(dens)
end

function scale_density!(hist, probs, id, n_options)
    for i ∈ 1:n_options
        hist[i][id][:y] .*= probs[i]
    end
    return nothing
end

function get_histogram_defaults(d)
    n_subplots = n_options(d)
    title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
    return (xaxis=("RT [s]"), yaxis = "density", grid=false,
        color = :grey, leg=false, title, layout=(n_subplots,1))
end

function get_histogram_defaults(d::AbstractCDDM)
    n_subplots = n_options(d)
    xlabel = fill("angle", n_subplots-1)
    push!(xlabel, "RT [s]")
    xlabel = reshape(xlabel, 1, n_subplots)
    return (;xlabel, yaxis = "density", grid=false,
        color = :grey, leg=false, layout=(n_subplots,1))
end