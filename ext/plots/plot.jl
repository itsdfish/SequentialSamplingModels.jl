"""
    plot(d::SSM2D; t_range=default_range(d), kwargs...)

Plots the probability density of a multi-alternative sequential sampling model.

# Arguments

- `d::SSM2D`: a model object for a mult-alternative sequential sampling model 

# Keywords 

- `t_range`: the range of time points over which the probability density is plotted 
- `kwargs...`: optional keyword arguments for configuring  plot options
"""
function plot(d::SSM2D; t_range=default_range(d), kwargs...)
    return ssm_plot(get_pdf_type(d), d; t_range, kwargs...)
end

"""
    plot(d::SSM1D; t_range=default_range(d), kwargs...)

Plots the probability density of a single alternative sequential sampling model.

# Arguments

- `d::SSM1D`: a model object for a single alternative sequential sampling model 

# Keywords 

- `t_range`: the range of time points over which the probability density is plotted 
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function plot(d::SSM1D; t_range=default_range(d), kwargs...)
    return ssm_plot(get_pdf_type(d), d; t_range, kwargs...)
end

"""
    plot(d::ContinuousMultivariateSSM; t_range=default_range(d), kwargs...)

Plots the marginal probability density of each N dimensional continuous sequential samping model.

# Arguments

- `d::ContinuousMultivariateSSM`: a model object for a N dimensional continuous sequential samping model.

# Keywords 

- `t_range`: the range of time points over which the probability density is plotted 
- `model_args=()`: optional positional arguments passed to `rand` if applicable
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function plot(d::ContinuousMultivariateSSM; t_range=default_range(d), kwargs...)
    return ssm_plot(get_pdf_type(d), d; t_range, kwargs...)
end

function ssm_plot(
        ::Type{<:Exact}, 
        d::ContinuousMultivariateSSM; 
        density_offset = 0, 
        density_scale = nothing, 
        model_args = (),
        model_kwargs = (),
        t_range, 
        kwargs...
    )
    n_subplots = n_options(d)
    pds = gen_pds(d, t_range, n_subplots; model_args, model_kwargs) 
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    ymax = maximum(vcat(pds...)) * 1.2
    defaults = get_plot_defaults(d)
    return plot(t_range, pds; defaults..., kwargs...)
end

function ssm_plot(
        ::Type{<:Exact}, 
        d; 
        model_args = (),
        model_kwargs = (),
        density_offset = 0, 
        density_scale = nothing, 
        t_range, 
        kwargs...
    )
    n_subplots = n_options(d)
    pds = gen_pds(d, t_range, n_subplots; model_args, model_kwargs) 
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    ymax = maximum(vcat(pds...)) * 1.2
    defaults = get_plot_defaults(d)
    return plot(t_range, pds; ylims = (0,ymax), defaults..., kwargs...)
end

function ssm_plot(
        ::Type{<:Approximate}, 
        d;
        density_offset = 0, 
        model_args = (), 
        model_kwargs = (),
        n_sim = 2000, 
        density_scale = nothing, 
        t_range, 
        kwargs...
    )
    n_subplots = n_options(d)
    choices, rts = rand(d, n_sim, model_args...; model_kwargs...)
    choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
    kdes = [kernel(rts[choices .== c]) for c ∈ 1:n_subplots]
    pds = gen_pds(kdes, t_range, choice_probs; model_args, model_kwargs)
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    ymax = maximum(vcat(pds...)) * 1.2
    defaults = get_plot_defaults(d)
    return plot(t_range, pds; ylims = (0,ymax), defaults..., kwargs...)
end

plot!(d::SSM2D; t_range=default_range(d), kwargs...) = plot!(Plots.current(), d; t_range, kwargs...)

plot!(d::SSM1D; t_range=default_range(d), kwargs...) = plot!(Plots.current(), d; t_range, kwargs...)

plot!(d::ContinuousMultivariateSSM; t_range=default_range(d), kwargs...) = plot!(Plots.current(), d; t_range, kwargs...)

"""
    plot!([cur_plot], d::SSM1D; t_range=default_range(d), kwargs...)

Adds the probability density of a single alternative sequential sampling model to an existing plot

# Arguments

- `cur_plot`: optional current plot
- `d::SSM1D`: a model object for a single alternative sequential sampling model 

# Keywords 

- `t_range`: the range of time points over which the probability density is plotted 
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function plot!(cur_plot::Plots.Plot, d::SSM1D; t_range=default_range(d), kwargs...)
    return ssm_plot!(get_pdf_type(d), d, cur_plot; t_range, kwargs...)
end

"""
    plot!(
        cur_plot::Plots.Plot,
        d::SSM2D; 
        t_range = default_range(d),
        model_args = (),
        model_kwargs = (),
        kwargs...
    )

Adds the probability density of a mult-alternative sequential sampling model to an existing plot

# Arguments
- `cur_plot`: optional current plot
- `d::SSM2D`: a model object for a mult-alternative sequential sampling model 

# Keywords 

- `t_range`: the range of time points over which the probability density is plotted 
- `model_args = ()`: optional positional arguments passed to the `rand` if applicable
- `model_kwargs = ()`: optional keyword arguments passed to the `rand` if applicable
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function plot!(
        cur_plot::Plots.Plot,
        d::SSM2D; 
        t_range = default_range(d),
        model_args = (),
        model_kwargs = (),
        kwargs...
    )
    return ssm_plot!(get_pdf_type(d), d, cur_plot; t_range, model_args, model_kwargs, kwargs...)
end

"""
    plot!([cur_plot], d::ContinuousMultivariateSSM; t_range=default_range(d), kwargs...)

Adds the marginal probability density of a multivariate continuous sequential sampling model to an existing plot

# Arguments
- `cur_plot`: optional current plot
- `d::ContinuousMultivariateSSM`: a multivariate continuous sequential sampling model

# Keywords 

- `t_range`: the range of time points over which the probability density is plotted 
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function plot!(cur_plot::Plots.Plot, d::ContinuousMultivariateSSM; t_range=default_range(d), kwargs...)
    return ssm_plot!(get_pdf_type(d), d, cur_plot; t_range, kwargs...)
end

function ssm_plot!(
        ::Type{<:Exact},
        d, cur_plot; 
        model_args = (),
        model_kwargs = (),
        density_offset = 0,
        t_range,
        density_scale = nothing, 
        kwargs...
    )
    n_subplots = n_options(d)
    pds = gen_pds(d, t_range, n_subplots; model_args, model_kwargs)
    scale_density!(pds, density_scale)
    if length(density_offset) > 1
        map!((x,i) -> x .+ density_offset[i], pds, pds, 1:n_subplots)
    else 
        map!(x -> x .+ density_offset, pds, pds)
    end
    ymax = maximum(vcat(pds...)) * 1.2
    defaults = get_plot_defaults(d)
    return plot!(cur_plot, t_range, pds; 
        ylims = (0,ymax), defaults..., kwargs...)
end

function ssm_plot!(
        ::Type{<:Exact},
        d::ContinuousMultivariateSSM,
        cur_plot; 
        model_args = (),
        model_kwargs = (),
        density_offset = 0, 
        density_scale = nothing, 
        t_range, 
        kwargs...
    )
    n_subplots = n_options(d)
    pds = gen_pds(d, t_range, n_subplots; model_args, model_kwargs) 
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    ymax = maximum(vcat(pds...)) * 1.2
    defaults = get_plot_defaults(d)
    return plot!(t_range, pds; defaults..., kwargs...)
end

function ssm_plot!(
        ::Type{<:Approximate},
        d,
        cur_plot; 
        density_offset=0, 
        n_sim = 2000, 
        model_args = (),
        model_kwargs = (),
        density_scale = nothing, 
        t_range, 
        kwargs...
    )
    n_subplots = n_options(d)
    choices, rts = rand(d, n_sim, model_args...; model_kwargs...)
    choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
    kdes = [kernel(rts[choices .== c]) for c ∈ 1:n_subplots]
    pds = gen_pds(kdes, t_range, choice_probs; model_args, model_kwargs)
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    ymax = maximum(vcat(pds...)) * 1.2
    defaults = get_plot_defaults(d)
    return plot!(cur_plot, t_range, pds; ylims = (0,ymax), defaults..., kwargs...)
end

function gen_pds(d::SSM2D, t_range, n_subplots; model_args, model_kwargs)
    return [pdf.(d, (i,), t_range, model_args...,; model_kwargs...) for i ∈ 1:n_subplots]
end

function gen_pds(d::SSM1D, t_range, n_subplots; _...)
    return [pdf.(d, t_range) for i ∈ 1:n_subplots]
end

function gen_pds(d::CDDM, t_range, n_subplots; _...)
    pdfs = (SSMs.pdf_angle,SSMs.pdf_rt)
    return [pdfs[i].(d, t_range[i]) for i ∈ 1:n_subplots]
end

function gen_pds(kdes, t_range, probs; _...)
    return [pdf(kdes[i], t_range) .* probs[i] for i ∈ 1:length(kdes)]
end

function default_range(d)
    n = n_options(d)
    return range(d.τ + eps(), d.τ+ .25 * log(n + 1), length=100)
end

function default_range(d::CDDM)
    (;ν,α) = d
    mag = 3 * log(1. + α / norm(ν))
    return [range(-π, π, length=100),
        range(d.τ + eps(), d.τ + mag, length=100)]
end

function get_plot_defaults(d)
    n_subplots = n_options(d)
    title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
    return (xlabel=("RT [s]"), ylabel = "density", grid=false,
        linewidth = 1.5, color = :black, leg=false, title,
         layout=(n_subplots,1))
end

function get_plot_defaults(d::AbstractCDDM)
    n_subplots = n_options(d)
    xlabel = fill("angle", n_subplots-1)
    push!(xlabel, "RT [s]")
    xlabel = reshape(xlabel, 1, n_subplots )

    return (;xlabel, ylabel = "density", grid=false,
        linewidth = 1.5, color = :black, leg=false,
         layout=(n_subplots,1))
end

scale_density!(pds, x::Nothing) = nothing

"""
    scale_density!(pds, scalar::Number)

Scale the height of the density.

# Arguments

- `pds`: a vector of probability densities 
- `scalar`: a scalar representing the maximum density
"""
function scale_density!(pds, scalar::Number)
    temp = vcat(pds...)
    filter!(!isnan, temp)
    max_dens = maximum(temp)
    pds .*= scalar / max_dens
    return nothing
end

scale_density!(pds, scalar::Vector) = scale_density!(pds, maximum(scalar))