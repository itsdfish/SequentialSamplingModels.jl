"""
    animate(
        model;
        add_density = false,
        density_kwargs = (),
        labels = get_default_labels(model),
        density_scale = compute_threshold(model),
        model_args = (),
        model_kwargs = (),
        kwargs...
    )
 
Animates the evidence accumulation process of the specified model. 

# Arguments

- `model`: a generic object representing an SSM

# Keywords 

- `add_density=false`: add density plot above threshold line if true 
- `density_kwargs=()`: pass optional keyword arguments to density plot 
- `labels = get_default_labels(model)`: a vector of parameter label options 
- `density_scale = compute_threshold(model)`: scale the maximum height of the density
- `file_path = giffn()`: the path in which the animation is saved. By default, it is saved to a temporary folder 
    called `tmp` using `giffn`.
- `fps = 30`: speed of animation in terms of frames per second 
- `model_args = ()`: optional positional arguments passed to the `rand` and `simulate`
- `model_kwargs = ()`: optional keyword arguments passed to the `rand` and `simulate`
- `t_range`: the range of time points over which the probability density is plotted 
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function animate(
    model;
    add_density = false,
    density_kwargs = (),
    labels = get_default_labels(model),
    density_scale = compute_threshold(model),
    file_path = giffn(),
    fps = 30,
    model_args = (),
    model_kwargs = (),
    t_range = default_range(model),
    kwargs...
)

    times, evidence = simulate(model, model_args...; model_kwargs...)
    y_min = minimum(evidence)

    n_subplots = n_options(model)
    defaults = get_model_plot_defaults(model)
    α = compute_threshold(model)
    ylims = (0, maximum(α))
    xlims = (0, max(maximum(times) + model.τ * 1.1, maximum(t_range)))

    animation = @animate for i ∈ 1:length(times)
        model_plot = plot(; defaults..., kwargs...)
        add_starting_point!(model, model_plot)

        add_threshold!(model, model_plot)
        for s ∈ 1:n_subplots
            annotate!(labels, subplot = s)
        end

        model_plot = plot(
            model_plot,
            times[1:i] .+ model.τ,
            evidence[1:i,:];
            xlims,
            ylims,
            defaults...,
            kwargs...
        )

        if add_density
            add_density!(
                model,
                model_plot;
                model_args,
                model_kwargs,
                density_scale,
                t_range,
                density_kwargs...
            )
        end
    end
    return gif(animation, file_path; fps)
end

"""
    animate(
        model::ContinuousMultivariateSSM;
        add_density = false,
        density_kwargs = (),
        labels = get_default_labels(model),
        kwargs...
    )

Animates the evidence accumulation process of a continous multivariate sequential sampling model.

# Arguments

- `model::ContinuousMultivariateSSM`: a continous multivariate sequential sampling model

# Keywords 

- `add_density=false`: add density plot above threshold line if true 
- `density_kwargs=()`: pass optional keyword arguments to density plot 
- `labels = get_default_labels(model)`: a vector of parameter label options 
- `file_path = giffn()`: the path in which the animation is saved. By default, it is saved to a temporary folder 
    called `tmp` using `giffn`.
- `fps = 90`: speed of animation in terms of frames per second 
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function animate(
    model::ContinuousMultivariateSSM;
    add_density = false,
    density_kwargs = (),
    labels = get_default_labels(model),
    file_path = giffn(),
    fps = 90,
    t_range = default_range(model),
    kwargs...
)
    defaults = get_model_plot_defaults(model)
    times, evidence = simulate(model)
    animation = @animate for i ∈ 1:4:length(times)
        model_plot = plot(evidence[1:i, 1], evidence[1:i, 2], line_z = times; defaults..., kwargs...)
        add_threshold!(model, model_plot)
    end
    return gif(animation, file_path; fps)
end