"""
    plot_model(model; 
        add_density = false, 
        density_kwargs = (), 
        labels = get_default_labels(model), 
        density_scale = compute_threshold(model),
        n_sim = 1,
        model_args = (),
        model_kwargs = (), 
        kwargs...
    )

Plot the evidence accumulation process of a generic SSM.

# Arguments

- `model`: a generic object representing an SSM

# Keywords 

- `add_density=false`: add density plot above threshold line if true 
- `density_kwargs=()`: pass optional keyword arguments to density plot 
- `labels = get_default_labels(model)`: a vector of parameter label options 
- `density_scale = compute_threshold(model)`: scale the maximum height of the density
- `n_sim=1`: the number of simulated decision processes per option
- `model_args = ()`: optional positional arguments passed to the `rand` and `simulate`
- `model_kwargs = ()`: optional keyword arguments passed to the `rand` and `simulate`
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function plot_model(model; 
            add_density = false, 
            density_kwargs = (), 
            labels = get_default_labels(model), 
            density_scale = compute_threshold(model),
            n_sim = 1,
            model_args = (),
            model_kwargs = (), 
            kwargs...
        )
    n_subplots = n_options(model)
    defaults = get_model_plot_defaults(model)
    model_plot = plot(; defaults..., kwargs...)
    add_starting_point!(model, model_plot)
    α = compute_threshold(model) 
    zs = Vector{Vector{Float64}}(undef,n_sim)
    for i ∈ 1:n_sim
        time_range,evidence = simulate(model, model_args...; model_kwargs...)
        plot!(model_plot, time_range .+ model.τ, evidence; 
            ylims=(0, maximum(α)), defaults..., kwargs...)
        zs[i] = evidence[1,:][:] 
    end
    add_threashold!(model, model_plot)
    #add_mean_drift_rate(model, model_plot, zs)
    for s ∈ 1:n_subplots
        annotate!(labels, subplot=s)
    end
    if add_density 
        add_density!(model, model_plot; model_args, model_kwargs, density_scale, density_kwargs...)
    end
    return model_plot
end

"""
    plot_model(model; 
        add_density=false, density_kwargs=(), n_sim=1, kwargs...)

Plot the evidence accumulation process of a continous multivariate sequential sampling model.

# Arguments

- `model::ContinuousMultivariateSSM`: a continous multivariate sequential sampling model

# Keywords 

- `add_density=false`: add density plot above threshold line if true 
- `density_kwargs=()`: pass optional keyword arguments to density plot 
- `labels = get_default_labels(model)`: a vector of parameter label options 
- `n_sim=1`: the number of simulated decision processes per option
- `kwargs...`: optional keyword arguments for configuring plot options
"""
function plot_model(model::ContinuousMultivariateSSM;
    add_density = false, 
    density_kwargs = (), 
    labels = get_default_labels(model), 
    n_sim = 1, 
    kwargs...)

    defaults = get_model_plot_defaults(model)
    ts,evidence = simulate(model)
    model_plot = plot(evidence[:,1], evidence[:,2], line_z=ts; 
        defaults..., kwargs...)
    add_threashold!(model, model_plot)
    return model_plot
end

compute_threshold(model) = model.α
compute_threshold(model::AbstractLBA) = model.A + model.k
compute_threshold(model::AbstractRDM) = model.A + model.k

"""
    get_default_labels(model::AbstractRDM)

Generates default parameter labels and locations for threshold and non-decision time 

# Arguments

- `model`: a generic model object
"""
function get_default_labels(model)
    (;τ,α) = model
    return [
        (0,α,text("α", 10, :bottom)),
        (τ/2,0,text("τ",10, :bottom)),
    ]
end

"""
    get_default_labels(model::AbstractLBA)

Generates default parameter labels and locations 

# Arguments

- `model::AbstractLBA`: an object for the linear ballistic accumulator
"""
function get_default_labels(model::AbstractLBA)
    (;τ,A,k) = model
    α = A + k
    return [
        (τ,A,text("A", 10, :right, :top)),
        (0,α,text("α", 10, :right)),
        (τ/2,0,text("τ",10, :bottom)),
    ]
end

"""
    get_default_labels(model::AbstractRDM)

Generates default parameter labels and locations 

# Arguments

- `model::AbstractRDM`: an object for the racing diffusion model
"""
function get_default_labels(model::AbstractRDM)
    (;τ,A,k) = model
    α = compute_threshold(model)
    return [
        (τ,A,text("A", 10, :right, :top)),
        (0,α,text("α", 10, :right)),
        (τ/2,0,text("τ",10, :bottom)),
    ]
end

add_starting_point!(model, model_plot; kwargs...) = nothing

"""
    add_starting_point!(model::AbstractLBA, cur_plot; kwargs...)

Adds a rectangle representing the starting point and non-decision time

# Arguments

- `model::AbstractLBA`: an object representing the linear ballistic accumulator model 
- `model_plot`: a plot object 

# Keywords 

- `kwargs...`: optional keyword arguments for configuring the plot 
"""
function add_starting_point!(model::AbstractLBA, cur_plot; kwargs...)
    (;τ,A) = model 
    for s ∈ 1:n_options(model)
        plot!(cur_plot, [τ, τ],[0, A], subplot=s, color=:black; kwargs...)
        plot!(cur_plot, [0, τ],[A, A], subplot=s, color=:black; kwargs...)
        plot!(cur_plot, [0, 0],[0, A], subplot=s, color=:black; kwargs...)
    end
    return nothing 
end

"""
    add_starting_point!(model::AbstractRDM, cur_plot; kwargs...)

Adds a rectangle representing the starting point and non-decision time

# Arguments

- `model::AbstractRDM`: an object representing the racing diffusion model 
- `model_plot`: a plot object 

# Keywords 

- `kwargs...`: optional keyword arguments for configuring the plot 
"""
function add_starting_point!(model::AbstractRDM, cur_plot; kwargs...)
    (;τ,A) = model 
    for s ∈ 1:n_options(model)
        plot!(cur_plot, [τ, τ],[0, A], subplot=s, color=:black; kwargs...)
        plot!(cur_plot, [0, τ],[A, A], subplot=s, color=:black; kwargs...)
        plot!(cur_plot, [0, 0],[0, A], subplot=s, color=:black; kwargs...)
    end
    return nothing 
end

"""
    add_threashold!(model, model_plot; kwargs...)

Adds a horizonal line reprenting the decision threshold. 

# Arguments

- `model`: an object representing either a `SSM1D` or `SSM2D` model
- `model_plot`: a plot object 

# Keywords 

- `kwargs...`: optional keyword arguments for configuring the plot 
"""
function add_threashold!(model, model_plot; kwargs...)
    α = compute_threshold(model)
    hline!(model_plot, fill(α, 1, n_options(model)), 
        linestyle=:dash, color=:black;  kwargs...)
    return nothing
end

"""
    add_threashold!(model, model_plot; kwargs...)

Adds a horizonal line reprenting the decision threshold. 

# Arguments

- `model::AbstractaDDM`: a subtype of AbstractaDDM
- `model_plot`: a plot object 

# Keywords 

- `kwargs...`: optional keyword arguments for configuring the plot 
"""
function add_threashold!(model::AbstractaDDM, model_plot; kwargs...)
    α = compute_threshold(model)
    hline!(model_plot, fill(α, 1, n_options(model)), 
        linestyle=:dash, color=:black;  kwargs...)
    hline!(model_plot, fill(-α, 1, n_options(model)), 
        linestyle=:dash, color=:black;  kwargs...)
    return nothing
end

function add_threashold!(model::AbstractPoissonRace, model_plot; kwargs...)
    α = compute_threshold(model)
    hline!(model_plot, α', 
        linestyle=:dash, color=:black;  kwargs...)
    return nothing
end

"""
    add_threashold!(model::AbstractCDDM, model_plot; kwargs...)

Adds a circle reprenting the decision threshold of an abstract CDDM.

# Arguments

- `model::AbstractCDDM`: an object representing a circular drift diffusion model 
- `model_plot`: a plot object 

# Keywords 

- `kwargs...`: optional keyword arguments for configuring the plot 
"""
function add_threashold!(model::AbstractCDDM, model_plot; kwargs...)
    plot!(model_plot, circle(0,0, model.α), aspect_ratio=1, color=:black, leg=false; kwargs...)
    return nothing
end

function circle(h, k, r)
    θ = range(0, 2π, length=200)
    return h .+ r * sin.(θ), k .+ r * cos.(θ)
end

"""
    add_threashold!(model, model_plot; kwargs...)

Adds a horizonal line reprenting the decision threshold. 

# Arguments

- `model`: an object representing either a `SSM1D` or `SSM2D` model
- `model_plot`: a plot object 

# Keywords 

- `kwargs...`: optional keyword arguments for configuring the plot 
"""
function add_threashold!(model::DDM, model_plot; kwargs...)
    α = compute_threshold(model)
    hline!(model_plot, fill(α, 1, n_options(model)), 
        linestyle=:dash, color=:black;  kwargs...)

    hline!(model_plot, fill(0, 1, n_options(model)), 
        linestyle=:dash, color=:black;  kwargs...)
    return nothing
end

"""
    get_model_plot_defaults(d::AbstractLCA)

Returns default plot options 

# Arguments

- `d::AbstractLCA`: an object for the leaky competing accumulator
"""
function get_model_plot_defaults(d::AbstractLCA)
    n_subplots = n_options(d)
    title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing,
         grid=false, linewidth = .75, color = :black, leg=false, title, layout=(n_subplots,1))
end

"""
    get_model_plot_defaults(d::AbstractWald)

Returns default plot options 

# Arguments

- `d::AbstractWald`: an object for the Wald model
"""
function get_model_plot_defaults(d::AbstractWald)
    n_subplots = n_options(d)
    title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing,
         grid=false, linewidth = .75, color = :black, leg=false, title, layout=(n_subplots,1))
end

"""
    get_model_plot_defaults(d::AbstractRDM)

Returns default plot options 

# Arguments

- `d::AbstractRDM`: an object for the racing diffusion model
"""
function get_model_plot_defaults(d::AbstractRDM)
    n_subplots = n_options(d)
    title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing, 
        linewidth = .75, color = :black, leg=false, title, layout=(n_subplots,1))
end

"""
    get_model_plot_defaults(d::AbstractLBA)

Returns default plot options 

# Arguments

- `d::AbstractLBA`: an object for the linear ballistic accumulator
- `n_subplots`: the number of subplots (i.e., choices)
"""
function get_model_plot_defaults(d::AbstractLBA)
    n_subplots = n_options(d)
    title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing, grid=false, 
        linewidth = .75, color = :black, leg=false, title, layout=(n_subplots,1), arrow=:closed)
end

"""
    get_model_plot_defaults(d::AbstractCDDM)

Returns default plot options 

# Arguments

- `d::AbstractCDDM`: an object for the linear ballistic accumulator
"""
function get_model_plot_defaults(d::AbstractCDDM)
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing, grid=false, 
        linewidth = .75, c=cgrad([:black,:purple,:darkorange], [.3,.6,.8]), colorbar_title="Time [s]", 
        framestyle=:box)
end

"""
    get_model_plot_defaults(d::AbstractPoissonRace)

Returns default plot options 

# Arguments

- `d::AbstractPoissonRace`: an object for the linear ballistic accumulator
"""
function get_model_plot_defaults(d::AbstractPoissonRace)
    n_subplots = n_options(d)
    title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing, grid=false, 
        linewidth = .75, color = :black, leg=false, title, layout=(n_subplots,1), arrow=:closed)
end

"""
    get_model_plot_defaults(d::DDM)

Returns default plot options 

# Arguments

- `d::DDM`: an object for the diffusion model
"""
function get_model_plot_defaults(d::DDM)
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing, 
        linewidth = .75, color = :black, leg=false, framestyle=:none)
end

"""
    get_model_plot_defaults(d::AbstractaDDM)

Returns default plot options 

# Arguments

- `d::AbstractaDDM`: an object for the diffusion model
"""
function get_model_plot_defaults(d::AbstractaDDM)
    return (xaxis=nothing, yaxis=nothing, xticks=nothing, yticks=nothing, 
        linewidth = .75, color = :black, leg=false, framestyle=:none)
end

function add_density!(model, model_plot; model_args, model_kwargs, density_scale, kwargs...)
    α = compute_threshold(model) 

    plot!(
        model_plot, 
        model; 
        model_args,
        model_kwargs,
        density_scale,
        density_offset = α  .+ .05,
        xlabel = "",
        ylabel = "", 
        xticks = nothing,
        yticks = nothing, 
        kwargs...)
    return nothing 
end

function add_density!(model::AbstractDDM, model_plot; model_args, model_kwargs, density_scale, kwargs...)
    α = compute_threshold(model)
    τ = model.τ 

    plot_top_density!(
        model, 
        model_plot; 
        model_args,
        model_kwargs,
        density_scale,
        density_offset = α  .+ .05,
        xlabel = "",
        ylabel = "", 
        xticks = nothing,
        yticks = nothing, 
        title = "",
        kwargs...)

    plot_bottom_density!(
        model, 
        model_plot;
        model_args,
        model_kwargs, 
        density_scale,
        density_offset = 0  .+ .05,
        xlabel = "",
        ylabel = "", 
        xticks = nothing,
        yticks = nothing, 
        title = "",
        kwargs...)

        plot!(model_plot, [0,0], [0,α], color=:black,)
        plot!(model_plot, [τ,τ], [0,α], color=:black,)
    return nothing 
end

function add_density!(model::AbstractaDDM, model_plot; model_args, model_kwargs, density_scale, kwargs...)
    α = compute_threshold(model)
    τ = model.τ 

    plot_top_density!(
        model, 
        model_plot; 
        model_args,
        model_kwargs,
        density_scale,
        density_offset = α + .05,
        xlabel = "",
        ylabel = "", 
        xticks = nothing,
        yticks = nothing, 
        title = "",
        kwargs...)

    plot_bottom_density!(
        model, 
        model_plot;
        model_args,
        model_kwargs, 
        density_scale,
        density_offset = -α  - .05,
        xlabel = "",
        ylabel = "", 
        xticks = nothing,
        yticks = nothing, 
        title = "",
        kwargs...)

        plot!(model_plot, [0,0], [0,α], color=:black,)
        plot!(model_plot, [τ,τ], [0,-α], color=:black,)
    return nothing 
end

plot_bottom_density!(d::AbstractDDM, cur_plot; kwargs...) = plot_bottom_density!(get_pdf_type(d), d, cur_plot; kwargs...)
plot_top_density!(d::AbstractDDM, cur_plot; kwargs...) = plot_top_density!(get_pdf_type(d), d, cur_plot; kwargs...)

function plot_top_density!(
        ::Type{<:Exact},
        d::AbstractDDM, 
        cur_plot; 
        density_offset = 0, 
        t_range = default_range(d), 
        model_args = (), 
        model_kwargs = (),
        density_scale = nothing,
        kwargs...
    )
    n_subplots = n_options(d)
    pds = gen_pds(d, t_range, n_subplots; model_args, model_kwargs)
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    pds_vec = vcat(pds...)
    filter!(!isnan, pds_vec)
    ymin = minimum(pds_vec) * -1.05
    ymax = maximum(pds_vec) * 1.05
    defaults = get_plot_defaults(d)
    return plot!(cur_plot, t_range, pds[1]; 
        ylims = (ymin,ymax), defaults..., kwargs...)
end

function plot_top_density!(
        ::Type{<:Approximate},
        d::AbstractDDM, 
        cur_plot; 
        n_sim = 2000,
        density_offset = 0, 
        t_range = default_range(d), 
        model_args = (), 
        model_kwargs = (),
        density_scale = nothing,
        kwargs...
    )

    n_subplots = n_options(d)
    choices, rts = rand(d, n_sim, model_args...; model_kwargs...)
    choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
    kdes = [kernel(rts[choices .== c]) for c ∈ 1:n_subplots]
    pds = gen_pds(kdes, t_range, choice_probs; model_args, model_kwargs)
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    pds_vec = vcat(pds...)
    filter!(!isnan, pds_vec)
    defaults = get_plot_defaults(d)
    return plot!(cur_plot, t_range, pds[1]; 
         defaults..., kwargs...)
end

function plot_bottom_density!(
        ::Type{<:Exact},
        d::AbstractDDM, 
        cur_plot; 
        density_offset = 0, 
        t_range = default_range(d), 
        model_args = (), 
        model_kwargs = (),
        density_scale = nothing,
        kwargs...
    )
    n_subplots = n_options(d)
    pds = gen_pds(d, t_range, n_subplots; model_args, model_kwargs)
    scale_density!(pds, density_scale)
    map!(x -> x .+ density_offset, pds, pds)
    pds_vec = vcat(pds...)
    filter!(!isnan, pds_vec)
    defaults = get_plot_defaults(d)
    dens = -1 * pds[2] 
    return plot!(cur_plot, t_range, dens; 
         defaults..., kwargs...)
end

function plot_bottom_density!(
        ::Type{<:Approximate},
        d::AbstractDDM, 
        cur_plot; 
        n_sim = 2_000,
        density_offset = 0, 
        t_range = default_range(d), 
        model_args = (), 
        model_kwargs = (),
        density_scale = nothing,
        kwargs...
    )
    n_subplots = n_options(d)
    choices, rts = rand(d, n_sim, model_args...; model_kwargs...)
    choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
    kdes = [kernel(rts[choices .== c]) for c ∈ 1:n_subplots]
    pds = gen_pds(kdes, t_range, choice_probs; model_args, model_kwargs)
    scale_density!(pds, density_scale)
    pds_vec = vcat(pds...)
    filter!(!isnan, pds_vec)
    defaults = get_plot_defaults(d)
    dens = -pds[2] .+ density_offset
    ymin = minimum(dens) * 1.05
    ymax = maximum(pds[1] .+ -1 * density_offset) * 1.05
    return plot!(cur_plot, t_range, dens; 
        ylims=(ymin,ymax), defaults..., kwargs...)
end

# n_subplots = n_options(d)
# choices, rts = rand(d, n_sim, model_args...; model_kwargs...)
# choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
# kdes = [kernel(rts[choices .== c]) for c ∈ 1:n_subplots]
# pds = gen_pds(kdes, t_range, choice_probs; model_args, model_kwargs)
# scale_density!(pds, density_scale)
# map!(x -> x .+ density_offset, pds, pds)
# ymax = maximum(vcat(pds...)) * 1.2
# defaults = get_plot_defaults(d)

# function add_mean_drift_rate(model, cur_plot, zs)
#     z = mean(zs)
#     x,y = make_mean_drift_rate_line(model, z)
#     plot!(cur_plot, x, y, color=:black)
# end

# function make_mean_drift_rate_line(model, z)
#     (;ν,k,A,τ) = model 
#     α = A + k
#     xmin = τ
#     xmax = @. (α - z) / ν + τ
#     x = collect.(range.(xmin, xmax, length=100))
#     y = map((x,ν,z) -> (x .- τ) * ν .+ z, x, ν, z)
#     return x,y
# end
