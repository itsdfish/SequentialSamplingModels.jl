module PlotsExt 

    using SequentialSamplingModels
    using SequentialSamplingModels: Approximate
    using SequentialSamplingModels: Exact
    using SequentialSamplingModels: get_pdf_type
    using KernelDensity
    using Plots 

    import Plots: histogram
    import Plots: histogram!
    import Plots: plot
    import Plots: plot! 
    include("kde.jl")

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

    function ssm_plot(::Type{<:Exact}, d; t_range, kwargs...)
        n_subplots = n_options(d)
        pds = gen_pds(d, t_range, n_subplots)
        ymax = maximum(vcat(pds...)) * 1.1
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        return plot(t_range, pds; layout=(n_subplots,1), 
            ylims = (0,ymax), xaxis=("RT [s]"), yaxis = "density", 
            grid=false, color = :black, title, leg=false, kwargs...)
    end

    function ssm_plot(::Type{<:Approximate}, d; n_sim = 2000, t_range, kwargs...)
        n_subplots = n_options(d)
        choices, rts = rand(d, n_sim)
        choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
        kdes = [kernel(rts[choices .== c]) for c ∈ 1:n_subplots]
        pds = gen_pds(kdes, t_range, choice_probs)
        ymax = maximum(vcat(pds...)) * 1.1
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        return plot(t_range, pds; layout=(n_subplots,1), 
            ylims = (0,ymax), xaxis=("RT [s]"), yaxis = "density", 
            grid=false, color = :black, title, leg=false, kwargs...)
    end

    plot!(d::SSM2D; t_range=default_range(d), kwargs...) = plot!(Plots.current(), d; t_range, kwargs...)

    plot!(d::SSM1D; t_range=default_range(d), kwargs...) = plot!(Plots.current(), d; t_range, kwargs...)

    function plot!(cur_plot::Plots.Plot, d::SSM1D; t_range=default_range(d), kwargs...)
        return ssm_plot!(get_pdf_type(d), d, cur_plot; t_range, kwargs...)
    end

    function plot!(cur_plot::Plots.Plot, d::SSM2D; t_range=default_range(d), kwargs...)
        return ssm_plot!(get_pdf_type(d), d, cur_plot; t_range, kwargs...)
    end

    function ssm_plot!(::Type{<:Exact}, d, cur_plot; t_range, kwargs...)
        n_subplots = n_options(d)
        pds = gen_pds(d, t_range, n_subplots)
        ymax = maximum(vcat(pds...)) * 1.1
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        return plot!(cur_plot, t_range, pds; layout=(n_subplots,1), 
            ylims = (0,ymax), xaxis=("RT [s]"), yaxis = "density", 
            grid=false, color = :black, title, leg=false, kwargs...)
    end

    function ssm_plot!(::Type{<:Approximate}, d, cur_plot; n_sim = 2000, t_range, kwargs...)
        n_subplots = n_options(d)
        choices, rts = rand(d, n_sim)
        choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
        kdes = [kernel(rts[choices .== c]) for c ∈ 1:n_subplots]
        pds = gen_pds(kdes, t_range, choice_probs)
        ymax = maximum(vcat(pds...)) * 1.1
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        return plot!(cur_plot, t_range, pds; layout=(n_subplots,1), 
            ylims = (0,ymax), xaxis=("RT [s]"), yaxis = "density", 
            grid=false, color = :black, title, leg=false, kwargs...)
    end

    function gen_pds(d::SSM2D, t_range, n_subplots)
        return [pdf.(d, (i,), t_range) for i ∈ 1:n_subplots]
    end

    function gen_pds(d::SSM1D, t_range, n_subplots)
        return [pdf.(d, t_range) for i ∈ 1:n_subplots]
    end

    function gen_pds(kdes, t_range, probs)
        return [pdf(kdes[i], t_range) .* probs[i] for i ∈ 1:length(kdes)]
    end

    function default_range(d)
        n = n_options(d)
        return range(d.τ + eps(), d.τ+ .25 * log(n + 1), length=100)
    end

    """
        histogram(d::SSM2D;  kwargs...)

    Plots the histogram of a multi-alternative sequential sampling model.

    # Arguments

    - `d::SSM2D`: a model object for a mult-alternative sequential sampling model 

    # Keywords 

    - `kwargs...`: optional keyword arguments for configuring  plot options
    """
    function histogram(d::SSM2D; norm=true, n_sim=2000, kwargs...)
        return ssm_histogram(d; norm, n_sim, kwargs...)
    end

    function ssm_histogram(d::SSM2D; norm, n_sim, kwargs...)
        n_subplots = n_options(d)
        choices, rts = rand(d, n_sim)
        choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
        rt_vecs = map(c -> rts[choices .== c], 1:n_subplots)
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        yaxis = norm ? "density" : "frequency"
        hist = histogram(rt_vecs; layout=(n_subplots,1), norm,
            xaxis=("RT [s]"), yaxis = "density", grid=false, color = :grey, 
            title, leg=false, kwargs...)
        ymax = get_y_max(hist, n_subplots) * 1.10
        plot!(hist, ylims=(0, ymax); kwargs...)
        norm ? scale_density!(hist, choice_probs, 1, n_subplots) : nothing 
        return hist
    end

    function histogram(d::SSM1D; norm=true, n_sim=2000, kwargs...)
        return ssm_histogram(d; norm, n_sim, kwargs...)
    end

    function ssm_histogram(d::SSM1D; norm, n_sim, kwargs...)
        n_subplots = n_options(d)
        rts = rand(d, n_sim)
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        yaxis = norm ? "density" : "frequency"
        hist = histogram(rts; norm, xaxis=("RT [s]"), yaxis = "density", grid=false,
            color = :grey, title, leg=false, kwargs...)
        return hist
    end

    histogram!(d::SSM2D; norm=true, n_sim=2000, kwargs...) = histogram!(Plots.current(), d; norm, n_sim, kwargs...)

    histogram!(d::SSM1D; norm=true, n_sim=2000, kwargs...) = histogram!(Plots.current(), d; norm, n_sim, kwargs...)


    function histogram!(cur_plot::Plots.Plot, d::SSM2D; norm=true, n_sim=2000, kwargs...)
        return ssm_histogram!(d, cur_plot; norm, n_sim, kwargs...)
    end

    function ssm_histogram!(d::SSM2D, cur_plot; norm, n_sim, kwargs...)
        n_subplots = n_options(d)
        choices, rts = rand(d, n_sim)
        choice_probs = map(c -> mean(choices .== c), 1:n_subplots)
        rt_vecs = map(c -> rts[choices .== c], 1:n_subplots)
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        yaxis = norm ? "density" : "frequency"
        hist = histogram!(cur_plot, rt_vecs; layout=(n_subplots,1), norm,
            xaxis=("RT [s]"), yaxis = "density", grid=false, color = :grey, 
            title, leg=false, kwargs...)
        ymax = get_y_max(hist, n_subplots) * 1.10
        norm ? scale_density!(hist, choice_probs, 2, n_subplots) : nothing 
        return hist
    end

    function histogram!(cur_plot::Plots.Plot, d::SSM1D; norm=true, n_sim=2000, kwargs...)
        return ssm_histogram!(d, cur_plot; norm, n_sim, kwargs...)
    end

    function ssm_histogram!(d::SSM1D, cur_plot; norm, n_sim, kwargs...)
        println(kwargs...)
        n_subplots = n_options(d)
        rts = rand(d, n_sim)
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        yaxis = norm ? "density" : "frequency"
        return histogram!(cur_plot, rts; norm, xaxis=("RT [s]"), yaxis = "density", grid=false,
            color = :grey, title, leg=false, kwargs...)
    end

    function get_y_max(hist, n_options)
        dens = mapreduce(i -> hist[i][1][:y], vcat, 1:n_options)
        filter!(!isnan, dens)
        return maximum(dens)
    end

    function scale_density!(hist, probs, id, n_options)
        for i ∈ 1:n_options
            hist[i][id][:y] .*= probs[i]
        end
        return nothing
    end
end