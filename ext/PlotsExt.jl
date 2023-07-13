module PlotsExt 

    using SequentialSamplingModels
    import Plots: plot 

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
        return ssm_plot(d; t_range, kwargs...)
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
        return ssm_plot(d; t_range, kwargs...)
    end

    function ssm_plot(d; t_range, kwargs...)
        n_subplots = n_options(d)
        pds = gen_pds(d, t_range, n_subplots)
        ymax = maximum(vcat(pds...)) * 1.1
        title = ["choice $i" for _ ∈ 1:1,  i ∈ 1:n_subplots]
        return plot(t_range, pds; layout=(n_subplots,1), 
            ylims = (0,ymax), xaxis=("RT [s]"), yaxis = "density", 
            color = :black, title, leg=false, kwargs...)
    end

    function gen_pds(d::SSM2D, t_range, n_subplots)
        return [pdf.(d, (i,), t_range) for i ∈ 1:n_subplots]
    end

    function gen_pds(d::SSM1D, t_range, n_subplots)
        return [pdf.(d, t_range) for i ∈ 1:n_subplots]
    end

    function default_range(d)
        n = n_options(d)
        return range(d.τ + eps(), d.τ+ .25 * log(n + 1), length=100)
    end
end