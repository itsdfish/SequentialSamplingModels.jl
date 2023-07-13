module PlotsExt 

    using SequentialSamplingModels
    import Plots: plot 

    function plot(dist::SSM2D; 
        t_range=range(.3,1.0, length=50), kwargs...)
        n_subplots = n_options(dist)
        pds = map(t -> pdf(dist,)
    end
end