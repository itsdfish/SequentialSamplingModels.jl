module PlotsExt 

    import Plots: histogram
    import Plots: histogram!
    import Plots: plot
    import Plots: plot!
    import SequentialSamplingModels as SSMs
    import SequentialSamplingModels: get_plot_defaults
    import SequentialSamplingModels: get_model_plot_defaults
    import SequentialSamplingModels: get_default_labels
    import SequentialSamplingModels: plot_model 
    import SequentialSamplingModels: plot_model!
    import SequentialSamplingModels: plot_quantiles
    import SequentialSamplingModels: plot_quantiles!  
    import SequentialSamplingModels: plot_choices
    import SequentialSamplingModels: plot_choices!

    using Interpolations
    using KernelDensity
    using KernelDensity: Epanechnikov
    using LinearAlgebra
    using Plots
    using SequentialSamplingModels
    using SequentialSamplingModels: Approximate
    using SequentialSamplingModels: Exact
    using SequentialSamplingModels: get_pdf_type
    using Statistics
    
    include("plots/plot.jl")
    include("plots/histogram.jl")
    include("plots/plot_model.jl")
    include("plots/plot_quantiles.jl")
    include("plots/plot_choices.jl")
    include("plots/kde.jl")
end