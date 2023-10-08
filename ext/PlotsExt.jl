module PlotsExt 

    import Plots: histogram
    import Plots: histogram!
    import Plots: plot
    import Plots: plot!
    import SequentialSamplingModels as SSMs
    import SequentialSamplingModels: plot_model 
    import SequentialSamplingModels: plot_model!
    import SequentialSamplingModels: plot_quantiles
    import SequentialSamplingModels: plot_quantiles!  

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
    
    include("Plots/plot.jl")
    include("Plots/histogram.jl")
    include("Plots/plot_model.jl")
    include("Plots/plot_quantiles.jl")
    include("Plots/kde.jl")
end