module PlotsExt 

    import Plots: histogram
    import Plots: histogram!
    import Plots: plot
    import Plots: plot!
    import SequentialSamplingModels: plot_model 

    using Interpolations
    using KernelDensity
    using KernelDensity: Epanechnikov
    using LinearAlgebra
    using Plots
    using SequentialSamplingModels
    using SequentialSamplingModels: Approximate
    using SequentialSamplingModels: Exact
    using SequentialSamplingModels: get_pdf_type
    using SequentialSamplingModels: simulate

    const SSMs = SequentialSamplingModels

    include("Plots/plot.jl")
    include("Plots/histogram.jl")
    include("Plots/plot_model.jl")
    include("Plots/kde.jl")
end