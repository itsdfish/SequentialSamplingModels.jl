using Documenter
using SequentialSamplingModels
using Turing
using Plots

makedocs(
    warnonly=true,
    sitename="SequentialSamplingModels",
    format=Documenter.HTML(
        assets=[
            asset(
                "https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap",
                class=:css,
            ),
        ],
        collapselevel=1,
    ),
    modules=[
        SequentialSamplingModels,
        Base.get_extension(SequentialSamplingModels, :TuringExt),
        Base.get_extension(SequentialSamplingModels, :PlotsExt),
    ],
    pages=[
        "Home" => "index.md",
        "Models" => [
            "Attentional Drift Diffusion (aDDM)" => "aDDM.md",
            "Circular Drift Diffusion Model (CDDM)" => "cddm.md",
            "Drift Diffusion Model (DDM)" => "DDM.md",
            "Ex-Gaussian Distribution" => "ex_gaussian.md",
            "Leaky Competing Accumulator (LCA)" => "lca.md",
            "Linear Ballistic Accumulator (LBA)" => "lba.md",
            "Lognormal Race Model (LNR)" => "lnr.md",
            "Muti-attribute attentional drift diffusion Model" => "maaDDM.md",
            "Poisson Race" => "poisson_race.md",
            "Racing Diffusion Model (RDM)" => "rdm.md",
            "Starting-time Drift Diffusion Model (stDDM)" => "stDDM.md",
            "Wald Model" => "wald.md",
            "Wald Mixture Model" => "wald_mixture.md",
        ],
        "Parameter Estimation" => [
            "Simple Bayesian Model" => "turing_simple.md",
            "Advanced Model Specification" => "turing_advanced.md",
            "Hierarchical Models" => "turing_hierarchical.md",
        ],
        "Model Comparison" => "bayes_factor.md",
        "Predictive Distributions" => "predictive_distributions.md",
        "Plotting" => [
            "Basic Example" => "basic_plot_example.md",
            "Changing the Layout" => "layout.md",
            "Plot Model Process" => "plot_model.md",
        ],
        "API" => "api.md",
        "Developer Guide" => "developer_guide.md",
    ],
)

deploydocs(repo="github.com/itsdfish/SequentialSamplingModels.jl.git")
