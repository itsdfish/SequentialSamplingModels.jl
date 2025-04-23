using Documenter
using SequentialSamplingModels
using Turing
using Plots

makedocs(
    warnonly = true,
    sitename = "SequentialSamplingModels",
    format = Documenter.HTML(
        assets = [
            asset(
            "https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap",
            class = :css
        )
        ],
        collapselevel = 1
    ),
    modules = [
        SequentialSamplingModels,
        Base.get_extension(SequentialSamplingModels, :TuringExt),
        Base.get_extension(SequentialSamplingModels, :PlotsExt)
    ],
    pages = [
        "Home" => "index.md",
        "Models" => [
            "Single Choice Models" => [
                "Ex-Gaussian" => "ex_gaussian.md",
                "Shifted Log Normal" => "shifted_lognormal.md",
                "Wald" => "wald.md",
                "Wald Mixture" => "wald_mixture.md"
            ],
            "Multi-choice Models" => [
                "Single Attribute Models" => [
                    "Drift Diffusion Model (DDM)" => "DDM.md",
                    "Leaky Competing Accumulator (LCA)" => "lca.md",
                    "Linear Ballistic Accumulator (LBA)" => "lba.md",
                    "Log Normal Race (LNR)" => "lnr.md",
                    "Poisson Race" => "poisson_race.md",
                    "Racing Diffusion Model (RDM)" => "rdm.md",
                    "Starting-time Drift Diffusion Model (stDDM)" => "stDDM.md"
                ],
                "Multi-attribute Models" => [
                    "Muti-attribute Attentional Drift Diffusion Model" => "maaDDM.md",
                    "Multi-attribute Decision Field Theory" => "mdft.md",
                    "Multi-attribute Linear Ballistic Accumulator" => "mlba.md"
                ]
            ],
            "Alternative Geometries" => [
                "Circular Drift Diffusion Model (CDDM)" => "cddm.md"
            ]
        ],
        "Parameter Estimation" => [
            "Mode Estimation" => "mode_estimation.md",
            "Simple Bayesian Model" => "turing_simple.md",
            "Advanced Model Specification" => "turing_advanced.md",
            "Hierarchical Models" => "turing_hierarchical.md",
            "Amortized Neural Estimation" => [
                "Point Estimation" => "amortized_point_estimation.md",
                "Bayesian Parameter Estimation" => "amortized_bayesian_parameter_estimation.md"
            ]
        ],
        "Model Comparison" => [
            "Bayes Factors" => "bayes_factor.md",
            "PSIS-LOO" => "loo_compare.md"
        ],
        "Predictive Distributions" => "predictive_distributions.md",
        "Plotting" => [
            "Basic Example" => "basic_plot_example.md",
            "Changing the Layout" => "layout.md",
            "Plot Model Process" => "plot_model.md"
        ],
        "Performance Tips" => "performance_tips.md",
        "API" => "api.md",
        "Developer Guide" => "developer_guide.md"
    ]
)

deploydocs(repo = "github.com/itsdfish/SequentialSamplingModels.jl.git")
