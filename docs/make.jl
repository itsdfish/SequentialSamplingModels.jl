using Documenter
using SequentialSamplingModels

makedocs(
    sitename = "SequentialSamplingModels",
    format = Documenter.HTML(
        assets = [
            asset(
                "https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap",
                class = :css,
            ),
        ],
        collapselevel = 1,
    ),
    modules = [SequentialSamplingModels],
    pages = ["Home" => "index.md",
            "API" => "api.md",
            "Models" => ["Attentional Drift Diffusion" => "aDDM.md",
                "Leaky Competing Accumulator" => "lca.md",
                "Linear Ballistic Accumulator" => "lba.md",
                "Lognormal Race Model" => "lnr.md",
                "Muti-attribute attentional drift diffusion Model" => "maaDDM.md",
                "Racing Diffusion Model" => "rdm.md",
                "Wald Model" => "wald.md",
                "Wald Mixture Model" => "wald_mixture.md"],
            "Parameter Estimation with Turing" => "turing.md",
            ]
)

deploydocs(
    repo = "github.com/itsdfish/SequentialSamplingModels.jl.git",
)