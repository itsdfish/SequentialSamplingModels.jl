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
    pages = ["home" => "index.md",
            "api" => "api.md",
            "examples" => ["Attentional Drift Diffusion" => "aDDM.md",
                 "Linear Ballistic Accumulator" => "lba.md"]]
)

deploydocs(
    repo = "github.com/itsdfish/SequentialSamplingModels.jl.git",
)