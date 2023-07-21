using Documenter
using SequentialSamplingModels

makedocs(
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
    modules=[SequentialSamplingModels],
    pages=[
        "Home" => "index.md",
        "Models" => [
            "Drift Diffusion Model (DDM)" => "DDM.md",
            "Linear Ballistic Accumulator (LBA)" => "lba.md",
            "Lognormal Race Model (LNR)" => "lnr.md",
            "Racing Diffusion Model (RDM)" => "rdm.md",
            "Leaky Competing Accumulator (LCA)" => "lca.md",
            "Attentional Drift Diffusion (aDDM)" => "aDDM.md",
            "Muti-attribute attentional drift diffusion Model" => "maaDDM.md",
            "Wald Model" => "wald.md",
            "Wald Mixture Model" => "wald_mixture.md"
        ],
        "Parameter Estimation" => [
            "Simple Bayesian" => "turing_simple.md",
            "Advanced Bayesian" => "turing_advanced.md",
        ],
        "Plotting" => "plotting.md",
        "API" => "api.md",
        "Developer Guide" => "developer_guide.md"
    ]
)

deploydocs(
    repo="github.com/itsdfish/SequentialSamplingModels.jl.git",
)