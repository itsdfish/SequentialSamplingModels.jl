@safetestset "CDDM" begin
    @safetestset "bessel_hm" begin
        using Test
        using SequentialSamplingModels
        using SequentialSamplingModels: bessel_hm
        using SequentialSamplingModels: precompute_bessel

        model = CDDM(; ν = [1.5, 0.5], η = [1, 1], σ = 1.0, α = 1.5, τ = 0.0)

        rts = [0:0.1:1;]
        ground_truth = [
            0.0,
            0.00286158,
            0.19383708,
            0.54859974,
            0.76896096,
            0.84213747,
            0.82883946,
            0.7744823,
            0.7043917,
            0.63131337,
            0.56121361
        ]

        densities = map(rt -> bessel_hm(model, rt), rts) * (2 * π)
        @test densities ≈ ground_truth

        j0, j01, j02 = precompute_bessel()
        densities = map(rt -> bessel_hm(model, rt, j0, j01, j02), rts) * (2 * π)
        @test densities ≈ ground_truth
    end

    @safetestset "bessel_s" begin
        using Test
        using SequentialSamplingModels
        using SequentialSamplingModels: bessel_s

        model = CDDM(; ν = [1.5, 0.5], η = [1, 1], σ = 1.0, α = 1.5, τ = 0.0)

        rts = [0:0.1:1;]
        ground_truth = [
            0.0,
            0.00280757,
            0.18601942,
            0.51358442,
            0.70062069,
            0.74526978,
            0.71124981,
            0.64357356,
            0.56624328,
            0.49064167,
            0.4215687
        ]

        densities = map(rt -> bessel_s(model, rt), rts)
        @test densities ≈ ground_truth atol = 1e-5
    end

    @safetestset "rand" begin
        @safetestset "rand 1" begin
            using Test
            using Distributions
            using StableRNGs
            using SequentialSamplingModels
            using Statistics
            include("../KDE.jl")

            rng = StableRNG(584)

            model = CDDM(; ν = [1.5, 1], η = [0.0, 0.0], σ = 1.0, α = 1.5, τ = 0.300)

            data = rand(rng, model, 100_000)
            approx_pdf = kernel(data[:, 1])

            x = range(-π, π, length = 100)

            μ = atan(model.ν[2], model.ν[1])
            κ = model.α * √(sum(model.ν .^ 2)) / model.σ^2
            y_true = pdf.(VonMises(μ, κ), x)
            y = pdf(approx_pdf, x)
            @test y ≈ y_true rtol = 0.015
            @test maximum(abs.(y_true .- y)) < 0.02
        end

        @safetestset "rand 2" begin
            using Test
            using Distributions
            using StableRNGs
            using SequentialSamplingModels
            using Statistics
            include("../KDE.jl")

            rng = StableRNG(102)

            model = CDDM(; ν = [1.5, -1], η = [0.0, 0.0], σ = 0.5, α = 2.5, τ = 0.400)

            data = rand(rng, model, 100_000)
            approx_pdf = kernel(data[:, 1])

            x = range(-π, π, length = 200)

            μ = atan(model.ν[2], model.ν[1])
            κ = model.α * √(sum(model.ν .^ 2)) / model.σ^2
            y_true = pdf.(VonMises(μ, κ), x)
            y = pdf(approx_pdf, x)
            @test y ≈ y_true rtol = 0.015
            @test maximum(abs.(y_true .- y)) < 0.02
        end
    end

    @safetestset "pdf_rt" begin
        @safetestset "pdf_rt 1" begin
            using Test
            using Distributions
            using StableRNGs
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_rt
            using Statistics
            include("../KDE.jl")

            rng = StableRNG(588)

            model = CDDM(; ν = [1.75, 1.0], η = [0.50, 0.50], σ = 0.50, α = 2.5, τ = 0.20)

            rts = range(model.τ, 3.5, length = 200)
            dens = map(rt -> pdf_rt(model, rt), rts)
            data = rand(rng, model, 100_000)

            approx_pdf = kernel(data[:, 2])
            true_dens = pdf(approx_pdf, rts)

            @test dens ≈ true_dens rtol = 0.05
            @test maximum(abs.(true_dens .- dens)) < 0.07
        end

        @safetestset "pdf_rt 2" begin
            using Test
            using Distributions
            using StableRNGs
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_rt
            using Statistics
            include("../KDE.jl")

            rng = StableRNG(112)

            model = CDDM(; ν = [1.75, 2.0], η = [0.50, 0.50], σ = 0.50, α = 1.0, τ = 0.30)

            rts = range(model.τ, 1.5, length = 200)
            dens = map(rt -> pdf_rt(model, rt), rts)
            data = rand(rng, model, 100_000)

            approx_pdf = kernel(data[:, 2])
            true_dens = pdf(approx_pdf, rts)

            @test dens ≈ true_dens rtol = 0.05
            @test maximum(abs.(true_dens .- dens)) < 0.15
        end
    end

    @safetestset "pdf_angle" begin
        @safetestset "pdf_angle 1" begin
            using Test
            using Distributions
            using StableRNGs
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_angle
            using Statistics
            include("../KDE.jl")

            rng = StableRNG(478)

            model = CDDM(; ν = [1.75, 1.0], η = [0.50, 0.50], σ = 0.50, α = 2.5, τ = 0.20)

            θs = range(-π, π, length = 200)
            dens = map(θ -> pdf_angle(model, θ), θs)
            data = rand(rng, model, 100_000)

            approx_pdf = kernel(data[:, 1])
            true_dens = pdf(approx_pdf, θs)

            @test dens ≈ true_dens rtol = 0.05
            @test maximum(abs.(true_dens .- dens)) < 0.07
        end

        @safetestset "pdf_angle 2" begin
            using Test
            using Distributions
            using StableRNGs
            using SequentialSamplingModels
            using SequentialSamplingModels: pdf_angle
            using Statistics
            include("../KDE.jl")

            rng = StableRNG(90)

            model = CDDM(; ν = [1.75, 2.0], η = [0.50, 0.50], σ = 0.50, α = 1.0, τ = 0.30)

            θs = range(-π, π, length = 200)
            dens = map(θ -> pdf_angle(model, θ), θs)
            data = rand(rng, model, 100_000)

            approx_pdf = kernel(data[:, 1])
            true_dens = pdf(approx_pdf, θs)

            @test dens ≈ true_dens rtol = 0.05
            @test maximum(abs.(true_dens .- dens)) < 0.07
        end
    end

    @safetestset "logpdf_term1" begin
        using Test
        using Distributions
        using SequentialSamplingModels
        using SequentialSamplingModels: logpdf_term1
        using SequentialSamplingModels: pdf_term1

        model = CDDM(; ν = [1.75, 2.0], η = [0.50, 0.50], σ = 0.50, α = 1.0, τ = 0.30)

        for θ ∈ range(-2π, 2π, length = 20), t ∈ range(0.3, 2, length = 20)
            @test logpdf_term1(model, θ, t) ≈ log(pdf_term1(model, θ, t))
        end
    end

    @safetestset "logpdf_term2" begin
        using Test
        using Distributions
        using SequentialSamplingModels
        using SequentialSamplingModels: logpdf_term2
        using SequentialSamplingModels: pdf_term2

        model = CDDM(; ν = [1.75, 2.0], η = [0.50, 0.50], σ = 0.50, α = 1.0, τ = 0.30)

        for t ∈ range(0.3, 2, length = 20)
            @test logpdf_term2(model, t) ≈ log(pdf_term2(model, t))
        end
    end

    @safetestset "logpdf" begin
        using Test
        using Distributions
        using SequentialSamplingModels
        using StableRNGs

        rng = StableRNG(665)

        sum_logpdf(model, data) = sum(logpdf(model, data))

        parms = (ν = [1.75, 1.0], η = [0.50, 0.50], σ = 1.0, α = 3.5, τ = 0.30)

        model = CDDM(; parms...)
        data = rand(rng, model, 2_000)

        τs = range(parms.τ * 0.5, parms.τ, length = 50)
        LLs = map(τ -> sum_logpdf(CDDM(; parms..., τ), data), τs)
        _, max_idx = findmax(LLs)
        @test parms.τ ≈ τs[max_idx] rtol = 0.05

        αs = range(parms.α * 0.8, parms.α * 1.2, length = 50)
        LLs = map(α -> sum_logpdf(CDDM(; parms..., α), data), αs)
        _, max_idx = findmax(LLs)
        @test parms.α ≈ αs[max_idx] rtol = 0.05

        σs = range(parms.σ * 0.8, parms.σ * 1.2, length = 50)
        LLs = map(σ -> sum_logpdf(CDDM(; parms..., σ), data), σs)
        _, max_idx = findmax(LLs)
        @test parms.σ ≈ σs[max_idx] rtol = 0.05

        ν1s = range(parms.ν[1] * 0.8, parms.ν[1] * 1.2, length = 50)
        LLs = map(ν1 -> sum_logpdf(CDDM(; parms..., ν = [ν1, parms.ν[2]]), data), ν1s)
        _, max_idx = findmax(LLs)
        @test parms.ν[1] ≈ ν1s[max_idx] rtol = 0.05

        ν2s = range(parms.ν[2] * 0.8, parms.ν[2] * 1.2, length = 50)
        LLs = map(ν2 -> sum_logpdf(CDDM(; parms..., ν = [parms.ν[1], ν2]), data), ν2s)
        _, max_idx = findmax(LLs)
        @test parms.ν[2] ≈ ν2s[max_idx] rtol = 0.05

        η1s = range(parms.η[1] * 0.8, parms.η[1] * 1.2, length = 50)
        LLs = map(η1 -> sum_logpdf(CDDM(; parms..., η = [η1, parms.η[2]]), data), η1s)
        _, max_idx = findmax(LLs)
        @test parms.η[1] ≈ η1s[max_idx] rtol = 0.05

        η2s = range(parms.η[2] * 0.8, parms.η[2] * 1.2, length = 50)
        LLs = map(η2 -> sum_logpdf(CDDM(; parms..., η = [parms.η[1], η2]), data), η2s)
        _, max_idx = findmax(LLs)
        @test parms.η[2] ≈ η2s[max_idx] atol = 0.05
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels

        parms = (; ν = [1, 0.5], σ = 1, η = [1, 1], α = 1.5, τ = 0.30)
        model = CDDM(; parms...)
        @test values(parms) == params(model)
    end

    @safetestset "parameter checks" begin
        @safetestset "all valid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = [1, 0.5], σ = 1, η = [1, 1], α = 1.5, τ = 0.30)
            CDDM(; parms...)
            CDDM(values(parms)...)
            @test true
        end

        @safetestset "σ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = [1, 0.5], σ = -1, η = [1, 1], α = 1.5, τ = 0.30)
            @test_throws ArgumentError CDDM(; parms...)
            @test_throws ArgumentError CDDM(values(parms)...)
        end

        @safetestset "σ invalid 1" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = [1, 0.5], σ = 1, η = [-1, 1], α = 1.5, τ = 0.30)
            @test_throws ArgumentError CDDM(; parms...)
            @test_throws ArgumentError CDDM(values(parms)...)
        end

        @safetestset "σ invalid 2" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = [1, 0.5], σ = 1, η = [1, -1], α = 1.5, τ = 0.30)
            @test_throws ArgumentError CDDM(; parms...)
            @test_throws ArgumentError CDDM(values(parms)...)
        end

        @safetestset "α invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = [1, 0.5], σ = 1, η = [1, 1], α = -1.5, τ = 0.30)
            @test_throws ArgumentError CDDM(; parms...)
            @test_throws ArgumentError CDDM(values(parms)...)
        end

        @safetestset "τ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = [1, 0.5], σ = 1, η = [1, 1], α = 1.5, τ = -0.30)
            @test_throws ArgumentError CDDM(; parms...)
            @test_throws ArgumentError CDDM(values(parms)...)
        end
    end
end
