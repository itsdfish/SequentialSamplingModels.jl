@safetestset "Wald Mixture" begin
    @safetestset "pd" begin
        @safetestset "pdf 1" begin
            using Test, SequentialSamplingModels, StableRNGs
            include("../KDE.jl")
            rng = StableRNG(2777)
            d = WaldMixture(3, 0.5, 1, 0.1)
            @test mean(d) ≈ (1 / 3) + 0.1 atol = 1e-5
            rts = rand(rng, d, 200_000)
            # remove outliers
            filter!(x -> x < 5, rts)
            approx_pdf = kde(rts)
            x = 0.11:0.01:1.5
            y′ = pdf(approx_pdf, x)
            y = pdf.(d, x)
            @test y′ ≈ y rtol = 0.03
            @test mean(rts) ≈ mean(d) atol = 5e-2
        end

        @safetestset "pdf 2" begin
            using Test
            using SequentialSamplingModels
            using QuadGK

            dist = WaldMixture(; ν = 3.0, η = 1.0, α = 0.5, τ = 0.130)
            integral, _ = quadgk(t -> pdf(dist, t), 0.130, 20.0)
            @test integral ≈ 1 atol = 1e-4

            dist = WaldMixture(; ν = 2.0, η = 0.5, α = 0.3, τ = 0.130)
            integral, _ = quadgk(t -> pdf(dist, t), 0.130, 20.0)
            @test integral ≈ 1 atol = 1e-4
        end
    end

    @safetestset "loglikelihood" begin
        using SequentialSamplingModels
        using Test
        using StableRNGs
        rng = StableRNG(67)

        dist = WaldMixture(2, 0.2, 1, 0.1)
        rt = rand(rng, dist, 10)

        sum_logpdf = logpdf.(dist, rt) |> sum
        loglike = loglikelihood(dist, rt)
        @test sum_logpdf ≈ loglike
    end

    @safetestset "logpdf" begin
        using SequentialSamplingModels
        using Test

        dist = WaldMixture(; ν = 2.0, η = 0.5, α = 0.3, τ = 0.130)
        t = range(0.131, 2, length = 20)

        pdfs = pdf.(dist, t)
        logpdfs = logpdf.(dist, t)

        @test logpdfs ≈ log.(pdfs) atol = 1e-8
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using StableRNGs

        rng = StableRNG(12)
        α = 0.80

        dist = WaldMixture(; α)

        time_steps, evidence = simulate(rng, dist; Δt = 0.0005)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ α atol = 0.02
    end

    @safetestset "CDF" begin
        @safetestset "1" begin
            using StableRNGs
            using SequentialSamplingModels
            using StatsBase
            using Test

            rng = StableRNG(345)
            n_sim = 10_000

            dist = WaldMixture(; ν = 3.0, η = 0.5, α = 0.5, τ = 0.130)
            rt = rand(rng, dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
                sim_x = mean(rt .≤ t)
                x = cdf(dist, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end

        @safetestset "2" begin
            using StableRNGs
            using SequentialSamplingModels
            using StatsBase
            using Test

            rng = StableRNG(44)
            n_sim = 10_000

            dist = WaldMixture(; ν = 2.0, η = 0.2, α = 0.8, τ = 0.130)
            rt = rand(rng, dist, n_sim)
            ul, ub = quantile(rt, [0.05, 0.95])
            for t ∈ range(ul, ub, length = 10)
                sim_x = mean(rt .≤ t)
                x = cdf(dist, t)
                @test sim_x ≈ x atol = 1e-2
            end
        end
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels

        parms = (; ν = 3.0, η = 0.2, α = 0.5, τ = 0.130)

        model = WaldMixture(; parms...)
        @test values(parms) == params(model)
    end
end
