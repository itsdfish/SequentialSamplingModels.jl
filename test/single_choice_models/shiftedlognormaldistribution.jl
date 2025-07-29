@safetestset "Shifted Lognormal Distribution" begin
    @safetestset "rand" begin
        using Distributions
        using StableRNGs
        using SequentialSamplingModels
        using Test

        rng = StableRNG(344)

        lognormal = LogNormal(-1, 0.3)
        shifted_lognormal = ShiftedLogNormal(ν = -1, σ = 0.3, τ = 0.5)
        mean(rand(rng, shifted_lognormal, 10_000))
        # τ is properly added
        @test mean(rand(rng, lognormal, 10_000)) -
              mean(rand(rng, shifted_lognormal, 10_000)) ≈ -0.5 atol =
            0.01
    end

    @safetestset "pdf" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        lognormal = LogNormal(-1, 0.3)
        shifted_lognormal = ShiftedLogNormal(ν = -1, σ = 0.3, τ = 0.0)
        x = rand(shifted_lognormal, 100)
        @test pdf.(lognormal, x) ≈ pdf.(shifted_lognormal, x)
    end

    @safetestset "logpdf 1" begin
        using Distributions
        using SequentialSamplingModels
        using Test

        lognormal = LogNormal(-1, 0.3)
        shifted_lognormal = ShiftedLogNormal(ν = -1, σ = 0.3, τ = 0.0)
        x = rand(shifted_lognormal, 100)
        @test logpdf.(lognormal, x) ≈ logpdf.(shifted_lognormal, x)
    end

    @safetestset "logpdf 2" begin
        using Distributions
        using StableRNGs
        using SequentialSamplingModels
        using Test

        rng = StableRNG(4)

        parms = (ν = -1, σ = 0.3, τ = 0.0)
        x = rand(rng, ShiftedLogNormal(; parms...), 15_000)

        νs = range(0.80 * parms.ν, 1.2 * parms.ν, length = 100)
        LLs = map(ν -> sum(logpdf.(ShiftedLogNormal(; parms..., ν), x)), νs)
        _, idx = findmax(LLs)
        @test νs[idx] ≈ parms.ν rtol = 0.01

        σs = range(0.80 * parms.σ, 1.2 * parms.σ, length = 100)
        LLs = map(σ -> sum(logpdf.(ShiftedLogNormal(; parms..., σ), x)), σs)
        _, idx = findmax(LLs)
        @test σs[idx] ≈ parms.σ rtol = 0.01

        τs = range(0.80 * parms.τ, parms.τ, length = 100)
        LLs = map(τ -> sum(logpdf.(ShiftedLogNormal(; parms..., τ), x)), τs)
        _, idx = findmax(LLs)
        @test τs[idx] ≈ parms.τ rtol = 0.01
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels

        parms = (; ν = -1, σ = 0.5, τ = 0.20)

        model = ShiftedLogNormal(; parms...)
        @test values(parms) == params(model)
    end

    @safetestset "parameter checks" begin
        @safetestset "all valid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = -1.0, σ = 0.5, τ = 0.20)
            ShiftedLogNormal(; parms...)
            ShiftedLogNormal(values(parms)...)
            @test true
        end

        @safetestset "σ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = -1.0, σ = -0.5, τ = 0.20)
            @test_throws ArgumentError ShiftedLogNormal(; parms...)
            @test_throws ArgumentError ShiftedLogNormal(values(parms)...)
        end

        @safetestset "τ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels

            parms = (; ν = -1.0, σ = 0.5, τ = -0.20)
            @test_throws ArgumentError ShiftedLogNormal(; parms...)
            @test_throws ArgumentError ShiftedLogNormal(values(parms)...)
        end
    end
end
