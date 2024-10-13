@safetestset "Shifted Lognormal Distribution" begin
    @safetestset "rand" begin
        using Distributions
        using Random
        using SequentialSamplingModels
        using Test

        Random.seed!(5411)

        lognormal = LogNormal(-1, 0.3)
        shifted_lognormal = ShiftedLogNormal(ν = -1, σ = 0.3, τ = 0.5)
        mean(rand(shifted_lognormal, 10_000))
        # τ is properly added
        @test mean(rand(lognormal, 10_000)) - mean(rand(shifted_lognormal, 10_000)) ≈ -0.5 atol =
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
        using Random
        using SequentialSamplingModels
        using Test

        Random.seed!(2008)

        parms = (ν = -1, σ = 0.3, τ = 0.0)
        x = rand(ShiftedLogNormal(; parms...), 15_000)

        νs = range(0.80 * parms.ν, 1.2 * parms.ν, length = 100)
        LLs = map(ν -> sum(logpdf.(ShiftedLogNormal(; parms..., ν), x)), νs)
        _, idx = findmax(LLs)
        @test νs[idx] ≈ parms.ν rtol = 0.01

        σs = range(0.80 * parms.σ, 1.2 * parms.σ, length = 100)
        LLs = map(σ -> sum(logpdf.(ShiftedLogNormal(; parms..., σ), x)), σs)
        _, idx = findmax(LLs)
        @test σs[idx] ≈ parms.σ rtol = 0.01

        τs = range(0.80 * parms.τ, 1.2 * parms.τ, length = 100)
        LLs = map(τ -> sum(logpdf.(ShiftedLogNormal(; parms..., τ), x)), τs)
        _, idx = findmax(LLs)
        @test τs[idx] ≈ parms.τ rtol = 0.01
    end
end
