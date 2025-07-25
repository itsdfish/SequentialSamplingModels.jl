@safetestset "MLBA Tests" begin
    @safetestset "compute_utility" begin
        @safetestset "1" begin
            using SequentialSamplingModels
            using SequentialSamplingModels: compute_utility
            using Test

            model = MLBA(; γ = 1)

            @test compute_utility(model, [3, 1]) ≈ [3, 1]
            @test compute_utility(model, [2, 2]) ≈ [2, 2]
        end

        @safetestset "2" begin
            using SequentialSamplingModels
            using SequentialSamplingModels: compute_utility
            using Test

            model = MLBA(; γ = 0.5)

            @test compute_utility(model, [3, 1]) ≈ [1.60770, 0.53590] atol = 1e-4
            @test compute_utility(model, [2, 2]) ≈ [1, 1]
        end

        @safetestset "3" begin
            using SequentialSamplingModels
            using SequentialSamplingModels: compute_utility
            using Test

            model = MLBA(; γ = 1.2)

            @test compute_utility(model, [3, 1]) ≈ [3.2828, 1.0943] atol = 1e-4
            @test compute_utility(model, [2, 2]) ≈ [2.2449, 2.2449] atol = 1e-4
        end
    end

    @safetestset "compute_weight" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_weight
        using Test

        model = MLBA(; λₚ = 1.5, λₙ = 2.5)

        @test compute_weight(model, 2, 3) ≈ 0.082085 atol = 1e-4
        @test compute_weight(model, 3, 2) ≈ 0.22313 atol = 1e-4
    end

    @safetestset "compare" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compare
        using Test

        model = MLBA(; λₚ = 1.5, λₙ = 2.5)

        @test compare(model, [3, 1], [1, 3]) ≈ 0.086098 atol = 1e-4
        @test compare(model, [1, 3], [3, 1]) ≈ 0.086098 atol = 1e-4
        @test compare(model, [2, 4], [3, 5]) ≈ -0.16417 atol = 1e-4
    end

    @safetestset "compute_drift_rates" begin
        @safetestset "1" begin
            using SequentialSamplingModels
            using SequentialSamplingModels: compute_drift_rates!
            using Test

            model = MLBA(; λₚ = 1.5, λₙ = 2.5, β₀ = 1, γ = 1.2)

            stimuli = [
                1 3
                3 1
                2 2
            ]

            compute_drift_rates!(model, stimuli)
            @test model.ν ≈ [1.2269, 1.2269, 1.2546] atol = 1e-4
        end

        @safetestset "2" begin
            using SequentialSamplingModels
            using SequentialSamplingModels: compute_drift_rates!
            using Test

            model = MLBA(; λₚ = 0.75, λₙ = 3, β₀ = 2, γ = 2)

            stimuli = [
                1 1
                4 2
                2 3
            ]

            compute_drift_rates!(model, stimuli)
            @test model.ν ≈ [1.9480, 3.0471, 3.3273] atol = 1e-4
        end
    end

    @safetestset "logpdf" begin
        using Random
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_drift_rates!
        using Test

        Random.seed!(80071)

        parms = (
            λₚ = 0.20,
            λₙ = 0.40,
            β₀ = 5,
            γ = 5,
            τ = 0.3,
            A = 0.8,
            k = 0.5
        )

        M = [
            1 4
            2 2
            4 1
        ]

        choice, rt = rand(MLBA(; parms...,), 10000, M)

        λₚs = range(0.80 * parms.λₚ, 1.2 * parms.λₚ, length = 100)
        LLs = map(λₚ -> sum(logpdf.(MLBA(; parms..., λₚ), choice, rt, (M,))), λₚs)
        _, mxidx = findmax(LLs)
        @test λₚs[mxidx] ≈ parms.λₚ rtol = 0.02

        λₙs = range(0.80 * parms.λₙ, 1.2 * parms.λₙ, length = 100)
        LLs = map(λₙ -> sum(logpdf.(MLBA(; parms..., λₙ), choice, rt, (M,))), λₙs)
        _, mxidx = findmax(LLs)
        @test λₙs[mxidx] ≈ parms.λₙ rtol = 0.02

        β₀s = range(0.80 * parms.β₀, 1.2 * parms.β₀, length = 100)
        LLs = map(β₀ -> sum(logpdf.(MLBA(; parms..., β₀), choice, rt, (M,))), β₀s)
        _, mxidx = findmax(LLs)
        @test β₀s[mxidx] ≈ parms.β₀ rtol = 0.02

        γs = range(0.80 * parms.γ, 1.2 * parms.γ, length = 100)
        LLs = map(γ -> sum(logpdf.(MLBA(; parms..., γ), choice, rt, (M,))), γs)
        _, mxidx = findmax(LLs)
        @test γs[mxidx] ≈ parms.γ rtol = 5e-2

        τs = range(0.80 * parms.τ, 1.2 * parms.τ, length = 100)
        LLs = map(τ -> sum(logpdf.(MLBA(; parms..., τ), choice, rt, (M,))), τs)
        _, mxidx = findmax(LLs)
        @test τs[mxidx] ≈ parms.τ rtol = 0.02

        As = range(0.80 * parms.A, 1.2 * parms.A, length = 100)
        LLs = map(A -> sum(logpdf.(MLBA(; parms..., A), choice, rt, (M,))), As)
        _, mxidx = findmax(LLs)
        @test As[mxidx] ≈ parms.A rtol = 0.02

        ks = range(0.80 * parms.k, 1.2 * parms.k, length = 100)
        LLs = map(k -> sum(logpdf.(MLBA(; parms..., k), choice, rt, (M,))), ks)
        _, mxidx = findmax(LLs)
        @test ks[mxidx] ≈ parms.k rtol = 0.02
    end
end
