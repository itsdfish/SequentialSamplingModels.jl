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
end
