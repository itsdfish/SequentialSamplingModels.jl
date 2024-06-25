@safetestset "ClassicMDFT tests" begin
    @safetestset "similarity effect" begin
        using SequentialSamplingModels
        using Random
        using Test

        Random.seed!(5484)
        # non-decision time 
        τ = 0.300
        # diffusion noise 
        σ = 1.0
        # decision threshold
        α = 17.5
        # attribute attention weights 
        w = [0.5, 0.5]
        # value matrix where rows correspond to attributes, and columns correspond to options
        M = [
            1.0 3.0
            3.0 1.0
            0.9 3.1
        ]
        # feedback matrix 
        S = [
            0.9500000 -0.0122316 -0.04999996
            -0.0122316 0.9500000 -0.00903030
            -0.0499996 -0.0090303 0.95000000
        ]

        model = ClassicMDFT(; σ, α, τ, w, S)
        choices, _ = rand(model, 100_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        ground_truth = [0.305315, 0.395226, 0.299459]
        @test probs ≈ ground_truth atol = 5e-3
    end

    @safetestset "compromise effect" begin
        using SequentialSamplingModels
        using Random
        using Test

        Random.seed!(6541)
        # non-decision time 
        τ = 0.300
        # diffusion noise 
        σ = 1.0
        # decision threshold
        α = 17.5
        # attribute attention weights 
        w = [0.5, 0.5]
        # value matrix where rows correspond to attributes, and columns correspond to options
        M = [
            1.0 3.0
            3.0 1.0
            2.0 2.0
        ]
        # feedback matrix 
        S = [
            0.950000 -0.012232 -0.045788
            -0.012232 0.950000 -0.045788
            -0.045788 -0.045788 0.950000
        ]

        model = ClassicMDFT(; σ, α, τ, w, S)
        choices, _ = rand(model, 100_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        ground_truth = [0.282626, 0.284605, 0.432769]
        @test probs ≈ ground_truth atol = 5e-3
    end

    @safetestset "attraction effect" begin
        using SequentialSamplingModels
        using Random
        using Test

        Random.seed!(201)
        # non-decision time 
        τ = 0.300
        # diffusion noise 
        σ = 1.0
        # decision threshold
        α = 17.5
        # attribute attention weights 
        w = [0.5, 0.5]
        # value matrix where rows correspond to attributes, and columns correspond to options
        M = [
            1.0 3.0
            3.0 1.0
            0.50 2.5
        ]
        # feedback matrix 
        S = [
            0.950000 -0.01223200 -0.02264700
            -0.012232 0.95000000 -0.00067034
            -0.022647 -0.00067034 0.95000000
        ]

        model = ClassicMDFT(; σ, α, τ, w, S)
        choices, _ = rand(model, 100_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        ground_truth = [0.559048, 0.440950, 0.000002]
        @test probs ≈ ground_truth atol = 5e-3
    end
end

@safetestset "MDFT" begin
    @safetestset "compute_distances" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_distances
        using Random
        using Test

        model = MDFT(;
            n_alternatives = 3,
            σ = 1.0,
            α = 1.0,
            τ = 0.30,
            γ = 1.0,
            κ = [0.1, 0.2],
            ϕ1 = 0.01,
            ϕ2 = 0.10,
            β = 10.0
        )

        M = [
            1 3
            2 2
            0 2
        ]
        distances = compute_distances(model, M)
        ground_truth = [
            0 2 20
            2 0 22
            20 22 0
        ]

        @test distances ≈ ground_truth atol = 5e-3
    end

    @safetestset "compute_distances" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_distances
        using SequentialSamplingModels: compute_feedback_matrix
        using Random
        using Test

        model = MDFT(;
            n_alternatives = 3,
            σ = 1.0,
            α = 1.0,
            τ = 0.30,
            γ = 1.0,
            κ = [0.1, 0.2],
            ϕ1 = 0.01,
            ϕ2 = 0.10,
            β = 10.0
        )

        M = [
            1 3
            2 2
            0 2
        ]
        distances = compute_distances(model, M)
        S = compute_feedback_matrix(model, distances)
        ground_truth = [
            0.90 -0.0961 -0.0018
            -0.0961 0.90 -0.0008
            -0.0018 -0.0008 0.90
        ]

        @test S ≈ ground_truth atol = 5e-3
    end

    @safetestset "similarity effect" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_distances
        using Random
        using Test

        Random.seed!(8474)

        model = MDFT(;
            n_alternatives = 3,
            σ = 0.3,
            α = 1.0,
            τ = 0.0,
            γ = 1.0,
            κ = [0.1, 0.2],
            ϕ1 = 0.01,
            ϕ2 = 0.10,
            β = 10.0
        )

        M = [
            1.0 3.0
            3.0 1.0
            0.9 3.1
        ]

        true_probs = [0.18424, 0.53014, 0.28556]
        true_mean_rts = [0.6503107, 0.4729380, 0.6221656]
        choices, rts = rand(model, 10_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        @test probs ≈ true_probs rtol = 0.01
        mean_rts = map(c -> mean(rts[choices .== c]), 1:3)
        @test mean_rts ≈ true_mean_rts rtol = 0.01
    end

    @safetestset "compromise effect" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_distances
        using Random
        using Test

        Random.seed!(6511)

        model = MDFT(;
            n_alternatives = 3,
            σ = 0.3,
            α = 1.0,
            τ = 0.0,
            γ = 1.0,
            κ = [0.1, 0.2],
            ϕ1 = 0.01,
            ϕ2 = 0.10,
            β = 10.0
        )

        M = [
            1.0 3.0
            3.0 1.0
            2.0 2.0
        ]

        true_probs = [0.47776, 0.51236, 0.00988]
        true_mean_rts = [0.5474727, 0.5713786, 1.4739737]

        choices, rts = rand(model, 10_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        @test probs ≈ true_probs rtol = 0.02
        mean_rts = map(c -> mean(rts[choices .== c]), 1:3)
        #@test mean_rts ≈ true_mean_rts rtol = .02
    end

    @safetestset "attraction effect" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_distances
        using Random
        using Test

        Random.seed!(2141)

        model = MDFT(;
            n_alternatives = 3,
            σ = 0.3,
            α = 1.0,
            τ = 0.0,
            γ = 1.0,
            κ = [0.1, 0.2],
            ϕ1 = 0.01,
            ϕ2 = 0.10,
            β = 10.0
        )

        M = [
            1.0 3.0
            3.0 1.0
            0.50 2.5
        ]

        true_probs = [0.44548, 0.52751, 0.02701]
        true_mean_rts = [0.6130686, 0.4260398, 0.6872297]

        choices, rts = rand(model, 10_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        @test probs ≈ true_probs rtol = 0.02
        mean_rts = map(c -> mean(rts[choices .== c]), 1:3)
        @test mean_rts ≈ true_mean_rts rtol = 0.02
    end
end
