@safetestset "MDFT tests" begin
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

        model = MDFT(; σ, α, τ, w, S)
        choices, _ = rand(model, 100_000, M; Δt = 1)
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

        model = MDFT(; σ, α, τ, w, S)
        choices, _ = rand(model, 100_000, M; Δt = 1)
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

        model = MDFT(; σ, α, τ, w, S)
        choices, _ = rand(model, 100_000, M; Δt = 1)
        probs = map(c -> mean(choices .== c), 1:3)
        ground_truth = [0.559048, 0.440950, 0.000002]
        @test probs ≈ ground_truth atol = 5e-3
    end
end
