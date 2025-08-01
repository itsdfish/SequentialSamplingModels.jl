@safetestset "ClassicMDFT tests" begin
    @safetestset "similarity effect" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test

        rng = StableRNG(1152)
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
        choices, _ = rand(rng, model, 100_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        ground_truth = [0.305315, 0.395226, 0.299459]
        @test probs ≈ ground_truth atol = 5e-3
    end

    @safetestset "compromise effect" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test

        rng = StableRNG(34)
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
        choices, _ = rand(rng, model, 100_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        ground_truth = [0.282626, 0.284605, 0.432769]
        @test probs ≈ ground_truth atol = 5e-3
    end

    @safetestset "attraction effect" begin
        using SequentialSamplingModels
        using StableRNGs
        using Test

        rng = StableRNG(322)
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
        choices, _ = rand(rng, model, 100_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        ground_truth = [0.559048, 0.440950, 0.000002]
        @test probs ≈ ground_truth atol = 5e-3
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels
        using SequentialSamplingModels: make_default_contrast

        parms = (;
            # diffusion noise 
            σ = 1.0,
            C = make_default_contrast(3),
            # feedback matrix 
            S = [
                0.950000 -0.01223200 -0.02264700
                -0.012232 0.95000000 -0.00067034
                -0.022647 -0.00067034 0.95000000
            ],
            # attribute attention weights 
            w = [0.5, 0.5],
            # decision threshold
            α = 17.5,
            # non-decision time 
            τ = 0.300
        )
        model = ClassicMDFT(; parms...)
        @test values(parms) == params(model)
    end

    @safetestset "parameter checks" begin
        @safetestset "all valid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                # diffusion noise 
                σ = 1.0,
                C = make_default_contrast(3),
                # feedback matrix 
                S = [
                    0.950000 -0.01223200 -0.02264700
                    -0.012232 0.95000000 -0.00067034
                    -0.022647 -0.00067034 0.95000000
                ],
                # attribute attention weights 
                w = [0.5, 0.5],
                # decision threshold
                α = 17.5,
                # non-decision time 
                τ = 0.300
            )
            ClassicMDFT(; parms...)
            ClassicMDFT(values(parms)...)
            @test true
        end

        @safetestset "σ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                # diffusion noise 
                σ = -1.0,
                C = make_default_contrast(3),
                # feedback matrix 
                S = [
                    0.950000 -0.01223200 -0.02264700
                    -0.012232 0.95000000 -0.00067034
                    -0.022647 -0.00067034 0.95000000
                ],
                # attribute attention weights 
                w = [0.5, 0.5],
                # decision threshold
                α = 17.5,
                # non-decision time 
                τ = 0.300
            )
            @test_throws ArgumentError ClassicMDFT(; parms...)
            @test_throws ArgumentError ClassicMDFT(values(parms)...)
        end

        @safetestset "w invalid 1" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                # diffusion noise 
                σ = 1.0,
                C = make_default_contrast(3),
                # feedback matrix 
                S = [
                    0.950000 -0.01223200 -0.02264700
                    -0.012232 0.95000000 -0.00067034
                    -0.022647 -0.00067034 0.95000000
                ],
                # attribute attention weights 
                w = [-0.5, 1.5],
                # decision threshold
                α = 17.5,
                # non-decision time 
                τ = 0.300
            )
            @test_throws ArgumentError ClassicMDFT(; parms...)
            @test_throws ArgumentError ClassicMDFT(values(parms)...)
        end

        @safetestset "w invalid 2" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                # diffusion noise 
                σ = 1.0,
                C = make_default_contrast(3),
                # feedback matrix 
                S = [
                    0.950000 -0.01223200 -0.02264700
                    -0.012232 0.95000000 -0.00067034
                    -0.022647 -0.00067034 0.95000000
                ],
                # attribute attention weights 
                w = [0.4, 1.5],
                # decision threshold
                α = 17.5,
                # non-decision time 
                τ = 0.300
            )
            @test_throws ArgumentError ClassicMDFT(; parms...)
            @test_throws ArgumentError ClassicMDFT(values(parms)...)
        end

        @safetestset "α invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                # diffusion noise 
                σ = 1.0,
                C = make_default_contrast(3),
                # feedback matrix 
                S = [
                    0.950000 -0.01223200 -0.02264700
                    -0.012232 0.95000000 -0.00067034
                    -0.022647 -0.00067034 0.95000000
                ],
                # attribute attention weights 
                w = [0.5, 0.5],
                # decision threshold
                α = -17.5,
                # non-decision time 
                τ = 0.300
            )
            @test_throws ArgumentError ClassicMDFT(; parms...)
            @test_throws ArgumentError ClassicMDFT(values(parms)...)
        end

        @safetestset "τ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                # diffusion noise 
                σ = 1.0,
                C = make_default_contrast(3),
                # feedback matrix 
                S = [
                    0.950000 -0.01223200 -0.02264700
                    -0.012232 0.95000000 -0.00067034
                    -0.022647 -0.00067034 0.95000000
                ],
                # attribute attention weights 
                w = [0.5, 0.5],
                # decision threshold
                α = 17.5,
                # non-decision time 
                τ = -0.300
            )
            @test_throws ArgumentError ClassicMDFT(; parms...)
            @test_throws ArgumentError ClassicMDFT(values(parms)...)
        end
    end
end

@safetestset "MDFT" begin
    @safetestset "compute_distances" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_distances
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

    @safetestset "make_default_contrast" begin
        using LinearAlgebra
        using SequentialSamplingModels
        using SequentialSamplingModels: make_default_contrast
        using Test

        offdiag(A) = (A[ι] for ι in CartesianIndices(A) if ι[1] ≠ ι[2])

        C = make_default_contrast(3)

        @test size(C) == (3, 3)
        @test C[diagind(C)] ≈ fill(1, 3)
        @test all(x -> x == -0.5, offdiag(C))
    end

    @safetestset "similarity effect" begin
        using SequentialSamplingModels
        using Distributions
        using StableRNGs
        using Test
        include("mdft_test_functions.jl")

        rng = StableRNG(1602)

        parms = (
            σ = 0.1,
            α = 0.50,
            τ = 0.0,
            γ = 1.0,
            ϕ1 = 0.01,
            ϕ2 = 0.1,
            β = 10,
            κ = [5, 5]
        )

        model = MDFT(; n_alternatives = 3, parms...)

        M = [
            1.0 3.0
            3.0 1.0
            0.9 3.1
        ]

        @test test_context_effect(
            rng,
            parms,
            (),
            M;
            test_func = test_similarity,
            n_sim = 1500
        )
        true_probs = [0.15667, 0.54876, 0.29457]
        true_mean_rts = [0.9106421, 0.5445200, 0.7405360]
        choices, rts = rand(rng, model, 10_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        @test probs ≈ true_probs rtol = 0.02
        mean_rts = map(c -> mean(rts[choices .== c]), 1:3)
        @test mean_rts ≈ true_mean_rts rtol = 0.02
    end

    @safetestset "compromise effect" begin
        using Distributions
        using SequentialSamplingModels
        using StableRNGs
        using Test
        include("mdft_test_functions.jl")

        rng = StableRNG(23)

        parms = (
            σ = 0.1,
            α = 1.0,
            τ = 0.0,
            γ = 1.0,
            ϕ1 = 0.03,
            ϕ2 = 1.2,
            β = 10.0,
            κ = [10, 10]
        )

        model = MDFT(; n_alternatives = 3, parms...)

        M = [
            1.0 3.0
            2.0 2.0
            3.0 1.0
        ]

        @test test_context_effect(
            rng,
            parms,
            (),
            M;
            test_func = test_compromise,
            n_sim = 2000
        )

        true_probs = [0.30025, 0.40453, 0.29522]
        true_mean_rts = [2.238162, 2.886065, 2.239521]

        choices, rts = rand(rng, model, 10_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        @test probs ≈ true_probs rtol = 0.02
        mean_rts = map(c -> mean(rts[choices .== c]), 1:3)
        @test mean_rts ≈ true_mean_rts rtol = 0.02
    end

    @safetestset "attraction effect" begin
        using Distributions
        using SequentialSamplingModels
        using StableRNGs
        using Test
        include("mdft_test_functions.jl")
        rng = StableRNG(5644)

        parms = (
            σ = 0.1,
            α = 0.50,
            τ = 0.0,
            γ = 1.0,
            ϕ1 = 0.01,
            ϕ2 = 0.1,
            β = 10,
            κ = [5, 5]
        )
        model = MDFT(; n_alternatives = 3, parms...)

        M = [
            3.0 1.0
            1.0 3.0
            0.50 2.5
        ]

        @test test_context_effect(
            rng,
            parms,
            (),
            M;
            test_func = test_attraction,
            n_sim = 1000
        )

        true_probs = [0.55643, 0.44356, 0.00001]
        true_mean_rts = [0.3985818, 0.5411759, 0.3875000]

        choices, rts = rand(rng, model, 10_000, M)
        probs = map(c -> mean(choices .== c), 1:3)
        @test probs ≈ true_probs rtol = 0.02
        mean_rts = map(c -> mean(rts[choices .== c]), 1:3)
        @test mean_rts[1:2] ≈ true_mean_rts[1:2] rtol = 0.02
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels
        using SequentialSamplingModels: make_default_contrast

        parms = (;
            σ = 0.1,
            γ = 1.0,
            κ = [6.0, 5.0],
            ϕ1 = 0.01,
            ϕ2 = 0.10,
            β = 10.0,
            C = make_default_contrast(3),
            α = 0.50,
            τ = 0.0
        )

        model = MDFT(; n_alternatives = 3, parms...)
        @test values(parms) == params(model)
    end

    @safetestset "parameter checks" begin
        @safetestset "all valid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = 0.1,
                γ = 1.0,
                κ = [6.0, 5.0],
                ϕ1 = 0.01,
                ϕ2 = 0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = 0.50,
                τ = 0.0
            )

            MDFT(; n_alternatives = 3, parms...)
            MDFT(values(parms)...)
            @test true
        end

        @safetestset "σ invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = -0.1,
                γ = 1.0,
                κ = [6.0, 5.0],
                ϕ1 = 0.01,
                ϕ2 = 0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = 0.50,
                τ = 0.0
            )

            @test_throws ArgumentError MDFT(; n_alternatives = 3, parms...)
            @test_throws ArgumentError MDFT(values(parms)...)
        end

        @safetestset "κ invalid 1" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = 0.1,
                γ = 1.0,
                κ = [-6.0, 5.0],
                ϕ1 = 0.01,
                ϕ2 = 0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = 0.50,
                τ = 0.0
            )

            @test_throws ArgumentError MDFT(; n_alternatives = 3, parms...)
            @test_throws ArgumentError MDFT(values(parms)...)
        end

        @safetestset "κ invalid 2" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = 0.1,
                γ = 1.0,
                κ = [6.0, -5.0],
                ϕ1 = 0.01,
                ϕ2 = 0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = 0.50,
                τ = 0.0
            )

            @test_throws ArgumentError MDFT(; n_alternatives = 3, parms...)
            @test_throws ArgumentError MDFT(values(parms)...)
        end

        @safetestset "ϕ1 invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = 0.1,
                γ = 1.0,
                κ = [6.0, 5.0],
                ϕ1 = -0.01,
                ϕ2 = 0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = 0.50,
                τ = 0.0
            )

            @test_throws ArgumentError MDFT(; n_alternatives = 3, parms...)
            @test_throws ArgumentError MDFT(values(parms)...)
        end

        @safetestset "ϕ2 invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = 0.1,
                γ = 1.0,
                κ = [6.0, 5.0],
                ϕ1 = 0.01,
                ϕ2 = -0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = 0.50,
                τ = 0.0
            )

            @test_throws ArgumentError MDFT(; n_alternatives = 3, parms...)
            @test_throws ArgumentError MDFT(values(parms)...)
        end

        @safetestset "α invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = 0.1,
                γ = 1.0,
                κ = [6.0, 5.0],
                ϕ1 = 0.01,
                ϕ2 = 0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = -0.50,
                τ = 0.0
            )

            @test_throws ArgumentError MDFT(; n_alternatives = 3, parms...)
            @test_throws ArgumentError MDFT(values(parms)...)
        end

        @safetestset "α invalid" begin
            using Test
            using Distributions
            using SequentialSamplingModels
            using SequentialSamplingModels: make_default_contrast

            parms = (;
                σ = 0.1,
                γ = 1.0,
                κ = [6.0, 5.0],
                ϕ1 = 0.01,
                ϕ2 = 0.10,
                β = 10.0,
                C = make_default_contrast(3),
                α = 0.50,
                τ = -0.1
            )

            @test_throws ArgumentError MDFT(; n_alternatives = 3, parms...)
            @test_throws ArgumentError MDFT(values(parms)...)
        end
    end
end
