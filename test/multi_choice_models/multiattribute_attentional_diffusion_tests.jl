@safetestset "maaDDM" begin
    @safetestset "increment1" begin
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: increment

        model = maaDDM(;
            ν = [1.0 2.0; 2.0 3.0],
            α = 1.0,
            z = 0.0,
            θ = 0.5,
            ϕ = 0.50,
            ω = 0.00,
            σ = eps(),
            Δ = 1.0
        )

        Δ1s = map(x -> increment(model, x), 1:4)
        @test Δ1s[1] > 0
        @test Δ1s[2] > 0
        @test Δ1s[3] < 0
        @test Δ1s[4] < 0

        model = maaDDM(;
            ν = [100.0 2.0; 2.0 3.0],
            α = 1.0,
            z = 0.0,
            θ = 0.5,
            ϕ = 0.50,
            ω = 0.00,
            σ = eps(),
            Δ = 1.0
        )

        Δ2s = map(x -> increment(model, x), 1:4)
        @test Δ1s ≈ Δ2s atol = 1e-5
    end

    @safetestset "increment2" begin
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: increment

        model = maaDDM(;
            ν = [1.0 2.0; 2.0 3.0],
            α = 1.0,
            z = 0.0,
            θ = 0.5,
            ϕ = 0.50,
            ω = 0.00,
            σ = eps(),
            Δ = 1.0
        )

        Δs = map(x -> increment(model, x), 1:4)

        @test Δs[1] > 0
        @test Δs[2] > 0
        @test Δs[3] < 0
        @test Δs[4] < 0
    end

    @safetestset "increment3" begin
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: increment

        model = maaDDM(
            ν = [1.0 1.0; 2.0 2.0],
            α = 1.0,
            z = 0.0,
            θ = 1.0,
            ϕ = 0.50,
            ω = 0.50,
            σ = eps(),
            Δ = 1.0
        )

        Δs = map(x -> increment(model, x), 1:4)
        @test all(Δs .≈ -0.75)
    end

    @safetestset "increment4" begin
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: increment

        model = maaDDM(
            ν = [1.0 1.0; 2.0 2.0],
            α = 1.0,
            z = 0.0,
            θ = 1.0,
            ϕ = 1.0,
            ω = 1.0,
            σ = eps(),
            Δ = 1.0
        )

        Δ1s = map(x -> increment(model, x), 1:4)

        model = maaDDM(
            ν = [1.0 1.0; 2.0 2.0],
            α = 1.0,
            z = 0.0,
            θ = 1.0,
            ϕ = 1.0,
            ω = 0.0,
            σ = eps(),
            Δ = 1.0
        )

        Δ2s = map(x -> increment(model, x), 1:4)

        @test all(Δ1s .≈ -1.0)
        @test all(Δ2s .≈ -1.0)
    end

    @safetestset "increment5" begin
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: increment

        model = maaDDM(
            ν = [1.0 1.0; 2.0 2.0],
            α = 1.0,
            z = 0.0,
            θ = 1.0,
            ϕ = 0.0,
            ω = 1.0,
            σ = eps(),
            Δ = 1.0
        )

        Δ2s = map(x -> increment(model, x), 1:4)

        @test Δ2s ≈ [-1, 0, -1, 0]
    end

    @safetestset "increment6" begin
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: increment

        model = maaDDM(
            ν = [1.0 1.0; 2.0 2.0],
            α = 1.0,
            z = 0.0,
            θ = 0.0,
            ϕ = 1.0,
            ω = 1.0,
            σ = eps(),
            Δ = 1.0
        )

        Δ2s = map(x -> increment(model, x), 1:4)

        @test Δ2s ≈ [1, 1, -2, -2]
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using StatsBase
        using Random
        Random.seed!(456)

        mutable struct Transition
            state::Int
            n::Int
            mat::Array{Float64, 2}
        end

        function Transition(mat)
            n = size(mat, 1)
            state = rand(1:n)
            return Transition(state, n, mat)
        end

        function fixate(transition)
            (; mat, n, state) = transition
            w = @view mat[state, :]
            next_state = sample(1:n, Weights(w))
            transition.state = next_state
            return next_state
        end

        ν = [5.0 5.0; 1.0 1.0]

        dist = maaDDM(; ν, ω = 0.50, ϕ = 0.2)

        tmat = Transition(
            [
            0.98 0.015 0.0025 0.0025
            0.015 0.98 0.0025 0.0025
            0.0025 0.0025 0.98 0.015
            0.0025 0.0025 0.015 0.98
        ],
        )

        time_steps, evidence = simulate(dist, tmat; fixate)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ 1 atol = 0.040

        ν = [1.0 1.0; 5.0 5.0]

        dist = maaDDM(; ν, ω = 0.50, ϕ = 0.2)

        time_steps, evidence = simulate(dist, tmat; fixate)

        @test evidence[end] ≈ -1 atol = 0.040
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels

        parms = (;
            ν = [4.0 5.0; 5.0 4.0],
            σ = 0.02,
            Δ = 0.0004,
            θ = 0.3,
            ϕ = 0.50,
            ω = 0.70,
            α = 1.0,
            z = 0.0,
            τ = 0.0
        )

        model = maaDDM(; parms...)
        @test values(parms) == params(model)
    end
end
