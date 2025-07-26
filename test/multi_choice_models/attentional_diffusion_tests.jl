@safetestset "Attentional Diffusion" begin
    @safetestset "aDDM" begin
        using Test, SequentialSamplingModels, StatsBase, Random
        Random.seed!(5511)

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

        function fixate(t)
            w = @view t.mat[t.state, :]
            next_state = sample(1:(t.n), Weights(w))
            t.state = next_state
            return next_state
        end

        model = aDDM(; ν = [5.0, 4.0])

        tmat = Transition([
            0.98 0.015 0.005
            0.015 0.98 0.005
            0.45 0.45 0.1
        ])

        choice, rts = rand(model, 1000, tmat; fixate)
        p1 = mean(choice .== 1)
        @test p1 > 0.50

        model = aDDM(; ν = [4.0, 5.0])
        choice, rts1 = rand(model, 1000, tmat; fixate)
        p1 = mean(choice .== 1)
        @test p1 < 0.50

        μ_rts1 = mean(rts1)
        model = aDDM(; ν = [5.0, 5.0])
        choice, rts3 = rand(model, 1000, tmat; fixate)
        μ_rts3 = mean(rts3)
        @test μ_rts1 < μ_rts3
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using StatsBase
        using Random
        Random.seed!(8477)

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

        tmat = Transition([
            0.98 0.015 0.005
            0.015 0.98 0.005
            0.45 0.45 0.1
        ])

        dist = aDDM(; ν = [3, 0])

        time_steps, evidence = simulate(dist, tmat; fixate)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ 1 atol = 0.030

        dist = aDDM(; ν = [0, 3])

        time_steps, evidence = simulate(dist, tmat; fixate)

        @test evidence[end] ≈ -1 atol = 0.030
    end

    @safetestset "params" begin
        using Test
        using Distributions
        using SequentialSamplingModels

        parms = (; ν = [5.0, 4.0], σ = 0.02, Δ = 0.0004, θ = 0.3, α = 1.0, z = 0.0, τ = 0.0)
        model = aDDM(; parms...)
        @test values(parms) == params(model)
    end
end
