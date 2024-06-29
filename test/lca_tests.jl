@safetestset "LCA Tests" begin
    @safetestset "LCA predictions" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(8414)

        parms = (α = 1.5, β = 0.20, λ = 0.10, ν = [2.5, 2.0], τ = 0.30, σ = 1.0)

        model = LCA(; parms...)
        choice, rt = rand(model, 10_000)

        @test mean(choice .== 1) ≈ 0.60024 atol = 5e-3
        @test mean(rt[choice .== 1]) ≈ 0.7478 atol = 5e-3
        @test mean(rt[choice .== 2]) ≈ 0.7555 atol = 5e-3
        @test std(rt[choice .== 1]) ≈ 0.1869 atol = 5e-3
        @test std(rt[choice .== 2]) ≈ 0.18607 atol = 5e-3
    end
    # import numpy as np

    # def lca_trial(v, a, ndt, la, ka, dt=1e-3, s=1.0, max_iter=1e5):
    #     """Generates a response time and choice from the LCA model given a set of parameters"""
    #     # get number of decision alternatives
    #     n_alt = len(v)
    #     # constant for diffusion process
    #     c = np.sqrt(dt * s)
    #     # initialize accumulator activities
    #     x = np.zeros(n_alt)
    #     # accumulation process
    #     n_iter = 0
    #     while np.all(x < a) and n_iter < max_iter:
    #         # iterate over accumulators
    #         for i in range(n_alt):
    #             x[i] = max(0.0, (x[i] + (v[i] - la*x[i] - ka*(np.sum(x)-x[i]))*dt + c*np.random.randn()))
    #         n_iter += 1
    #     # determine respnose time and choice
    #     rt = n_iter*dt + ndt
    #     if n_iter < max_iter:
    #         resp = float(np.where(x >= a)[0][0])
    #     else:
    #         resp = -1.0

    #     return rt, resp

    # rts = np.array([])
    # resps = np.array([])
    # for _ in range(0, 100_000):
    #     rt, resp = lca_trial(v=np.array([2.5,2.0]), a=1.5, ndt=0.3, la=0.1, ka=0.2, s=1.0, max_iter = 1e7)
    #     resps = np.append(resps, resp)
    #     rts = np.append(rts, rt)

    # np.mean(resps == 0)

    # np.mean(rts[resps == 0])
    # np.mean(rts[resps == 1])

    # np.std(rts[resps == 0])
    # np.std(rts[resps == 1])

    @safetestset "compute_mean_evidence!" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: compute_mean_evidence!
        using Test

        parms = (α = 1.5, β = 0.20, λ = 0.10, ν = [2.5, 2.0], τ = 0.30, σ = 1.0)

        model = LCA(; parms...)

        Δμ = [0.0, 0.0]
        x = [1.0, 2.0]

        compute_mean_evidence!(model, x, Δμ)
        @test Δμ[1] ≈ (2.5 - 0.1 - 0.4)
        @test Δμ[2] ≈ (2.0 - 0.2 - 0.2)

        parms = (α = 1.5, β = 0.00, λ = 0.00, ν = [2.5, 2.0], τ = 0.30, σ = 1.0)
        model = LCA(; parms...)
        Δμ = [0.0, 0.0]
        x = [1.0, 2.0]

        compute_mean_evidence!(model, x, Δμ)
        @test Δμ[1] ≈ 2.5
        @test Δμ[2] ≈ 2.0

        parms = (α = 1.5, β = 0.20, λ = 0.10, ν = [0.0, 0.0], τ = 0.30, σ = 1.0)
        model = LCA(; parms...)
        Δμ = [0.0, 0.0]
        x = [1.0, 2.0]

        compute_mean_evidence!(model, x, Δμ)
        @test Δμ[1] ≈ (-0.1 - 0.4)
        @test Δμ[2] ≈ (-0.2 - 0.2)
    end

    @safetestset "increment!" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: increment!
        using Test
        using Random
        Random.seed!(6521)

        Δt = 0.001
        model = LCA(;
            ν = [2.5, 2.0],
            β = 0.20,
            λ = 0.10,
            σ = 0.10
        )

        Δμ = [0.0, 0.0]
        x = [0.0, 0.0]

        n_reps = 1000
        evidence = fill(0.0, n_reps, 2)
        for i ∈ 1:n_reps
            x .= 1.0
            increment!(model, x, Δμ; Δt)
            evidence[i, :] = x
        end

        true_std = model.σ * sqrt(Δt)
        true_means = [(2.5 - 0.1 - 0.4) (2.0 - 0.2 - 0.2)] * Δt .+ 1.0

        @test mean(evidence, dims = 1) ≈ true_means atol = 5e-4
        @test std(evidence, dims = 1) ≈ fill(true_std, 1, 2) atol = 5e-4
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using Random

        Random.seed!(843)
        α = 0.80
        Δt = 0.0005
        dist = LCA(; α, ν = [2, 1])

        time_steps, evidence = simulate(dist; Δt)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == size(evidence, 1)
        @test size(evidence, 2) == 2
        @test maximum(evidence[end, :]) ≈ α atol = 0.005
    end
end
