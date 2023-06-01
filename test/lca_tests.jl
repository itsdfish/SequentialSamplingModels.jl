@safetestset "LCA Tests" begin
    @safetestset "LCA predictions" begin
        using SequentialSamplingModels
        using Test
        using Random 
        Random.seed!(8414)

        parms = (α = 1.5, 
            β=0.20,
             λ=0.10, 
             ν=[2.5,2.0], 
             Δt=.001, 
             τ=.30, 
             σ=1.0)

        model = LCA(;parms...)
        choice,rt = rand(model, 100_000)
        
        @test mean(choice .== 1) ≈ 0.60024 atol = 5e-3
        @test mean(rt[choice .== 1]) ≈ 0.7478 atol = 5e-3
        @test mean(rt[choice .== 2]) ≈ 0.7555 atol = 5e-3
        @test std(rt[choice .== 1]) ≈ 0.1869 atol = 5e-3
        @test std(rt[choice .== 2]) ≈ 0.18607 atol = 5e-3
    end
    #import numpy as np

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
end