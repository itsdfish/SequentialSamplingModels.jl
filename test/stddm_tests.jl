@safetestset "stDDM Tests" begin
    @safetestset "stDDM predictions" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(8414)

        parms = (ν = [1.0, 1.2], α = 0.8, τ = 0.30 , s = 0.0, z = 0.50)
        model = stDDM(; parms...)
        choice, rt = rand(model, 10_000)
        # use validated simulator
        #Lombardi, G., & Hare, T. Piecewise constant averaging methods allow for fast and 
        # accurate hierarchical Bayesian estimation of drift diffusion models with 
        # time-varying evidence accumulation rates. PsyArXiv, (2021). https://doi.org/10.31234/osf.io/5azyx
        #code to validate
        #https://github.com/galombardi/method_HtSSM_aDDM/tree/master/RecoveryFitting/stDDM
        @test mean(choice .== 1) ≈  0.8192000  atol = 1e-2
        @test mean(rt[choice.==1]) ≈ 0.4332338 atol = 1e-2
        @test mean(rt[choice.==2]) ≈ 0.4555747 atol = 1e-2
        @test std(rt[choice.==1]) ≈ 0.1106120 atol = 1e-2
        @test std(rt[choice.==2]) ≈ 0.1318918 atol = 1e-2
    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using Random

        Random.seed!(8477)
        α = 0.80
        Δt = 0.0001
        dist = stDDM(; α, ν = [3, 3], Δt)

        time_steps, evidence = simulate(dist)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ α atol = 0.05

        dist = stDDM(; α, ν = [-3, -3], Δt)
        time_steps, evidence = simulate(dist)
        @test evidence[end] ≈ 0.0 atol = 0.05
    end

end
