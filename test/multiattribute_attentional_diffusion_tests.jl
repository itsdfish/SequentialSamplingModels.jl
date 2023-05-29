@safetestset "maaDDM" begin 
    @safetestset "update1" begin 
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: update 

        model = maaDDM(ν₁₁ = 1.0, 
                        ν₁₂ = 2.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 3.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = .5, 
                        ϕ = .50, 
                        ω = .00, 
                        σ = eps(), 
                        Δ = 1.0)

        Δ1s = map(x -> update(model, x), 1:4)
        @test Δ1s[1] > 0
        @test Δ1s[2] > 0
        @test Δ1s[3] < 0
        @test Δ1s[4] < 0

        model = maaDDM(ν₁₁ = 100.0, 
                        ν₁₂ = 2.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 3.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = .5, 
                        ϕ = .50, 
                        ω = .00, 
                        σ = eps(), 
                        Δ = 1.0)

        Δ2s = map(x -> update(model, x), 1:4)
        @test Δ1s ≈ Δ2s atol = 1e-5
    end

    @safetestset "update2" begin 
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: update 

        model = maaDDM(ν₁₁ = 1.0, 
                        ν₁₂ = 2.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 3.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = .5, 
                        ϕ = .50, 
                        ω = .00, 
                        σ = eps(), 
                        Δ = 1.0)

        Δs = map(x -> update(model, x), 1:4)

        @test Δs[1] > 0
        @test Δs[2] > 0
        @test Δs[3] < 0
        @test Δs[4] < 0
    end

    @safetestset "update3" begin 
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: update 

        model = maaDDM(ν₁₁ = 1.0, 
                        ν₁₂ = 1.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 2.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = 1.0, 
                        ϕ = .50, 
                        ω = .50, 
                        σ = eps(), 
                        Δ = 1.0)

        Δs = map(x -> update(model, x), 1:4)
        @test all(Δs .≈ -.75)
    end

    @safetestset "update4" begin 
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: update 

        model = maaDDM(ν₁₁ = 1.0, 
                        ν₁₂ = 1.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 2.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = 1.0, 
                        ϕ = 1.0, 
                        ω = 1.0, 
                        σ = eps(), 
                        Δ = 1.0)

        Δ1s = map(x -> update(model, x), 1:4)

        model = maaDDM(ν₁₁ = 1.0, 
                        ν₁₂ = 1.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 2.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = 1.0, 
                        ϕ = 1.0, 
                        ω = 0.0, 
                        σ = eps(), 
                        Δ = 1.0)

        Δ2s = map(x -> update(model, x), 1:4)

        @test all(Δ1s .≈ -1.0)
        @test all(Δ2s .≈ -1.0)
    end

    @safetestset "update5" begin 
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: update 

        model = maaDDM(ν₁₁ = 1.0, 
                        ν₁₂ = 1.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 2.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = 1.0, 
                        ϕ = 0.0, 
                        ω = 1.0, 
                        σ = eps(), 
                        Δ = 1.0)

        Δ2s = map(x -> update(model, x), 1:4)

        @test Δ2s ≈ [-1,0,-1,0]
    end

    @safetestset "update6" begin 
        using Test, SequentialSamplingModels
        using SequentialSamplingModels: update 

        model = maaDDM(ν₁₁ = 1.0, 
                        ν₁₂ = 1.0, 
                        ν₂₁ = 2.0, 
                        ν₂₂ = 2.0, 
                        α = 1.0, 
                        z = 0.0, 
                        θ = 0.0, 
                        ϕ = 1.0, 
                        ω = 1.0, 
                        σ = eps(), 
                        Δ = 1.0)

        Δ2s = map(x -> update(model, x), 1:4)

        @test Δ2s ≈ [1,1,-2,-2]
    end
end
