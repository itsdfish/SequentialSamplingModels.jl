@safetestset "CDDM" begin
    @safetestset "solve_zeros" begin 
        using Test
        using SpecialFunctions
        using SequentialSamplingModels: solve_zeros
        
        n = 50
        solutions = solve_zeros(0, n)

        @test besselj.(0, solutions) ≈ zeros(n) atol = 1e-14
    end

    @safetestset "bassel_hm" begin 
        using Test 
        using SequentialSamplingModels
        using SequentialSamplingModels: bassel_hm 

        model = CDDM(;
            ν=[1.5,.5],
            η = [1,1],
            σ = 1.0,
            α = 1.5,
            τ = .300,
            zτ = .100)

        rts = [0:.1:1;]
        ground_truth = [0., 0.00286158, 0.19383708, 0.54859974, 0.76896096,
            0.84213747, 0.82883946, 0.7744823 , 0.7043917 , 0.63131337, 0.56121361]

        densities = map(rt -> bassel_hm(model, rt), rts) * (2 * π)
        @test densities ≈ ground_truth 
    end

    @safetestset "bassel_s" begin 
        using Test 
        using SequentialSamplingModels
        using SequentialSamplingModels: bassel_s

        model = CDDM(;
            ν=[1.5,.5],
            η = [1,1],
            σ = 1.0,
            α = 1.5,
            τ = .300,
            zτ = .100)

        rts = [0:.1:1;]
        ground_truth = [0. , 0.00280757, 0.18601942, 0.51358442, 0.70062069,
            0.74526978, 0.71124981, 0.64357356, 0.56624328, 0.49064167,0.4215687] 

        densities = map(rt -> bassel_s(model, rt), rts)
        @test densities ≈ ground_truth atol = 1e-5
    end
end


