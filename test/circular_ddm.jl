@safetestset "CDDM" begin
    @safetestset "solve_zeros" begin 
        using Test
        using SpecialFunctions
        using SequentialSamplingModels: solve_zeros
        
        n = 50
        solutions = solve_zeros(0, n)

        @test besselj.(0, solutions) ≈ zeros(n) atol = 1e-14
    end

    @safetestset "pdf_rt_hm" begin 
        using Test 
        using SequentialSamplingModels
        using SequentialSamplingModels: pdf_rt_hm 

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

        densities = map(rt -> pdf_rt_hm(model, rt), rts) * (2 * π)
        @test densities ≈ ground_truth 
    end
end