@safetestset "CDDM" begin
    @safetestset "solve_zeros" begin 
        using Test
        using SpecialFunctions
        using SequentialSamplingModels: solve_zeros
        
        n = 50
        solutions = solve_zeros(0, n)

        @test besselj.(0, solutions) ≈ zeros(n) atol = 1e-14
    end

    @safetestset "bessel_hm" begin 
        using Test 
        using SequentialSamplingModels
        using SequentialSamplingModels: bessel_hm 

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

        densities = map(rt -> bessel_hm(model, rt), rts) * (2 * π)
        @test densities ≈ ground_truth 
    end

    @safetestset "bessel_s" begin 
        using Test 
        using SequentialSamplingModels
        using SequentialSamplingModels: bessel_s

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

        densities = map(rt -> bessel_s(model, rt), rts)
        @test densities ≈ ground_truth atol = 1e-5
    end

    @safetestset "rand" begin 
        @safetestset "rand 1" begin 
            using Test 
            using Random
            using SequentialSamplingModels
            using Statistics 

            Random.seed!(5)

            model = CDDM(;
                ν=[1.5,1],
                η = [0,0],
                σ = 1.0,
                α = 1.5,
                τ = .300,
                zτ = .100)

            n_sim = 100_000
            probs = .1:.1:.9

            sim_data = rand(model, n_sim)
            rt_ground_truth = [0.3928865, 0.4444666, 0.4931019, 0.5430096, 0.5972292, 0.6595818, 0.7356822, 0.8383617, 1.0057995]
            qs_rt = quantile(sim_data[:,2], probs)
            @test qs_rt ≈ rt_ground_truth atol = .005

            θ_ground_truth = [0.2097635, 0.3756054, 0.5224047, 0.6652165, 0.8135603, 0.9811396, 1.1971366, 1.5990255, 5.9120380] 
            qs_θ = quantile(sim_data[:,1], probs)
            @test qs_θ ≈ θ_ground_truth atol = .005
        end

        @safetestset "rand 2" begin 
            using Test 
            using Random
            using SequentialSamplingModels
            using Statistics 
    
            Random.seed!(9385)
    
            model = CDDM(;
                ν=[.6,-.5],
                η = [0,0],
                σ = .5,
                α = .5,
                τ = .200,
                zτ = .100)
    
            n_sim = 100_000
            probs = .1:.1:.9
    
            sim_data = rand(model, n_sim)
            rt_ground_truth = [0.2158301, 0.2335248, 0.2535496, 0.2766709, 0.3039753, 0.3375892, 0.3806297, 0.4414586, 0.5451512]
            qs_rt = quantile(sim_data[:,2], probs)
            @test qs_rt ≈ rt_ground_truth atol = .005
    
            θ_ground_truth = [0.4052218, 1.4416240, 4.1459015, 4.7103539, 5.0401302, 5.2995585, 5.5305477, 5.7569943, 5.9982289] 
            qs_θ = quantile(sim_data[:,1], probs)
            @test qs_θ ≈ θ_ground_truth rtol = .005
        end
    end
end


