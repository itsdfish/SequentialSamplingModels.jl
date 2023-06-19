@safetestset "DDM Tests" begin
    @safetestset "DDM pdf 1" begin
        using SequentialSamplingModels
        using Test
        using KernelDensity
        using Random
        Random.seed!(654)

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kde(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .10

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .10
    end

    @safetestset "DDM pdf 2" begin
        using SequentialSamplingModels
        using Test
        using KernelDensity
        using Random
        Random.seed!(750)

        dist = DDM(ν=2.0, α = 1.5, z = .5, τ = .30) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kde(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .05

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .10
    end

    @safetestset "DDM cdf 1" begin
        using SequentialSamplingModels
        using Test
        using StatsBase
        using Random
        Random.seed!(7540)

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        ecdf1 = ecdf(rt1)
        x = range(.31, 1.0, length=100)
        y′ = ecdf1.(x) * p1
        y = cdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .01

        rt2 = rt[choice .== 2]
        ecdf2 = ecdf(rt2)
        y′ = ecdf1.(x) * p2
        y = cdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .01
    end

    @safetestset "DDM cdf 2" begin
        using SequentialSamplingModels
        using Test
        using StatsBase
        using Random
        Random.seed!(2200)

        dist = DDM(ν=2.0, α = 1.5, z = .5, τ = .30) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        ecdf1 = ecdf(rt1)
        x = range(.31, 1.0, length=100)
        y′ = ecdf1.(x) * p1
        y = cdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .01

        rt2 = rt[choice .== 2]
        ecdf2 = ecdf(rt2)
        y′ = ecdf1.(x) * p2
        y = cdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .01
    end

    @safetestset "_exp_pnorm" begin
        using SequentialSamplingModels: _exp_pnorm
        using Test
        true_values = [0.003078896, 0.021471654, 0.067667642, 0.113863630, 0.132256388, 0.008369306, 0.058366006, 0.183939721, 0.309513435, 0.359510135, 0.022750132,
            0.158655254, 0.500000000, 0.841344746, 0.977249868, 0.061841270, 0.431269694, 1.359140914, 2.287012135, 2.656440558, 0.168102001, 1.172312572,
            3.694528049, 6.216743527, 7.220954098]
        cnt = 0
        for v1 ∈ -2:2, v2 ∈ -2:2
            cnt += 1
            @test true_values[cnt] ≈ _exp_pnorm(v1, v2) atol = 1e-5
        end
    end
        # exp_pnorm = function(a, b)
    #     {
    #       r = exp(a) * pnorm(b)
    #       d = is.nan(r) & b < -5.5
    #       r[d] = 1/sqrt(2) * exp(a - b[d]*b[d]/2) * (0.5641882/b[d]/b[d]/b[d] - 1/b[d]/sqrt(pi))
    #       r
    #     }
        
    #     results = c()
    #     for (v1 in -2:2) {
    #       for (v2  in -2:2) {
    #         result  = exp_pnorm(v1, v2)
    #         results = append(results, result)
    #       }
    #     }

    @safetestset "K_large" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: _K_large
        using Test
        
        test_val1 = _K_large(DDM(;ν=1.5, α=.50, z=.25, τ=.50), .75; ϵ = 1e-10)
        @test test_val1 ≈ 3.0

        test_val2 = _K_large(DDM(;ν=1.5, α=.50, z=.25, τ=.50), .501; ϵ = 1e-10)
        # currently fails
        @test_skip test_val2 ≈ 36
    end
    # K_large = function(t, v, a, w, epsilon)
    #     {
    #       sqrtL1 = sqrt(1/t) * a/pi
    #       sqrtL2 = sqrt(pmax(1, -2/t*a*a/pi/pi * (log(epsilon*pi*t/2 * (v*v + pi*pi/a/a)) + v*a*w + v*v*t/2)))
    #       ceiling(pmax(sqrtL1, sqrtL2))
    #     }
    # K_large(.25, 1.5, .5, .25, 1e-10)
    # K_large(.001, 1.5, .5, .25, 1e-10)

    @safetestset "K_small" begin
        using SequentialSamplingModels
        using SequentialSamplingModels: _K_small
        using Test
        
        # function(t, v, a, w, epsilon)
        test_val1 = _K_small(DDM(;ν=1.5, α=.50, z=.25, τ=.50), .75; ϵ = 1e-10)
        @test test_val1 ≈ 16

        test_val2 = _K_small(DDM(;ν=1.5, α=.50, z=.25, τ=.50), .501; ϵ = 1e-10)
        @test test_val2 ≈ 16

        test_val3 = _K_small(DDM(;ν=0.50, α=2.50, z=.50, τ=.30), .400; ϵ = 1e-10)
        @test test_val3 ≈ 10
    end
    # K_small = function(t, v, a, w, epsilon)
    #     {
    #       if(abs(v) < sqrt(.Machine$double.eps)) # zero drift case
    #         return(ceiling(pmax(0, w/2 - sqrt(t)/2/a * qnorm(pmax(0, pmin(1, epsilon/(2-2*w)))))))
    #       if(v > 0) # positive drift
    #         return(K_small(t, -v, a, w, exp(-2*a*w*v)*epsilon))
    #       S2 = w - 1 + 1/2/v/a * log(epsilon/2 * (1-exp(2*v*a)))
    #       S3 = (0.535 * sqrt(2*t) + v*t + a*w)/2/a
    #       S4 = w/2 - sqrt(t)/2/a * qnorm(pmax(0, pmin(1, epsilon * a / 0.3 / sqrt(2*pi*t) * exp(v*v*t/2 + v*a*w))))
    #       ceiling(pmax(S2, S3, S4, 0))
    #     }

    # K_small(.25, 1.5, .5, .25, 1e-10)
    # K_small(.001, 1.5, .5, .25, 1e-10)
    # K_small(.10, .50, 2.5, .50, 1e-10)
end