@safetestset "DDM Tests" begin
    @safetestset "DDM pdf 1" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(654)
        include("KDE.jl")

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3, η = 0.00, sz = 0.00, st = 0.00, σ = 1.0) 
        choice,rt = rand(dist, 10^6)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kernel(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .05

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .05
    end

    @safetestset "DDM pdf 2" begin
        using SequentialSamplingModels
        using Test
        using Random
        include("KDE.jl")
        Random.seed!(750)

        dist = DDM(ν=2.0, α = 1.5, z = .5, τ = .30, η = 0.00, sz = 0.00, st = 0.00, σ = 1.0) 
        choice,rt = rand(dist, 10^6)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kernel(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .02

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .05
    end

    @safetestset "Full DDM pdf 1 (η)" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(654)
        include("KDE.jl")

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3, η = 0.08, sz = 0.00, st = 0.00, σ = 1.0) 
        choice,rt = rand(dist, 10^6)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kernel(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .05

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .05
    end

    @safetestset "Full DDM pdf 2 (sz)" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(654)
        include("KDE.jl")

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3, η = 0.00, sz = 0.10, st = 0.00, σ = 1.0) 
        choice,rt = rand(dist, 10^6)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kernel(rt1)
        x = range(.301, 1.5, length=100)
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .05

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .05
    end

    @safetestset "Full DDM pdf 3 (st)" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(654)
        include("KDE.jl")

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3, η = 0.00, sz = 0.00, st = 0.10, σ = 1.0) 
        choice,rt = rand(dist, 10^6)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kernel(rt1)
        x = range(.301 + .10, 1.5, length=100) # τ and st start the lower bound
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .05

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .05
    end

    @safetestset "Full DDM pdf 4 (η,sz,st)" begin
        using SequentialSamplingModels
        using Test
        using Random
        Random.seed!(654)
        include("KDE.jl")

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3, η = 0.08, sz = 0.10, st = 0.10, σ = 1.0) 
        choice,rt = rand(dist, 10^6)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        approx_pdf = kernel(rt1)
        x = range(.301 + .10, 1.5, length=100) # τ and st start the lower bound
        y′ = pdf(approx_pdf, x) * p1
        y = pdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .05

        rt2 = rt[choice .== 2]
        approx_pdf = kde(rt2)
        y′ = pdf(approx_pdf, x) * p2
        y = pdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .05
    end

    @safetestset "DDM cdf 1" begin
        using SequentialSamplingModels
        using Test
        using StatsBase
        using Random
        Random.seed!(7540)

        dist = DDM(ν=1.0, α = .8, z = .5, τ = .3, η = 0.00, sz = 0.00, st = 0.00, σ = 1.0) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        ecdf1 = ecdf(rt1)
        x = range(.301, 1.0, length=100)
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

        dist = DDM(ν=2.0, α = 1.5, z = .5, τ = .30, η = 0.00, sz = 0.00, st = 0.00, σ = 1.0) 
        choice,rt = rand(dist, 10^5)
        rt1 = rt[choice .== 1]
        p1 = mean(choice .== 1)
        p2 = 1 - p1
        ecdf1 = ecdf(rt1)
        x = range(.301, 1.0, length=100)
        y′ = ecdf1.(x) * p1
        y = cdf.(dist, (1,), x)
        @test y′ ≈ y rtol = .01

        rt2 = rt[choice .== 2]
        ecdf2 = ecdf(rt2)
        y′ = ecdf1.(x) * p2
        y = cdf.(dist, (2,), x)
        @test y′ ≈ y rtol = .02 #note the change in tol
    end

    # @safetestset "_exp_pnorm" begin
    #     using SequentialSamplingModels: _exp_pnorm
    #     using Test
    #     true_values = [0.003078896, 0.021471654, 0.067667642, 0.113863630, 0.132256388, 0.008369306, 0.058366006, 0.183939721, 0.309513435, 0.359510135, 0.022750132,
    #         0.158655254, 0.500000000, 0.841344746, 0.977249868, 0.061841270, 0.431269694, 1.359140914, 2.287012135, 2.656440558, 0.168102001, 1.172312572,
    #         3.694528049, 6.216743527, 7.220954098]
    #     cnt = 0
    #     for v1 ∈ -2:2, v2 ∈ -2:2
    #         cnt += 1
    #         @test true_values[cnt] ≈ _exp_pnorm(v1, v2) atol = 1e-5
    #     end
    # end
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

    # @safetestset "K_large" begin
    #     using SequentialSamplingModels
    #     using SequentialSamplingModels: _K_large
    #     using Test
        
    #     test_val1 = _K_large(DDM(;ν=1.5, α=.50, z=.25, τ=.50), .75; ϵ = 1e-10)
    #     @test test_val1 ≈ 3.0

    #     test_val2 = _K_large(DDM(;ν=1.5, α=.50, z=.25, τ=.50), .501; ϵ = 1e-10)
    #     @test test_val2 ≈ 36
    # end
    # K_large = function(t, v, a, w, epsilon)
    #     {
    #       sqrtL1 = sqrt(1/t) * a/pi
    #       sqrtL2 = sqrt(pmax(1, -2/t*a*a/pi/pi * (log(epsilon*pi*t/2 * (v*v + pi*pi/a/a)) + v*a*w + v*v*t/2)))
    #       ceiling(pmax(sqrtL1, sqrtL2))
    #     }
    # K_large(.25, 1.5, .5, .25, 1e-10)
    # K_large(.001, 1.5, .5, .25, 1e-10)

    # @safetestset "K_small" begin
    #     using SequentialSamplingModels
    #     using SequentialSamplingModels: _K_small
    #     using Test
        
    #     # function(t, v, a, w, epsilon)
    #     test_val1 = _K_small(DDM(;ν=1.5, α=.50, z=.25, τ=.50, η = 0.00, sz = 0.0, st = 0.0, σ = 1.0), .75; ϵ = 1e-10)
    #     @test test_val1 ≈ 16

    #     test_val2 = _K_small(DDM(;ν=1.5, α=.50, z=.25, τ=.50, η = 0.00, sz = 0.0, st = 0.0, σ = 1.0), .501; ϵ = 1e-10)
    #     @test test_val2 ≈ 16

    #     test_val3 = _K_small(DDM(;ν=0.50, α=2.50, z=.50, τ=.30, η = 0.00, sz = 0.0, st = 0.0, σ = 1.0), .400; ϵ = 1e-10)
    #     @test test_val3 ≈ 10
    # end
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

    # @safetestset "_P_upper" begin
    #     using SequentialSamplingModels
    #     using SequentialSamplingModels: _P_upper
    #     using Test
        
    #     test_val1 = _P_upper(1., .5, .5)
    #     @test test_val1 ≈ 0.3775407 atol = 1e-5

    #     test_val2 = _P_upper(-1., .75, .25)
    #     @test test_val2 ≈ .8693188 atol = 1e-5

    #     test_val3 = _P_upper(-1e10, .75, .25)
    #     @test test_val3 ≈ 1

    #     test_val4 = _P_upper(eps(), .75, .25)
    #     @test test_val4 ≈ .75
    # end

    # @safetestset "_Fs_lower" begin
    #     using SequentialSamplingModels
    #     using SequentialSamplingModels: _Fs_lower
    #     using Test
        
    #     test_val1 = _Fs_lower(DDM(;ν=1.5, α=.50, z=.25, τ=.50), 10, .75)
    #     @test test_val1 ≈ 0.5955567 atol = 1e-5

    #     test_val2 = _Fs_lower(DDM(;ν=1.5, α=.50, z=.25, τ=.50), 10, .501)
    #     @test test_val2 ≈ 6.393096e-05 atol = 1e-8
    # end

    # Fs_lower = function(t, v, a, w, K)
    #     {
    #       if(abs(v) < sqrt(.Machine$double.eps)) # zero drift case
    #         return(Fs0_lower(t, a, w, K))
    #       S1 = S2 = numeric(length(t))
    #       sqt = sqrt(t)
    #       for(k in K:1)
    #       {
    #         S1 = S1 + exp_pnorm(2*v*a*k, -sign(v)*(2*a*k+a*w+v*t)/sqt) -
    #           exp_pnorm(-2*v*a*k - 2*v*a*w, sign(v)*(2*a*k+a*w-v*t)/sqt)
    #         S2 = S2 + exp_pnorm(-2*v*a*k, sign(v)*(2*a*k-a*w-v*t)/sqt) -
    #           exp_pnorm(2*v*a*k - 2*v*a*w, -sign(v)*(2*a*k-a*w+v*t)/sqt)
    #       }
    #       Pu(v, a, w) + sign(v) * ((pnorm(-sign(v) * (a*w+v*t)/sqt) -
    #                                   exp_pnorm(-2*v*a*w, sign(v) * (a*w-v*t)/sqt)) + S1 + S2)
    #     }
        
    #     Fs_lower(.25, 1.5, .5, .25, 10)
    #     Fs_lower(.001, 1.5, .5, .25, 10)

    # @safetestset "_Fl_lower" begin
    #     using SequentialSamplingModels
    #     using SequentialSamplingModels: _Fl_lower
    #     using Test
        
    #     test_val1 = _Fl_lower(DDM(;ν=1.5, α=.50, z=.25, τ=.50), 10, .75)
    #     @test test_val1 ≈ 0.5955567 atol = 1e-5

    #     test_val2 = _Fl_lower(DDM(;ν=1.5, α=.50, z=.25, τ=.50), 10, .501)
    #     @test test_val2 ≈  0.001206451 atol = 1e-8
    # end
    # Fl_lower = function(t, v, a, w, K)
    #     {
    #       F = numeric(length(t))
    #       for(k in K:1)
    #         F = F - k / (v*v + k*k*pi*pi/a/a) * exp(-v*a*w - 1/2*v*v*t - 1/2*k*k*pi*pi/a/a*t) * sin(pi*k*w)
    #       Pu(v, a, w) + 2*pi/a/a * F
    #     } 
        
    #     Fl_lower(.25, 1.5, .5, .25, 10)
    #     Fl_lower(.001, 1.5, .5, .25, 10)

    @safetestset "pdf" begin
        using SequentialSamplingModels
        using Test
        # tested against rtdists
        test_val1 = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η = 0.00, sz = 0.0, st = 0.0, σ = 1.0), 1, .5)
        @test test_val1 ≈ 2.131129 atol = 1e-5

        test_val2 = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η = 0.00, sz = 0.0, st = 0.0, σ = 1.0), 2, .5)
        @test test_val2 ≈ 0.2884169 atol = 1e-5

        test_val3 = pdf(DDM(;ν=.8, α=.5, z=.3, τ=.2, η = 0.00, sz = 0.0, st = 0.0, σ = 1.0), 1, .35)
        @test test_val3 ≈ 0.6635714 atol = 1e-5

        test_val4 = pdf(DDM(;ν=.8, α=.5, z=.3, τ=.2, η = 0.00, sz = 0.0, st = 0.0, σ = 1.0), 2, .35)
        @test test_val4 ≈ 0.4450956 atol = 1e-5
    end

    @safetestset "full pdf" begin
        using SequentialSamplingModels
        using Test
        # tested against rtdists
        ## eta
        # test_val1 = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 1, .5)
        # @test test_val1 ≈ 2.129834 atol = 1e-1

        # test_val2 = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 2, .5)
        # @test test_val2 ≈ 0.2889796 atol = 1e-1

        # test_val3 = pdf(DDM(;ν=.8, α=.5, z=.3, τ=.2, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 1, .35)
        # @test test_val3 ≈ 0.6633653 atol = 1e-1

        # test_val4 = pdf(DDM(;ν=.8, α=.5, z=.3, τ=.2, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 2, .35)
        # @test test_val4 ≈ 0.4449858 atol = 1e-1
        #try a bunch of combinations from values from rtdists
        sz_values = [0.00, 0.05, 0.10, 0.30]
        st_values = [0.00, 0.05, 0.10, 0.30]
        η_values  = [0.00, 0.08, 0.16]

        combinations = [(sz, st, η) for sz in sz_values for st in st_values for η in η_values]

        # for (sz, st, η) in combinations println("$sz, $st, $η") end

        test_vals_upper = [1.81597, 1.814461, 1.809956, 1.974649, 1.973321, 1.969353, 
        2.123612, 2.122583, 2.119502, 1.624452, 1.624509, 1.62468, 1.815153, 
        1.813646, 1.809147, 1.973663, 1.972337, 1.968375, 2.122466, 2.121439, 
        2.118364, 1.624535, 1.624593, 1.624764, 1.812704, 1.811203, 1.806723, 
        1.970707, 1.969387, 1.965443, 2.119034, 2.118012, 2.114953, 1.624786, 
        1.624843, 1.625014, 1.785586, 1.784153, 1.779875, 1.938027, 1.936776, 
        1.933035, 2.081122, 2.080158, 2.077274, 1.627546, 1.627602, 1.627769]

        test_vals_lower = [0.074023, 0.074416, 0.075599, 0.080491, 0.080889, 0.082087, 
        0.086563, 0.08696, 0.088155, 0.066216, 0.066472, 0.067242, 0.074072, 
        0.074465, 0.075648, 0.080556, 0.080954, 0.082151, 0.086653, 0.087051, 
        0.088245, 0.066413, 0.066669, 0.067439, 0.074218, 0.074611, 0.075792, 
        0.080751, 0.081149, 0.082345, 0.086923, 0.08732, 0.088513, 0.067004, 
        0.06726, 0.068031, 0.075803, 0.076192, 0.077358, 0.082871, 0.083265, 
        0.084447, 0.089867, 0.09026, 0.091442, 0.073773, 0.074034, 0.074818]

        for (i, (sz, st, η)) in enumerate(combinations)
            # Print out the current parameter set
           println("Parameter set [$i]: sz = $sz, st = $st, η = $η")
           # Compute the pdf for the current values of sz, st, and η
           pdf_value = pdf(DDM(;ν=2.0, α=1.6, z=.5, τ=.2, η=η, sz=sz, st=st, σ = 1.0), 1, .5)
           println("Test value upper[$i]: ", test_vals_upper[i])
           println("PDF value upper: ", pdf_value)
           @test test_vals_upper[i] ≈ pdf_value atol = 1e-1
           pdf_value = pdf(DDM(;ν=2.0, α=1.6, z=.5, τ=.2, η=η, sz=sz, st=st, σ = 1.0), 2, .5)
           println("Test value lower[$i]: ", test_vals_lower[i])
           println("PDF value lower: ", pdf_value)
           @test test_vals_lower[i] ≈ pdf_value atol = 1e-1
       end

#     sz_values = [0.00]
#     st_values = [0.00]
#     η_values  = [0.00, 0.08, 0.16]

#     combinations = [(sz, st, η) for sz in sz_values for st in st_values for η in η_values]

#     test_vals_upper = [1.81597, 1.814461, 1.809956]
#     test_vals_lower = [0.074023, 0.074416, 0.075599]

#     for (i, (sz, st, η)) in enumerate(combinations)
#         # Print out the current parameter set
#        println("Parameter set [$i]: sz = $sz, st = $st, η = $η")
#        # Compute the pdf for the current values of sz, st, and η
#        pdf_value = pdf(DDM(;ν=2.0, α=1.6, z=.5, τ=.2, η=η, sz=sz, st=st, σ = 1.0), 1, .5)
#        println("Test value upper[$i]: ", test_vals_upper[i])
#        println("PDF value upper: ", pdf_value)
#        @test test_vals_upper[i] ≈ pdf_value atol = 1e-3
#        pdf_value = pdf(DDM(;ν=2.0, α=1.6, z=.5, τ=.2, η=η, sz=sz, st=st, σ = 1.0), 2, .5)
#        println("Test value lower[$i]: ", test_vals_lower[i])
#        println("PDF value lower: ", pdf_value)
#        @test test_vals_lower[i] ≈ pdf_value atol = 1e-3
#    end

        # 0.05, 0.05, 0.08
        # pdf_value = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η=.08, sz=.05, st=.05, σ = 1.0), 1, .5)
        # @test 2.537052 ≈ pdf_value atol = 1e-1

    end

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using Random 

        Random.seed!(8477)
        α = .80
        dist = DDM(;α, ν=3)

        time_steps,evidence = simulate(dist; Δt = .0001)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ α atol = .010

        dist = DDM(;α, ν=-3)
        time_steps,evidence = simulate(dist; Δt = .0001)
        @test evidence[end] ≈ 0.0 atol = .010
    end

end
