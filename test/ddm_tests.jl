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
        test_val1 = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 1, .5)
        @test test_val1 ≈ 2.129834 atol = 1e-1

        test_val2 = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 2, .5)
        @test test_val2 ≈ 0.2889796 atol = 1e-1

        test_val3 = pdf(DDM(;ν=.8, α=.5, z=.3, τ=.2, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 1, .35)
        @test test_val3 ≈ 0.6633653 atol = 1e-1

        test_val4 = pdf(DDM(;ν=.8, α=.5, z=.3, τ=.2, η = 0.08, sz = 0.0, st = 0.0, σ = 1.0), 2, .35)
        @test test_val4 ≈ 0.4449858 atol = 1e-1
        #try a bunch of combinations from values from rtdists
        sz_values = [0.00,0.05, 0.1, 0.2, 0.3]
        st_values = [0.00,0.05, 0.1, 0.2, 0.3]
        η_values  =  [0.00,0.08, 0.9, 0.13, 0.16]
        combinations = [(sz, st, η) for sz in sz_values for st in st_values for η in η_values]

        test_vals_upper = [2.131129, 2.129384, 2.124152, 2.101896, 2.065028, 2.540625, 
        2.538263, 2.531186, 2.501185, 2.45186, 3.025843, 3.022579, 3.01281, 
        2.971536, 2.90416, 2.866027, 2.865954, 2.865732, 2.864744, 2.862934, 
        1.910684, 1.910636, 1.910488, 1.909829, 1.908623, 2.129834, 2.128092, 
        2.122868, 2.100647, 2.063835, 2.539411, 2.537052, 2.529984, 2.500024, 
        2.450762, 3.024903, 3.021643, 3.011883, 2.970646, 2.903329, 2.865857, 
        2.865785, 2.865565, 2.864583, 2.862783, 1.910572, 1.910523, 1.910377, 
        1.909722, 1.908522, 1.983902, 1.98245, 1.978094, 1.95952, 1.9286, 
        2.399602, 2.397588, 2.391552, 2.365916, 2.323594, 2.913938, 2.911015, 
        2.902262, 2.86524, 2.804655, 2.846349, 2.846329, 2.846268, 2.845959, 
        2.845273, 1.897566, 1.897553, 1.897512, 1.897306, 1.896849, 2.127715, 
        2.125978, 2.120768, 2.098603, 2.061881, 2.537423, 2.535069, 2.528017, 
        2.498121, 2.448964, 3.023364, 3.020108, 3.010363, 2.969186, 2.901967, 
        2.86558, 2.865508, 2.865291, 2.864319, 2.862536, 1.910387, 1.910339, 
        1.910194, 1.909546, 1.908358, 2.125965, 2.124231, 2.119032, 2.096913, 
        2.060266, 2.53578, 2.53343, 2.526391, 2.496549, 2.447477, 3.02209, 
        3.018839, 3.009105, 2.967979, 2.900839, 2.865351, 2.86528, 2.865064, 
        2.864101, 2.862332, 1.910234, 1.910187, 1.910043, 1.909401, 1.908221]

        test_vals_lower = [0.288417, 0.288327, 0.288055, 0.286843, 0.284638, 0.343836, 
        0.343781, 0.343611, 0.342803, 0.341161, 0.409503, 0.409585, 0.409821, 
        0.410666, 0.411506, 0.387875, 0.389054, 0.3926, 0.407855, 0.43373, 
        0.258583, 0.25937, 0.261734, 0.271903, 0.289153, 0.28898, 0.288889, 
        0.288615, 0.287391, 0.285169, 0.344436, 0.34438, 0.344207, 0.343387, 
        0.341724, 0.410134, 0.410215, 0.410447, 0.411278, 0.412097, 0.388387, 
        0.389566, 0.393111, 0.408359, 0.434222, 0.258925, 0.259711, 0.262074, 
        0.272239, 0.289481, 0.354834, 0.354647, 0.354081, 0.351634, 0.34743, 
        0.415484, 0.415315, 0.414806, 0.412568, 0.408615, 0.486035, 0.485984, 
        0.485823, 0.484998, 0.483146, 0.451208, 0.452322, 0.455669, 0.470067, 
        0.494477, 0.300805, 0.301548, 0.303779, 0.313378, 0.329651, 0.289902, 
        0.28981, 0.289531, 0.28829, 0.286039, 0.345418, 0.34536, 0.345183, 
        0.344343, 0.342648, 0.411169, 0.411247, 0.411475, 0.412283, 0.413065, 
        0.389227, 0.390406, 0.393948, 0.409185, 0.435029, 0.259485, 0.26027, 
        0.262632, 0.27279, 0.290019, 0.290664, 0.290571, 0.290289, 0.289034, 
        0.286759, 0.346231, 0.346172, 0.34599, 0.345134, 0.343412, 0.412025, 
        0.412102, 0.412325, 0.413114, 0.413866, 0.389923, 0.391101, 0.394641, 
        0.409869, 0.435697, 0.259949, 0.260734, 0.263094, 0.273246, 0.290465]

        for (i, (sz, st, η)) in enumerate(combinations)
            # Compute the pdf for the current values of sz, st, and η
            pdf_value = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η=η, sz=sz, st=st, σ = 1.0), 1, .5)
            @test test_vals_upper[i] ≈ pdf_value atol = 1e-1
            pdf_value = pdf(DDM(;ν=2.0, α=1.0, z=.5, τ=.3, η=η, sz=sz, st=st, σ = 1.0), 2, .5)
            @test test_vals_lower[i] ≈ pdf_value atol = 1e-1
        end

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
