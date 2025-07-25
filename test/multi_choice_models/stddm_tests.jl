@safetestset "stDDM Tests" begin
    @safetestset "stDDM predictions" begin
        using SequentialSamplingModels
        using Test
        using StableRNGs
        rng = StableRNG(15)

        parms = (ν = [1.0, 1.2], α = 0.8, τ = 0.30, s = 0.0, z = 0.50)
        model = stDDM(; parms...)
        choice, rt = rand(rng, model, 10_000)
        # use validated simulator
        #Lombardi, G., & Hare, T. Piecewise constant averaging methods allow for fast and 
        # accurate hierarchical Bayesian estimation of drift diffusion models with 
        # time-varying evidence accumulation rates. PsyArXiv, (2021). https://doi.org/10.31234/osf.io/5azyx
        #code to validate
        #https://github.com/galombardi/method_HtSSM_aDDM/tree/master/RecoveryFitting/stDDM

        @test mean(choice .== 1) ≈ 0.8192 atol = 1e-1
        @test mean(rt[choice .== 1]) ≈ 0.4332 atol = 1e-1
        @test mean(rt[choice .== 2]) ≈ 0.4555 atol = 1e-1
        @test std(rt[choice .== 1]) ≈ 0.1106 atol = 1e-1
        @test std(rt[choice .== 2]) ≈ 0.1318 atol = 1e-1

        parms = (ν = [2, 1], α = 1.5, τ = 0.30, s = 0.20, z = 0.50)
        model = stDDM(; parms...)
        choice, rt = rand(rng, model, 10_000)

        #R:      0.8883 0.6639 0.6303 0.2377 0.3146
        #Python: 0.8918 0.6694 0.6285 0.2478 0.3156
        #Julia:  0.8897 0.6635 0.6298 0.2464 0.3037

        @test mean(choice .== 1) ≈ 0.8883 atol = 1e-1
        @test mean(rt[choice .== 1]) ≈ 0.6639 atol = 1e-1
        @test mean(rt[choice .== 2]) ≈ 0.6303 atol = 1e-1
        @test std(rt[choice .== 1]) ≈ 0.2377 atol = 1e-1
        @test std(rt[choice .== 2]) ≈ 0.3146 atol = 1e-1

        parms = (ν = [0.7, 0.9], α = 2, τ = 0.40, s = 0.60, z = 0.40)
        model = stDDM(; parms...)
        choice, rt = rand(rng, model, 10_000)

        #R:      0.7417 1.1632 1.0788 0.5329 0.6273
        #Python: 0.7345 1.1686 1.0444 0.5515 0.6125
        #Julia:  0.7335 1.1642 1.0546 0.533 0.6285

        @test mean(choice .== 1) ≈ 0.7417 atol = 1e-1
        @test mean(rt[choice .== 1]) ≈ 1.1632 atol = 1e-1
        @test mean(rt[choice .== 2]) ≈ 1.0788 atol = 1e-1
        @test std(rt[choice .== 1]) ≈ 0.5329 atol = 1e-1
        @test std(rt[choice .== 2]) ≈ 0.6273 atol = 1e-1
    end

    #   R:
    #   Rstddm <- function(d_v, d_h, thres, nDT, tIn, bias, sd_n, N, seed = NULL) {
    #   if (!is.null(seed)) set.seed(seed)

    #   T <- 6 # Total time
    #   dt <- 0.001 # Time step
    #   lt <- as.integer(T / dt) # Number of time steps
    #   vec_tHealth <- rep(1, lt)
    #   vec_tVal <- rep(1, lt)
    #   aux <- abs(as.integer(tIn / dt))

    #   # Adjusting health and value vectors based on tIn
    #   if (tIn > 0) {
    #     vec_tVal[1:aux] <- 0
    #   } else if (tIn < 0) {
    #     vec_tHealth[1:aux] <- 0
    #   }

    #   simulate_ddm <- function() {
    #     X <- bias * thres # Initial accumulation value
    #     flag <- FALSE
    #     cont <- 0
    #     vecOut <- T

    #     Sigma = matrix(c(1,0,0,1),2,2)
    #     d_vht <- MASS::mvrnorm(1, c(d_v, d_h), Sigma)
    #     while (!flag && cont < lt) {
    #       noise <- rnorm(1, mean = 0, sd = sd_n) * sqrt(dt)
    #       X <- X + (d_vht[[1]] *1* vec_tVal[cont + 1] + d_vht[[2]] *1* vec_tHealth[cont + 1]) * dt + noise #note 1's are VD and HD

    #       if (X > thres) {
    #         flag <- TRUE
    #         vecOut <- nDT + cont * dt
    #       } else if (X < 0) {
    #         flag <- TRUE
    #         vecOut <- -nDT - cont * dt
    #       }
    #       cont <- cont + 1
    #     }

    #     return(vecOut)
    #   }

    #   results <- numeric(N)
    #   for (i in 1:N) {
    #     results[i] <- simulate_ddm()
    #   }

    #   return(results)
    # }
    #res = Rstddm(d_v = 1, d_h = 1.2, thres = 0.8, nDT = 0.30, tIn = 0.0, bias = .5, sd_n = 1, N = 10000, seed = 8414)
    #res = Rstddm(d_v = 2, d_h = 1, thres = 1.5, nDT = 0.30, tIn = 0.2, bias = .5, sd_n = 1, N = 10000, seed = 8414)
    #res = Rstddm(d_v = .7, d_h = .9, thres = 2, nDT = 0.40, tIn = 0.6, bias = .5, sd_n = 1, N = 10000, seed = 8414)
    #
    #   Python:
    # import numpy as np
    # def pystddm(d_v, d_h, thres, nDT, tIn, bias, sd_n, N, seed=None):
    #     # Set a seed for reproducibility if provided
    #     if seed is not None:
    #         np.StableRNGs.seed(seed)

    #     # Constants and Preparations
    #     T = 6  # Total time
    #     dt = 0.001  # Time step
    #     lt = int(T / dt)  # Number of time steps
    #     vec_tHealth = np.ones(lt)
    #     vec_tVal = np.ones(lt)
    #     aux = abs(int(tIn / dt))

    #     # Adjusting health and value vectors based on tIn
    #     if tIn > 0:
    #         vec_tVal[:aux] = 0
    #     elif tIn < 0:
    #         vec_tHealth[:aux] = 0

    #     def simulate_ddm():
    #         X = bias * thres  # Initial accumulation value
    #         flag = False
    #         cont = 0
    #         vecOut = T

    #         Sigma = np.array([[1, 0], [0, 1]])
    #         d_vht = np.StableRNGs.multivariate_normal([d_v, d_h], Sigma)

    #         while not flag and cont < lt:
    #             noise = np.StableRNGs.normal(0, sd_n) * np.sqrt(dt)
    #             X += (d_vht[0] * vec_tVal[cont] + d_vht[1] * vec_tHealth[cont]) * dt + noise

    #             if X > thres:
    #                 flag = True
    #                 vecOut = nDT + cont * dt
    #             elif X < 0:
    #                 flag = True
    #                 vecOut = -nDT - cont * dt
    #             cont += 1

    #         return vecOut

    #     # Apply the simulation for each output element
    #     results = [simulate_ddm() for _ in range(N)]

    #     return results

    #res = pystddm(d_v = 1, d_h = 1.2, thres = 0.8, nDT = 0.30, tIn = 0.0, bias = .5, sd_n = 1, N = 10000, seed = 8414)
    #res = pystddm(d_v = 2, d_h = 1, thres = 1.5, nDT = 0.30, tIn = 0.2, bias = .5, sd_n = 1, N = 10000, seed = 8414)
    #res = pystddm(
    #     d_v=.7,    # Weight of the drift for the first attribute
    #     d_h=.9,    # Weight of the drift for the second attribute
    #     thres=2,  # Decision threshold
    #     nDT=0.4,  # Additional decision time
    #     tIn=0.60,  # Time influence of the second attribute
    #     bias=.4,   # Initial value (starting point) of the accumulation
    #     sd_n=1,   # Standard deviation of the noise
    #     N=10000,    # Number of simulations
    #     seed=8414  # Seed for reproducibility
    # )

    @safetestset "simulate" begin
        using SequentialSamplingModels
        using Test
        using StableRNGs

        rng = StableRNG(1554)
        α = 0.80
        Δt = 0.0001
        dist = stDDM(; α, ν = [3, 3])

        time_steps, evidence = simulate(rng, dist; Δt)

        @test time_steps[1] ≈ 0
        @test length(time_steps) == length(evidence)
        @test evidence[end] ≈ α atol = 0.05

        dist = stDDM(; α, ν = [-3, -3])
        time_steps, evidence = simulate(rng, dist; Δt)
        @test evidence[end] ≈ 0.0 atol = 0.05
    end
end
