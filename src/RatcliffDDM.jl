"""
    RatcliffDDM{T<:Real} <: SSM2D

    Model object for the Ratcliff Diffusion Model.

# Parameters
    - `ν`: drift rate. Average slope of the information accumulation process. The drift gives information about the speed and direction of the accumulation of information. Typical range: -5 < ν < 5
    - `α`: boundary threshold separation. The amount of information that is considered for a decision. Typical range: 0.5 < α < 2
    - `τ`: non-decision time. The duration for a non-decisional processes (encoding and response execution). Typical range: 0.1 < τ < 0.5 
    - `z`: starting point. Indicator of an an initial bias towards a decision. The z parameter is relative to a (i.e. it ranges from 0 to 1).
    - `η`:  across-trial-variability of drift rate. Typical range: 0 < η < 2. Default is 0.
    - `sz`: across-trial-variability of starting point. Typical range: 0 < sz < 0.5. Default is 0.
    - `st`: across-trial-variability of non-decision time. Typical range: 0 < st < 0.2. Default is 0.
    - `σ`: diffusion noise constant. Default is 1.

# Constructors 

    RatcliffDDM(ν, α, τ, z, η, sz, st, σ)
    
    RatcliffDDM(; ν = 1.00,
    α = 0.80,
    τ = 0.30,
    z = 0.25,
    η = 0.16,
    sz = 0.05,
    st = 0.10,
    σ = 1.0
    )

# Example 

````julia
using SequentialSamplingModels
dist = RatcliffDDM(ν = 1.0,α = 0.80,τ = 0.30,z = 0.25,η = 0.16,sz = 0.05,st = .10,σ = 1) 
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
````
    
# References
        
Ratcliff, R., & McKoon, G. (2008). The Diffusion Decision Model: Theory and Data for Two-Choice Decision Tasks. Neural Computation, 20(4), 873–922.
Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59–108. https://doi.org/10.1037/0033-295X.85.2.59
"""
mutable struct RatcliffDDM{T<:Real} <: SSM2D
    ν::T
    α::T
    τ::T
    z::T
    η::T
    sz::T
    st::T
    σ::T
end

function RatcliffDDM(ν, α, τ, z, η, sz, st, σ)
    return RatcliffDDM(promote(ν, α, τ, z, η, sz, st, σ)...)
end

function params(d::RatcliffDDM)
    (d.ν, d.α, d.τ, d.z,d.η, d.sz, d.st, d.σ)    
end

function RatcliffDDM(; ν = 1.00,
    α = 0.80,
    τ = 0.30,
    z = 0.25,
    η = 0.16,
    sz = 0.05,
    st = 0.10,
    σ = 1.0)
    return RatcliffDDM(ν, α, τ, z, η, sz, st, σ)
end

# probability density function over the lower boundary

# uses analytic integration of the likelihood function for variability in drift-rate 
# function pdf_sv(d::RatcliffDDM, choice, rt; ϵ::Real = 1.0e-12)
#     if choice == 1
#         (ν, α, τ, z, η, sz, st, σ) = params(d)
#         ν = -ν
#         z = 1 - z
#         return pdf_sv(RatcliffDDM(ν, α, τ, z, η, sz, st, σ), rt; ϵ)
#     end
#     return pdf_sv(d, rt; ϵ)
# end

function _pdf_sv(d::RatcliffDDM{T}, rt::Real; ϵ::Real = 1.0e-12) where {T<:Real}
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    if η == 0
        return _pdf(SequentialSamplingModels.DDM(ν, α, τ, z), rt; ϵ)
    end
    # if isless(ν,0)
    #     return pdf(SequentialSamplingModels.DDM(ν, α, τ, z), t; ϵ)  + (  ( (α*z*η)^2 - 2*ν*α*z - (ν^2)*t ) / (2*(η^2)*t+2)  ) - log(sqrt((η^2)*t+1)) + ν*α*z + (ν^2)*t*0.5
    # end
    # return pdf(SequentialSamplingModels.DDM(ν, α, τ, z), t; ϵ)  + (  ( (α*(1-z)*η)^2 + 2*ν*α*(1-z) - (ν^2)*t ) / (2*(η^2)*t+2)  ) - log(sqrt((η^2)*t+1)) - ν*α*(1-z) + (ν^2)*t*0.5
    return _pdf(SequentialSamplingModels.DDM(ν, α, τ, z), rt; ϵ)  + (  ( (α*z*η)^2 - 2*ν*α*z - (ν^2)*(rt-τ) ) / (2*(η^2)*(rt-τ)+2)  ) - log(sqrt((η^2)*(rt-τ)+1)) + ν*α*z + (ν^2)*(rt-τ)*0.5
end

function pdf(d::RatcliffDDM, choice, rt; ϵ::Real = 1.0e-12)
    if choice == 1
        (ν, α, τ, z, η, sz, st, σ) = params(d)
        return _pdf(RatcliffDDM(-ν, α, τ, 1-z, η, sz, st, σ), rt; ϵ)
    end
    return _pdf(d, rt; ϵ)
end

#use numerical integration for variability in non-decision time and bias (Ratcliff and Tuerlinckx, 2002)
function _pdf(d::RatcliffDDM{T}, rt; ϵ::Real = 1.0e-12, n_st::Int=2, n_sz::Int=2)  where {T<:Real}
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    if τ ≥ rt
        return T(NaN)
    end

    if st < 1.0e-3 
        st = 0
    end
    if sz  < 1.0e-3
        sz = 0
    end

    if sz==0
        if st==0 #sv=0,sz=0,st=0
            return  _pdf_sv(d, rt; ϵ)
        else #sv=0,sz=0,st=$
            return _simpson_1D(rt, ν, η, α, z, τ, ϵ, z, z, 0, τ-st/2., τ+st/2., n_st)
        end
    else #sz=$
        if st==0 #sv=0,sz=$,st=0
            return _simpson_1D(rt, ν, η, α, z, τ, ϵ, z-sz/2., z+sz/2., n_sz, τ, τ , 0)
        else     #sv=0,sz=$,st=$
            return _simpson_2D(rt, ν, η, α, z, τ, ϵ, z-sz/2., z+sz/2., n_sz, τ-st/2., τ+st/2., n_st)
        end
    end
end

#####################################################
#  Numerical Integration with Simpson's Method      #
#  https://en.wikipedia.org/wiki/Simpson%27s_rule   #
#####################################################

# Simpson's Method one dimentional case
function _simpson_1D(x::Real, ν::Real, η::Real, α::Real, z::Real, τ::Real, ϵ::Real, lb_z::Real, ub_z::Real, n_sz::Int, lb_t::Real, ub_t::Real, n_st::Int)
   
 #   @assert (n_sz & 1) == 0 && (n_st & 1) == 0     # n_st and n_sz have to be even

    n = max(n_st, n_sz)

    if n_st == 0 #integration over z
        hz = (ub_z-lb_z)/n
        ht = 0
        lb_t = τ
        ub_t = τ
    else        #integration over t
        hz = 0
        ht = (ub_t-lb_t)/n
        lb_z = z
        ub_z = z
    end

    S =  _pdf_sv(RatcliffDDM(ν, α, lb_t, lb_z, η, 0, 0, 1), x; ϵ)
    
    y = 0 
    z_tag = 0 
    t_tag = 0 

    for i in 1:n
        z_tag = lb_z + hz * i
        t_tag = lb_t + ht * i
       
        y = _pdf_sv(RatcliffDDM(ν, α, t_tag, z_tag, η, 0, 0, 1), x; ϵ)
           
        if isodd(i)
            S += 4 * y
        else
            S += 2 * y
        end

    end
    
    S = S - y  # the last term should be f(b) and not 2*f(b) so we subtract y
    S = S / ((ub_t - lb_t) + (ub_z - lb_z))  # the right function if pdf_sv()/sz or pdf_sv()/st
    
    return (ht + hz) * S / 3

end

# Simpson's Method two dimentional case
function _simpson_2D(x::Real, ν::Real, η::Real, α::Real, z::Real, τ::Real, ϵ::Real, lb_z::Real, ub_z::Real, n_sz::Int, lb_t::Real, ub_t::Real, n_st::Int)
 #   @assert (n_sz & 1) == 0 && (n_st & 1) == 0     # n_st and n_sz have to be even
 #   @assert (ub_t-lb_t)*(ub_z-lb_z)>0 && (n_sz*n_st)>0  # 2D-integration only

    ht = (ub_t-lb_t)/n_st
    S = _simpson_1D(x, ν, η, α, z, τ, ϵ, lb_z, ub_z, n_sz, 0, 0, 0)

    t_tag = 0
    y = 0 
    for i_t in 1:n_st
        t_tag = lb_t + ht * i_t
        y = _simpson_1D(x, ν, η, α, z, t_tag, ϵ, lb_z, ub_z, n_sz, 0, 0, 0)

        if isodd(i_t)
            S += 4 * y
        else
            S += 2 * y
        end
    end

    S = S - y  # the last term should be f(b) and not 2*f(b) so we subtract y
    S = S / (ub_t - lb_t)

    return ht * S / 3

end

logpdf(d::RatcliffDDM, choice, rt; ϵ::Real = 1.0e-12) = log(pdf(d, choice, rt; ϵ))

function logpdf(d::RatcliffDDM, data::T) where {T<:NamedTuple}
    return sum(logpdf.(d, data...))
end

function logpdf(dist::RatcliffDDM, data::Array{<:Tuple,1})
    LL = 0.0
    for d in data
        LL += logpdf(dist, d...)
    end
    return LL
end

logpdf(d::RatcliffDDM, data::Tuple) = logpdf(d, data...)

########################################################################################################################################################################
# Calculate Cumulative Distribution Function
#
#  This source codes are adpated from Henrik Singmann's Density.h (rtdists) &  Voss & Voss's density.c (fast-dm).
#
#  # References
#
# - Singmann H, Brown S, Gretton M, Heathcote A (2022). _rtdists: Response Time Distributions_. Rpackage version 0.11-5, <https://CRAN.R-project.org/package=rtdists>.
# - Voss, A., Rothermund, K., & Voss, J. (2004). Interpreting the parameters of the diffusion model: An empirical validation. *Memory and Cognition, 32(7)*, 1206-1220.
# - Ratcliff, R. (1978). A theory of memory retrieval. *Psychology Review, 85(2)*, 59-108.
# - Voss, A., Voss, J., & Lerche, V. (2015). Assessing cognitive processes with diffusion model analyses: A tutorial based on fast-dm-30. *Frontiers in Psychology, 6*, Article 336. https://doi.org/10.3389/fpsyg.2015.00336
#
################################################################################################################################################################

TUNE_PDE_DT_MIN = 1e-6
TUNE_PDE_DT_MAX = 1e-6
TUNE_PDE_DT_SCALE = 0.0

TUNE_DZ = 0.0
TUNE_DV = 0.0
TUNE_DT0 = 0.0

TUNE_INT_T0 = 0
TUNE_INT_Z = 0

precision_set = 0

function cdf(d::RatcliffDDM, choice, rt; ϵ::Real = 1.0e-12, precision::Real = 3)
    if choice == 1
        (ν, α, τ, z, η, sz, st, σ) = params(d)
        return cdf(DDM(-ν, α, τ, 1-z), rt; ϵ, precision) #over the upper boundary (g_plus)
    end

    return cdf(d, rt; ϵ, precision)
end

# cumulative density function over the lower boundary (g_minus)
function cdf(d::RatcliffDDM{T}, x::Real; ϵ::Real= 1.0e-6, precision::Real = 3) where {T<:Real}
    if d.τ ≥ t
        return T(NaN)
    end
    _set_precision(precision)
    DT = x - τ - 0.5 * z
    return _integral_τ_g_minus(x, d)
end

function _set_precision(precision::Real = 3)
    """
    Precision of calculation. 
    Corresponds roughly to the number of decimals of the predicted CDFs that are calculated accurately. Default is 3.
    The function adjusts various parameters used in the calculations based on the precision value. 	
    """
    global TUNE_PDE_DT_MIN = (-0.400825*precision-1.422813)^10
    global TUNE_PDE_DT_MAX = (-0.627224*precision+0.492689)^10
    global TUNE_PDE_DT_SCALE = (-1.012677*precision+2.261668)^10
    global TUNE_DZ = (-0.5*precision-0.033403)^10
    global TUNE_DV = (-1.0*precision+1.4)^10
    global TUNE_DT0 = (-0.5*precision-0.323859)^10

    global TUNE_INT_T0 = 0.089045 * exp(-1.037580*precision)
    global TUNE_INT_Z = 0.508061 * exp(-1.022373*precision)
  
    global precision_set = 1
end

function _integrate(F::Function, d::RatcliffDDM, a::Int, b::Int, step_width::Real)
    """
    These functions perform numerical integration using a specified function, range, and step width. The integrate function performs the integration sequentially
    """
    width = b - a  # integration width
    N = max(4, Int(width / step_width))  # N at least equals 4
    step = width / N
    x = a + 0.5 * step
    out = 0
    while x < b
        out += step * F(x, d)
        x += step
    end
    return out
end

function _g_minus_small_time(x::Real, d::RatcliffDDM, N::Int)
    """
    calculate the densities g- for the first exit time for small time
    """
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    DT = x - τ #make into decision time

    sum = 0.0
    for i = -N:N÷2
        d = 2*i + z
        sum += exp(-d*d / (2*DT)) * d
    end
    return sum / sqrt(2π*DT*DT*DT)
end

function _g_minus_large_time(x::Real, d::RatcliffDDM, N::Int)
    """
    calculate the densities g- for the first exit time for large time values
    """
    DT = x - τ #make into decision time

    sum = 0.0
    for i = 1:N
        d = i * π
        sum += exp(-0.5 * d*d * DT) * sin(d*z) * i
    end
    return sum * π
end

function _g_minus_no_var(x::Real, d::RatcliffDDM)
    """
    calculates the density g- when there is no variability in the input parameters.
    """
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    DT = x - τ #make into decision time

    N_small = 0
    N_large = 0
    simple = 0.0
    factor = exp(-α*z*ν - 0.5*ν*ν*DT) / (α*α)  # Front term in A3
    ϵ = ϵ / factor

    ta = x / (α*α)

    N_large = ceil(1 / (π*sqrt(DT)))
    if π*ta*ϵ < 1
        N_large = max(N_large, ceil(sqrt(-2*log(π*ta*ϵ) / (π*π*ta))))
    end

    if 2*sqrt(2*π*ta)*ϵ < 1
        N_small = ceil(max(sqrt(ta) + 1, 2 + sqrt(-2*ta*log(2*ϵ*sqrt(2*π*ta)))))
    else
        N_small = 2
    end

    if N_small < N_large
        simple = _g_minus_small_time(x / (α*α), z, N_small)
    else
        simple = _g_minus_large_time(x / (α*α), z, N_large)
    end

    out = isinf(factor) ? 0 : (factor * simple)
    return out
end

function _integral_v_g_minus(Real::x, d::RatcliffDDM; ϵ::Real = 1e-6)
    """
    calculates the integral of the density g- over the variable ν for a given set of input parameters. It takes into account variability in the parameters η
    """
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    DT = x - τ #make into decision time

    N_small = 0
    N_large = 0
    simple = 0.0
    factor = 1 / (α*α * sqrt(DT * η*η + 1)) *
        exp(-0.5 * (ν*ν*DT + 2*ν*α*z - α*z*α*z*η*η) / (DT*η*η+1))
    ϵ = ϵ / factor

    ta = DT / (α*α)

    N_large = ceil(1 / (π*sqrt(DT)))
    if π*ta*ϵ < 1
        N_large = max(N_large, ceil(sqrt(-2*log(π*ta*ϵ) / (π*π*ta))))
    end

    if 2*sqrt(2*π*ta)*ϵ < 1
        N_small = ceil(max(sqrt(ta)+1, 2+sqrt(-2*ta*log(2*ϵ*sqrt(2*π*ta))))) 
    else
        N_small = 2
    end

    if isinf(factor)
        out = 0
    elseif η == 0
        out = _g_minus_no_var(x, d)
    elseif N_small < N_large
        simple = _g_minus_small_time(x/(α*α), d, N_small)
        out = factor * simple
    else
        simple = _g_minus_large_time(x/(a*a), d, N_large)
        out = factor * simple
    end

    return out
end

function _integral_z_g_minus(x::Real, d::RatcliffDDM)
    """
    calculate the integral of integral_v_g_minus over the variable zr for a given set of input parameters. They handle variability in the parameter sz
    """
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    DT = x - τ #make into decision time

    out = 0.0
    
    if DT <= 0  # if DT <= 0
        out = 0
    elseif sz == 0  # this should be sz
        out = _integral_v_g_minus(x, d)
    else
        a = z - 0.5*sz  # zr - 0.5*szr; uniform variability
        b = z + 0.5*sz  # zr + 0.5*szr
        step_width = TUNE_INT_Z 
        out = _integrate(integral_v_g_minus, d, a, b, step_width) / sz
    end

    return out
end

function _integral_τ_g_minus(x::Real, d::RatcliffDDM)
    """
    calculate the integral of integral_z_g_minus over the variable τ for a given set of input parameters. They handle variability in the parameter st
    """
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    DT = x - τ #make into decision time

    out = 0.0
    if st == 0
        out = _integral_z_g_minus(x, d)  # should send t as RT
    else
        a = DT - 0.5*st  # DT - 0.5*st
        b = DT + 0.5*st  # DT + 0.5*st
        step_width = TUNE_INT_T0
        out = _integrate(integral_z_g_minus, d, a, b, step_width) / sts
    end

    return out
end

"""
    rand(dist::RatcliffDDM)

Generate a random choice and rt for the Ratcliff Diffusion Model

# Arguments
- `dist`: model object for Ratcliff Diffusion Model. 
- `method`: method simulating the diffusion process. 
    "rejection" uses Tuerlinckx et al., 2001 rejection-based method for the general wiener process
    "stochastic" uses the stochastic Euler method to directly simulate the stochastic differential equation

# References

    Tuerlinckx, F., Maris, E., Ratcliff, R., & De Boeck, P. (2001). 
    A comparison of four methods for simulating the diffusion process. 
    Behavior Research Methods, Instruments, & Computers, 33, 443-456.

    Converted from Rhddmjagsutils.R R script by Kianté Fernandez

    See also https://github.com/kiante-fernandez/Rhddmjags.
"""
function rand(rng::AbstractRNG, d::RatcliffDDM)
    # method::Char = "rejection"
    return _rand_rejection(rng, d)
    # method::Char = "stochastic"
#    return _rand_stochastic(rng, d)
end

function _rand_rejection(rng::AbstractRNG, d::RatcliffDDM; N::Int = 1)
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    if η == 0
        η = 1e-16
    end

    # Initialize output vectors
    result = zeros(N)
    T = zeros(N)
    XX = zeros(N)

    # Called sigma in 2001 paper
    D = σ^2 / 2

    # Program specifications
    ϵ = eps()  # precision from 1.0 to next double-precision number
    Δ = ϵ

    for n in 1:N
        r1 = randn()
        μ = ν + r1 * η
        bb = z - sz / 2 + sz * rand()
        zz = bb * α
        finish = 0
        totaltime = 0
        startpos = 0
        Aupper = α - zz
        Alower = -zz
        radius = min(abs(Aupper), abs(Alower))

        while finish == 0
            λ = 0.25 * μ^2 / D + 0.25 * D * π^2 / radius^2
            # eq. formula (13) in 2001 paper with D = sigma^2/2 and radius = Alpha/2
            F = D * π / (radius * μ)
            F = F^2 / (1 + F^2)
            # formula p447 in 2001 paper
            prob = exp(radius * μ / D)
            prob = prob / (1 + prob)
            dir_ = 2 * (rand() < prob) - 1
            l = -1
            s2 = 0
            s1 = 0
            while s2 > l
                s2 = rand()
                s1 = rand()
                tnew = 0
                told = 0
                uu = 0
                while abs(tnew - told) > ϵ || uu == 0
                    told = tnew
                    uu += 1
                    tnew = told + (2 * uu + 1) * (-1)^uu * s1^(F * (2 * uu + 1)^2)
                    # infinite sum in formula (16) in BRMIC,2001
                end
                l = 1 + s1^(-F) * tnew
            end
            # rest of formula (16)
            t = abs(log(s1)) / λ
            # is the negative of t* in (14) in BRMIC,2001
            totaltime += t
            dir_ = startpos + dir_ * radius
            ndt = τ - st / 2 + st * rand()
            if (dir_ + Δ) > Aupper
                T[n] = ndt + totaltime
                XX[n] = 1
                finish = 1
            elseif (dir_ - Δ) < Alower
                T[n] = ndt + totaltime
                XX[n] = 2
                finish = 1
            else
                startpos = dir_
                radius = minimum(abs.([Aupper, Alower] .- startpos))
            end
        end
    end
    return (choice=XX,rt=T)
end

function _rand_stochastic(rng::AbstractRNG, d::RatcliffDDM; N::Int = 1, nsteps::Int=300, step_length::Int=0.01)
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    if η == 0
        η = 1e-16
    end

    # Initialize output vectors
    choice = fill(0, N)
    rt = fill(0.0, N)

    for n in 1:N
        random_walk = Array{Float64}(undef, nsteps)
        start_point = (z - sz/2) + ((z + sz/2) - (z - sz/2)) * rand()
        ndt = (τ - st/2) + ((τ + st/2) - (τ - st/2)) * rand()
        drift = rand(Distributions.Normal(ν, η))
        random_walk[1] = start_point * α
        for s in 2:nsteps
            random_walk[s] = random_walk[s-1] + rand(Distributions.Normal(drift * step_length, σ * sqrt(step_length)))
            if random_walk[s] >= α
                random_walk[s:end] .= α
                rts[n] = s * step_length + ndt
                choice[n] = 1
                break
            elseif random_walk[s] <= 0
                random_walk[s:end] .= 0
                rts[n] = s * step_length + ndt
                choice[n] = 2
                break
            elseif s == nsteps
                rts[n] = NaN
                choice[n] = NaN
                break
            end
        end
    end   
    return  (choice=choice,rt=rts)
end

"""
    rand(dist::DDM, n_sim::Int)

Generate `n_sim` random choice-rt pairs for the Diffusion Decision Model.

# Arguments
- `dist`: model object for the Drift Diffusion Model.
- `n_sim::Int`: the number of simulated rts  
"""

function rand(rng::AbstractRNG, d::RatcliffDDM, n_sim::Int)
    return _rand_rejection(rng, d, N = n_sim)
end

sampler(rng::AbstractRNG, d::RatcliffDDM) = rand(rng::AbstractRNG, d::RatcliffDDM)
