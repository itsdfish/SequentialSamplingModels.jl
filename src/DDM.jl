"""
    DDM{T<:Real} <: SSM2D

    Model object for the Ratcliff (Full) Diffusion Decision Model.

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

    DDM(ν, α, τ, z, η, sz, st, σ)
    
    DDM(; ν = 1.00,
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
dist = DDM(ν = 1.0,α = 0.80,τ = 0.30,z = 0.25,η = 0.16,sz = 0.05,st = .10,σ = 1) 
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
````
    
# References
        
Ratcliff, R., & McKoon, G. (2008). The Diffusion Decision Model: Theory and Data for Two-Choice Decision Tasks. Neural Computation, 20(4), 873–922.
Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59–108. https://doi.org/10.1037/0033-295X.85.2.59
"""
mutable struct DDM{T<:Real} <: SSM2D
    ν::T
    α::T
    τ::T
    z::T
    η::T
    sz::T
    st::T
    σ::T
end

function DDM(ν, α, τ, z, η, sz, st, σ)
    return DDM(promote(ν, α, τ, z, η, sz, st, σ)...)
end

function params(d::DDM)
    (d.ν, d.α, d.τ, d.z,d.η, d.sz, d.st, d.σ)    
end

function DDM(; ν = 1.00,
    α = 0.80,
    τ = 0.30,
    z = 0.25,
    η = 0.16,
    sz = 0.05,
    st = 0.10,
    σ = 1.0)
    return DDM(ν, α, τ, z, η, sz, st, σ)
end

function pdf(d::DDM, choice, rt; ϵ::Real = 1.0e-12)
    if choice == 1
        (ν, α, τ, z, η, sz, st, σ) = params(d)
        return _pdf_Full(DDM(-ν, α, τ, 1-z, η, sz, st, σ), rt; ϵ)
    end
    return _pdf_Full(d, rt; ϵ)
end


"""
    _pdf_Full(d::DDM{T}, rt; ϵ::Real = 1.0e-12, n_st::Int=2, n_sz::Int=2) where {T<:Real}

Calculate the probability density function (PDF) for a Diffusion Decision Model (DDM) object. This 
function applies numerical integration to account for variability in non-decision time and bias, as suggested 
by Ratcliff and Tuerlinckx (2002).

# Arguments
- `d::DDM{T}`: a DDM distribution object
- `rt`: reaction time.

# Optional arguments
- `ϵ::Real`: a small constant to prevent divide by zero errors, default is 1.0e-12.
- `n_st::Int`: specifies the number of subintervals in the Simpson's rule for the integration associated with non-decision time variability. Default is 2.
- `n_sz::Int`: specifies the number of subintervals in the Simpson's rule for the integration associated with starting point variability. Default is 2. 

"""
function _pdf_Full(d::DDM{T}, rt; ϵ::Real = 1.0e-12, n_st::Int=2, n_sz::Int=2)  where {T<:Real}

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

"""
    _pdf_sv(d::DDM, rt; ϵ::Real = 1.0e-12)

Computes the Probability Density Function (PDF) for a given Diffusion Decision Model
with across-trial variability in drift-rate. This function uses analytic integration of the likelihood function 
for variability in drift-rate. 

# Arguments
- `d::DDM`: a DDM distribution constructor object
- `rt`: Reaction time for which the PDF is to be computed.

# Returns
- Returns the computed PDF value.

"""
function _pdf_sv(d::DDM, rt; ϵ::Real = 1.0e-12)
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    if η == 0
        return _pdf(DDM(ν, α, τ, z), rt; ϵ)
    end

    return _pdf(DDM(ν, α, τ, z), rt; ϵ)  + (  ( (α*z*η)^2 - 2*ν*α*z - (ν^2)*(rt-τ) ) / (2*(η^2)*(rt-τ)+2)  ) - log(sqrt((η^2)*(rt-τ)+1)) + ν*α*z + (ν^2)*(rt-τ)*0.5
end

"""
    _pdf(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T<:Real}

Computes the Probability Density Function (PDF) for a given Drift Diffusion Model (DDM). 
The function uses normalized time and applies an infinite sum algorithm. The implementation
is based on the work of Navarro & Fuss (2009) and Wabersich & Vandekerckhove (2014). 

# Arguments
- `d::DDM{T}`: A DDM object containing the parameters of the Diffusion Model.
- `t::Real`: Time for which the PDF is to be computed.

# Optional Arguments
- `ϵ::Real = 1.0e-12`: A very small number representing machine epsilon.

# Returns
- Returns the computed PDF value. 

# See also:
 - Converted from WienerDiffusionModel.jl repository orginally by Tobias Alfers: https://github.com/t-alfers/WienerDiffusionModel.jl    
"""

function _pdf(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T<:Real}
    (ν, α, τ, z) = params(d)
    if τ ≥ t
        return T(NaN)
    end
    u = (t - τ) / α^2 #use normalized time

    K_s = 2.0
    K_l = 1 / (π * sqrt(u))
    # number of terms needed for large-time expansion
    if (π*u*ϵ) < 1
        K_l = max(sqrt((-2*log(π*u*ϵ)) / (π^2 * u)), K_l)
    end
    # number of terms needed for small-time expansion
    if (2*sqrt(2*π*u)*ϵ) < 1
        K_s = max(2 + sqrt(-2u * log(2ϵ*sqrt(2*π*u))), sqrt(u)+1)
    end

    p = exp((-α*z*ν) - (0.5*(ν^2)*(t-τ))) / (α^2)

    # decision rule for infinite sum algorithm
    if K_s < K_l
        return p * _small_time_pdf(u, z, ceil(Int, K_s))
    end
    
    return p * _large_time_pdf(u, z, ceil(Int, K_l))
end

# small-time expansion
function _small_time_pdf(u::T, z::T, K::Int) where {T<:Real}
    inf_sum = zero(T)

    k_series = -floor(Int, 0.5*(K-1)):ceil(Int, 0.5*(K-1))
    for k in k_series
        inf_sum += ((2k + z) * exp(-((2k + z)^2 / (2u))))
    end

    return inf_sum / sqrt(2π*u^3)
end

# large-time expansion
function _large_time_pdf(u::T, z::T, K::Int) where {T<:Real}
    inf_sum = zero(T)

    for k in 1:K 
        inf_sum += (k * exp(-0.5*(k^2*π^2*u)) * sin(k*π*z)) 
    end

    return π * inf_sum
end

"""
    _simpson_1D(x::Real, ν::Real, η::Real, α::Real, z::Real, τ::Real, ϵ::Real, lb_z::Real, ub_z::Real, n_sz::Int, lb_t::Real, ub_t::Real, n_st::Int)

Calculate the 1-dimensional Simpson's numerical integration for a drift diffusion model with given parameters. This function is used for integrating over either the starting point or the non-decision time.

# Arguments
- `x::Real`: Reaction time for which the probability density function is being computed.
- `ν::Real`, `η::Real`, `α::Real`, `z::Real`, `τ::Real`: Parameters of the Ratcliff Drift Diffusion Model.
- `ϵ::Real`: A small constant to prevent divide by zero errors.
- `lb_z::Real`, `ub_z::Real`: Lower and upper bounds for z (starting point).
- `n_sz::Int`: Specifies the number of subintervals for Simpson's rule in the z dimension.
- `lb_t::Real`, `ub_t::Real`: Lower and upper bounds for t (non-decision time).
- `n_st::Int`: Specifies the number of subintervals for Simpson's rule in the t dimension.

# Returns
- Returns the Simpson's numerical integration of the PDF of the Ratcliff Drift Diffusion Model at the given reaction time `x`, over the specified bounds and subintervals. 

# Note
- If `n_st` is 0, the function integrates over z (starting point). If `n_sz` is 0, the function integrates over t (non-decision time).

# References

https://en.wikipedia.org/wiki/Simpson%27s_rule 

"""
# Simpson's Method one dimentional case
function _simpson_1D(x::Real, ν::Real, η::Real, α::Real, z::Real, τ::Real, ϵ::Real, lb_z::Real, ub_z::Real, n_sz::Int, lb_t::Real, ub_t::Real, n_st::Int)
   
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

    S =  _pdf_sv(DDM(ν, α, lb_t, lb_z, η, 0, 0, 1), x; ϵ)
    
    y = 0 
    z_tag = 0 
    t_tag = 0 

    for i in 1:n
        z_tag = lb_z + hz * i
        t_tag = lb_t + ht * i
       
        y = _pdf_sv(DDM(ν, α, t_tag, z_tag, η, 0, 0, 1), x; ϵ)
           
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

"""
    _simpson_2D(x::Real, ν::Real, η::Real, α::Real, z::Real, τ::Real, ϵ::Real, lb_z::Real, ub_z::Real, n_sz::Int, lb_t::Real, ub_t::Real, n_st::Int)

Calculate the 2-dimensional Simpson's numerical integration for a drift diffusion model with given parameters. This function is used for integrating over both the starting point and the non-decision time.

# Arguments
- `x::Real`: Reaction time for which the probability density function is being computed.
- `ν::Real`, `η::Real`, `α::Real`, `z::Real`, `τ::Real`: Parameters of the Ratcliff Drift Diffusion Model.
- `ϵ::Real`: A small constant to prevent divide by zero errors.
- `lb_z::Real`, `ub_z::Real`: Lower and upper bounds for z (starting point).
- `n_sz::Int`: Specifies the number of subintervals for Simpson's rule in the z dimension.
- `lb_t::Real`, `ub_t::Real`: Lower and upper bounds for t (non-decision time).
- `n_st::Int`: Specifies the number of subintervals for Simpson's rule in the t dimension.

# Returns
- Returns the Simpson's numerical integration of the PDF of the Ratcliff Drift Diffusion Model at the given reaction time `x`, over the specified bounds and subintervals in both dimensions.

# Note
- The function calls `_simpson_1D` to perform the 1-dimensional integrations over each dimension.

"""
# Simpson's Method two dimentional case
function _simpson_2D(x::Real, ν::Real, η::Real, α::Real, z::Real, τ::Real, ϵ::Real, lb_z::Real, ub_z::Real, n_sz::Int, lb_t::Real, ub_t::Real, n_st::Int)

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

logpdf(d::DDM, choice, rt; ϵ::Real = 1.0e-12) = log(pdf(d, choice, rt; ϵ))

# function logpdf(d::DDM, data::T) where {T<:NamedTuple}
#     return sum(logpdf.(d, data...))
# end

# function logpdf(dist::DDM, data::Array{<:Tuple,1})
#     LL = 0.0
#     for d in data
#         LL += logpdf(dist, d...)
#     end
#     return LL
# end

# logpdf(d::DDM, data::Tuple) = logpdf(d, data...)

"""
    cdf(d::DDM, choice, rt; ϵ::Real = 1e-7)

Compute the Cumulative Distribution Function (CDF) for the Ratcliff Diffusion model. This function uses 6 Gaussian quadrature for numerical integration.

# Arguments
- `d`: an instance of DDM Constructor
- `choice`: an input representing the choice.
- `rt`: response time.
- `ϵ`: a small constant to avoid division by zero, defaults to 1e-7.

# Returns
- `y`: an array representing the CDF of the Ratcliff Diffusion model.

# Reference
Tuerlinckx, F. (2004). The efficient computation of the cumulative distribution and probability density functions in the diffusion model, Behavior Research Methods, Instruments, & Computers, 36 (4), 702-716.

# See also
- Converted from cdfdif.c C script by Joachim Vandekerckhove: https://ppw.kuleuven.be/okp/software/dmat/
"""
function cdf(d::DDM, choice, rt; ϵ::Real = 1e-7)

    (ν, α, τ, z, η, sz, st, σ) = params(d)

    size = length(rt)
    y = zeros(Float64, size)
    epsi = 1e-10

    #transform parameters
    α = α/10.
    τ = τ
    η = η/10. + epsi
    z = z*(α/10.)
    sz = sz*(α/10.) + epsi
    st = st + epsi
    ν = ν/10.

    p_boundary = 0.0

    for i in 1:size
        y[i] = _cdf(d,rt[i],choice[i],p_boundary; ϵ)     
        y[i] = (1 - p_boundary) + rt[i]*y[i]
    end

    return y
end
"""
    _cdf(d::DDM{T}, choice, rt, prob; ϵ::Real = 1e-7) where {T<:Real}

A helper function to compute the Cumulative Distribution Function (CDF) for the Full Diffusion model.

# Arguments
- `d`: an instance of DDM Constructor
- `choice`: an input representing the choice.
- `rt`: response time.
- `prob`: a probability value.
- `ϵ`: a small constant to avoid division by zero, defaults to 1e-7.

# Returns
- `Fnew`: the computed CDF for the given parameters.

"""
function _cdf(d::DDM, choice, rt, prob; ϵ::Real = 1e-7)
    
    (ν, α, τ, z, η, sz, st, σ) = params(d)
    #Explcit recode of the choice from 2(lower) & 1(upper) to 0(lower) and 1(upper)
    #note we need to make sure this is consistent in the all the relative bound models
    if choice == 2 #lower
        choice = 0
    elseif choice == 1 #upper
        choice = 1
    end

    # Initializing variables
    a2 = α*α
    Z_U = (1-choice)*z+choice*(α-z)+sz/2
    Z_L = (1-choice)*z+choice*(α-z)-sz/2
    lower_t = τ-st/2
    upper_t = 0.0
    Δ = 1e-29
    min_rt=0.001
    v_max = 5000 # maximum number of terms in a partial sum approximating infinite series

    Fnew = 0.0
    sum_z=0.0
    sum_ν=0.0
    p1 = 0.0
    p0 = 0.0
    sum_hist = zeros(3)
    denom = 0.0
    sifa = 0.0
    upp = 0.0
    low = 0.0
    fact = 0.0
    exdif = 0.0
    su = 0.0
    sl = 0.0
    zzz = 0.0
    ser = 0.0
    nr_ν = 6
    nr_z = 6

    # Defining Gauss-Hermite abscissae and weights for numerical integration
    gk = [-2.3506049736744922818,-1.3358490740136970132,-.43607741192761650950,.43607741192761650950,1.3358490740136970132,2.3506049736744922818]
    w_gh = [.45300099055088421593e-2,.15706732032114842368,.72462959522439207571,.72462959522439207571,.15706732032114842368,.45300099055088421593e-2]
    gz = [-.93246951420315193904,-.66120938646626381541,-.23861918608319693247,.23861918608319712676,.66120938646626459256,.93246951420315160597]
    w_g = [.17132449237917049545,.36076157304813916138,.46791393457269092604,.46791393457269092604,.36076157304813843973,.17132449237917132812]

    # Adjusting Gauss-Hermite abscissae and weights
    for i=1:nr_ν
        gk[i] = 1.41421356237309505*gk[i]*η+ν
        w_gh[i] = w_gh[i]/1.772453850905515882
    end
    for i=1:nr_z
        gz[i] = (.5*sz*gz[i])+z
    end

    #   numerical integration
    for i=1:nr_z
        sum_ν=0.0
            #   numerical integration 
        for m=1:nr_ν
            if abs(gk[m])>ϵ
                sum_ν+=(exp(-200*gz[i]*gk[m])-1)/(exp(-200*α*gk[m])-1)*w_gh[m]
            else
                sum_ν+=gz[i]/α*w_gh[m]
            end
        end
        sum_z+=sum_ν*w_g[i]/2
    end
    prob = sum_z

    if (rt-τ+st/2 > min_rt) # is t larger than lower boundary τ distribution?
        upper_t = min(rt, τ+st/2)
        p1 = prob*(upper_t-lower_t)/st # integrate probability with respect to t
        p0 = (1-prob)*(upper_t-lower_t)/st
        if rt > τ+st/2 # is t larger than upper boundary Ter distribution?
            sum_hist = zeros(3)
            for v in 1:v_max # infinite series
                sum_hist = circshift(sum_hist, 1)
                sum_ν = 0
                sifa = π*v/α
                for m in 1:nr_ν # numerical integration with respect to xi
                    denom = (100*gk[m]*gk[m] + (π*π)*(v*v)/(100*a2))
                    upp = exp((2*choice-1)*Z_U*gk[m]*100 - 3*log(denom) + log(w_gh[m]) - 2*log(100))
                    low = exp((2*choice-1)*Z_L*gk[m]*100 - 3*log(denom) + log(w_gh[m]) - 2*log(100))
                    fact = upp*((2*choice-1)*gk[m]*sin(sifa*Z_U)*100 - sifa*cos(sifa*Z_U)) - 
                           low*((2*choice-1)*gk[m]*sin(sifa*Z_L)*100 - sifa*cos(sifa*Z_L))
                    exdif = exp((-.5*denom*(rt-upper_t)) + log(1-exp(-.5*denom*(upper_t-lower_t))))
                    sum_ν += fact*exdif
                end
                sum_hist[3] = sum_hist[2] + v*sum_ν
                if abs(sum_hist[1] - sum_hist[2]) < Δ && abs(sum_hist[2] - sum_hist[3]) < Δ && sum_hist[3] > 0
                    break
                end
            end

            Fnew = (p0*(1-choice) + p1*choice) - sum_hist[3]*4*π/(a2*sz*st)
            # cumulative distribution function for t and x
        elseif t <= τ+st/2 # is t lower than upper boundary Ter distribution?
            sum_ν = 0
            for m in 1:nr_ν
                if abs(gk[m]) > ϵ
                    sum_z = 0
                    for i in 1:nr_z
                        zzz = (α - gz[i])*choice + gz[i]*(1 - choice)
                        ser = -((α*a2)/((1 - 2*choice)*gk[m]*π*.01))*sinh(zzz*(1 - 2*x)*gk[m]/.01)/
                              (sinh((1 - 2*choice)*gk[m]*α/.01)^2) +
                              (zzz*a2)/((1 - 2*choice)*gk[m]*π*.01)*cosh((α - zzz)*(1 - 2*choice)*gk[m]/.01)/
                              sinh((1 - 2*choice)*gk[m]*α/.01)
                        sum_hist = zeros(3)
                        for v in 1:v_max
                            sum_hist = circshift(sum_hist, 1)
                            sifa = π*v/α
                            denom = (gk[m]*gk[m]*100 + (π*v)*(π*v)/(a2*100))
                            sum_hist[3] = sum_hist[2] + v*sin(sifa*zzz)*exp(-.5*denom*(rt - lower_t) - 2*log(denom))
                            if abs(sum_hist[1] - sum_hist[2]) < Δ && abs(sum_hist[2] - sum_hist[3]) < Δ && sum_hist[3] > 0
                                break
                            end
                        end
                        sum_z += .5*w_g[i]*(ser - 4*sum_hist[3])*(π/100)/(a2*st)*exp((2*choice - 1)*zzz*gk[m]*100)
                    end
                else
                    sum_hist = zeros(3)
                    su = -(Z_U*Z_U)/(12*a2) + (Z_U*Z_U*Z_U)/(12*α*a2) - (Z_U*Z_U*Z_U*Z_U)/(48*a2*a2)
                    sl = -(Z_L*Z_L)/(12*a2) + (Z_L*Z_L*Z_L)/(12*α*a2) - (Z_L*Z_L*Z_L*Z_L)/(48*a2*a2)
                    for v in 1:v_max
                        sum_hist = circshift(sum_hist, 1)
                        sifa = π*v/α
                        denom = (π*v)*(π*v)/(a2*100)
                        sum_hist[3] = sum_hist[2] + 1/(π*π*π*π*v*v*v*v)*(cos(sifa*Z_L) - cos(sifa*Z_U))*
                                      exp(-.5*denom*(rt - lower_t))
                        if abs(sum_hist[1] - sum_hist[2]) < Δ && abs(sum_hist[2] - sum_hist[3]) < Δ && sum_hist[3] > 0
                            break
                        end
                    end
                    sum_z = 400*a2*α*(sl - su - sum_hist[3])/(st*sz)
                end
                sum_ν += sum_z*w_gh[m]
            end
            Fnew = (p0*(1 - choice) + p1*choice) - sum_ν
        end
    elseif rt - τ + st/2 <= min_rt # is t lower than lower boundary Ter distr?
        Fnew = 0
    end
    
    Fnew = Fnew > Δ ? Fnew : 0

    return    Fnew

end

"""
    rand(dist::DDM)

Generate a random choice and rt for the Ratcliff Diffusion Model

# Arguments
- `dist`: model object for Ratcliff Diffusion Model. 
    
method simulating the diffusion process: 
_rand_rejection uses Tuerlinckx et al., 2001 rejection-based method for the general wiener process

# References

    Tuerlinckx, F., Maris, E., Ratcliff, R., & De Boeck, P. (2001). 
    A comparison of four methods for simulating the diffusion process. 
    Behavior Research Methods, Instruments, & Computers, 33, 443-456.

    Converted from Rhddmjagsutils.R R script by Kianté Fernandez
    See also https://github.com/kiante-fernandez/Rhddmjags.
"""
function rand(rng::AbstractRNG, d::DDM)
    return _rand_rejection(rng, d)
end

function _rand_rejection(rng::AbstractRNG, d::DDM; N::Int = 1)
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    if η == 0
        η = 1e-16
    end

    # Initialize output vectors
    choice = fill(0, N)
    rt = fill(0.0, N)

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
                rt[n] = ndt + totaltime
                choice[n] = 1
                finish = 1
            elseif (dir_ - Δ) < Alower
                rt[n] = ndt + totaltime
                choice[n] = 2
                finish = 1
            else
                startpos = dir_
                radius = minimum(abs.([Aupper, Alower] .- startpos))
            end
        end
    end
    return (choice=choice,rt=rt)
end

"""
    rand(dist::DDM, n_sim::Int)

Generate `n_sim` random choice-rt pairs for the Ratcliff Diffusion Decision Model.

# Arguments
- `dist`: model object for the Ratcliff DDM.
- `n_sim::Int`: the number of simulated rts  
"""

function rand(rng::AbstractRNG, d::DDM, n_sim::Int)
    return _rand_rejection(rng, d, N = n_sim)
end

"""
    n_options(dist::DDM)

Returns 2 for the number of choice options

# Arguments

- `d::DDM`: a model object for the drift diffusion model
"""
n_options(d::DDM) = 2


#######
# old cdf code for debug checking
#########################################
# Cumulative density function           #
# Blurton, Kesselmeier, & Gondan (2012) #
#########################################
 
# function cdf(d::DDM, choice, rt; ϵ::Real = 1.0e-12)
#     if choice == 1
#         (ν, α, τ, z) = params(d)
#         return cdf(DDM(-ν, α, τ, 1-z), rt; ϵ)
#     end

#     return cdf(d, rt; ϵ)
# end

# # cumulative density function over the lower boundary
# function cdf(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T<:Real}
#     if d.τ ≥ t
#         return T(NaN)
#     end

#     K_l = _K_large(d, t; ϵ)
#     K_s = _K_small(d, t; ϵ)

#     if K_l < 10*K_s
#         return _Fl_lower(d, K_l, t)
#     end
#     return _Fs_lower(d, K_s, t)
# end

# # Large time representation of lower subdistribution
# function _Fl_lower(d::DDM{T}, K::Int, t::Real) where {T<:Real}
#     (ν, α, τ, z) = params(d)
#     F = zero(T)
#     K_series = K:-1:1
#     for k in K_series
#         F -= (k/(ν^2 + k^2*π^2/(α^2)) * 
#             exp(-ν*α*z - 0.5*ν^2*(t-τ) - 0.5*k^2*π^2/(α^2)*(t-τ)) *
#             sin(π * k * z))
#     end
#     return _P_upper(ν, α, z) + 2*π/(α^2) * F
# end

# # Small time representation of the upper subdistribution
# function _Fs_lower(d::DDM{T}, K::Int, t::Real) where {T<:Real}
#     (ν, α, τ, z) = params(d)
#     if abs(ν) < sqrt(eps(T))
#         return _Fs0_lower(d, K, t)
#     end

#     sqt = sqrt(t-τ)

#     S1 = zero(T)
#     S2 = zero(T)
#     K_series = K:-1:1

#     for k in K_series
#         S1 += (_exp_pnorm(2*ν*α*k, -sign(ν)*(2*α*k+α*z+ν*(t-τ))/sqt) -
#             _exp_pnorm(-2*ν*α*k-2*ν*α*z, sign(ν)*(2*α*k+α*z-ν*(t-τ))/sqt))

#         S2 += (_exp_pnorm(-2*ν*α*k, sign(ν)*(2*α*k-α*z-ν*(t-τ))/sqt) - 
#             _exp_pnorm(2*ν*α*k-2*ν*α*z, -sign(ν)*(2*α*k-α*z+ν*(t-τ))/sqt))
#     end

#     return _P_upper(ν, α, z) + sign(ν) * ((cdf(Normal(), -sign(ν) * (α*z+ν*(t-τ))/sqt) -
#              _exp_pnorm(-2*ν*α*z, sign(ν) * (α*z-ν*(t-τ)) / sqt)) + S1 + S2)
# end

# # Zero drift version
# function _Fs0_lower(d::DDM{T}, K::Int, t::Real) where {T<:Real}
#     (_, α, τ, z) = params(d)
#     F = zero(T)
#     K_series = K:-1:0
#     for k in K_series
#         F -= (cdf(Distributions.Normal(), (-2*k - 2+z) * α / sqrt(t-τ)) + cdf(Distributions.Normal(), (-2*k -z) * α / sqrt(t-τ)))
#     end
#     return 2*F
# end
# # Number of terms required for large time representation
# function _K_large(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T<:Real}
#     (ν, α, τ, z) = params(d)
#     x = t-τ
#     sqrtL1 = sqrt(1/x) * α/π
#     sqrtL2 = sqrt(max(1, -2/x*α*α/π/π * (log(ϵ*π*x/2 * (ν*ν + π*π/α/α)) + ν*α*z + ν*ν*x/2)))
#     return ceil(Int, max(sqrtL1, sqrtL2))
# end

# # Number of terms required for small time representation
# function _K_small(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T<:Real}
#     (ν, α, τ, z) = params(d)
#     if abs(ν) < sqrt(eps(T))
#         return ceil(Int, max(0, z/2 - sqrt(t-τ)/(2*α) * quantile(Normal(), max(0, min(1, ϵ/(2-2*z))))))
#     end
#     if ν > 0
#         return _K_small(DDM(-ν, α, τ, z), t; ϵ = exp(-2*α*z*ν)*ϵ)
#     end
#     S2 = z - 1 + 1/(2*ν*α) * log(ϵ/2 * (1-exp(2*ν*α)))
#     S3 = (0.535 * sqrt(2*(t-τ)) + ν*(t-τ) + α*z)/(2*α)
#     S4 = z/2 - sqrt(t-τ)/(2*α) * quantile(Normal(), max(0, min(1, ϵ*α/(0.3 * sqrt(2*π*(t-τ))) * exp(ν^2*(t-τ)/2 + ν*α*z) ))) 
#     return ceil(Int, max(0, S2, S3, S4))
# end
# # Probability for absorption at upper barrier
# function _P_upper(ν::T, α::T, z::T) where {T<:Real}
#     e = exp(-2 * ν * α * (1-z))
#     if isinf(e)
#         return 1.0
#     end
#     if abs(e-1) < sqrt(eps(T))
#         return 1-z
#     end
#     return (1-e)/(exp(2*ν*α*z) - e)
# end

# # Calculates exp(a) * pnorm(b) using an approximation by Kiani et al. (2008)
# function _exp_pnorm(a::T, b::T) where {T<:Real}
#     r = exp(a) * cdf(Distributions.Normal(), b)
#     if isnan(r) && b < -5.5
#         r = (1/sqrt(2)) * exp(a - b^2/2) * (0.5641882/(b^3) - 1/(b * sqrt(π))) 
#     end
#     return r
# end