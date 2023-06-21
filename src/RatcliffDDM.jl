"""
    RatcliffDDM 

    Model object for the Ratcliff Diffusion Model.

# Fields
    - `ν`: drift rate. Average slope of the information accumulation process. The drift gives information about the speed and direction of the accumulation of information. Typical range: -5 < ν < 5
    - `α`: boundary threshold separation. The amount of information that is considered for a decision. Typical range: 0.5 < α < 2
    - `τ`: non-decision time. The duration for a non-decisional processes (encoding and response execution). Typical range: 0.1 < τ < 0.5 
    - `z`: starting point. Indicator of an an initial bias towards a decision. The z parameter is relative to a (i.e. it ranges from 0 to 1).
    - `η`:  across-trial-variability of drift rate. Typical range: 0 < η < 2. Default is 0.
    - `sz`: across-trial-variability of starting point. Typical range: 0 < sz < 0.5. Default is 0.
    - `st`: across-trial-variability of non-decision time. Typical range: 0 < st < 0.2. Default is 0.
    - `σ`: diffusion noise constant. Default is 1.

# Example 

````julia
using SequentialSamplingModels
dist = RatcliffDDM(ν = 0.50,α = 0.08,τ = 0.30,z = 0.04,η = 0.10,sz = 0.02,st = .02,σ = 0.10) 
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
````
    
# References
        
Ratcliff, R., & McKoon, G. (2008). The Diffusion Decision Model: Theory and Data for Two-Choice Decision Tasks. Neural Computation, 20(4), 873–922.
Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59–108. https://doi.org/10.1037/0033-295X.85.2.59
"""
mutable struct RatcliffDDM{T1,T2,T3,T4,T5,T6,T7,T8} <: SequentialSamplingModel
    ν::T1
    α::T2
    τ::T3
    z::T4
    η::T5
    sz::T6
    st::T7
    σ::T8
end

Base.broadcastable(x::RatcliffDDM) = Ref(x)

function params(d::RatcliffDDM)
    (d.ν, d.α, d.τ, d.z,d.η, d.sz, d.st, d.σ)    
end

loglikelihood(d::RatcliffDDM, data) = sum(logpdf.(d, data...))

"""
RatcliffDDM(; ν = 1.00,
    α = 0.80,
    τ = 0.30,
    z = 0.25,
    η = 0.16,
    sz = 0.05,
    st = 0.10,
    σ = 1.0
    )

Constructor for the Ratcliff Diffusion Model. 
    
# Keywords 
- `ν`: drift rate. Average slope of the information accumulation process. The drift gives information about the speed and direction of the accumulation of information. Typical range: -5 < ν < 5
- `α`: boundary threshold separation. The amount of information that is considered for a decision. Typical range: 0.5 < α < 2
- `τ`: non-decision time. The duration for a non-decisional processes (encoding and response execution). Typical range: 0.1 < τ < 0.5 
- `z`: starting point. Indicator of an an initial bias towards a decision. The z parameter is relative to a (i.e. it ranges from 0 to 1).
- `η`:  across-trial-variability of drift rate. Typical range: 0 < η < 2. Default is 0.
- `sz`: across-trial-variability of starting point. Typical range: 0 < sz < 0.5. Default is 0.
- `st`: across-trial-variability of non-decision time. Typical range: 0 < st < 0.2. Default is 0.
- `σ`: diffusion noise constant. Default is 1.

# Example 

```julia
using SequentialSamplingModels
dist = RatcliffDDM(;ν = 1.00,α = 0.80,τ = 0.30,z = 0.25,η = 0.16,sz = 0.05,st = 0.10,σ = 1.0) 
```
"""
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

function pdf_sv(d::RatcliffDDM{T}, rt::Real; ϵ::Real = 1.0e-12) where {T<:Real}
    (ν, α, τ, z, η, sz, st, σ) = params(d)

    if η == 0
        return pdf(SequentialSamplingModels.DDM(ν, α, τ, z), rt; ϵ)
    end
    # if isless(ν,0)
    #     return pdf(SequentialSamplingModels.DDM(ν, α, τ, z), t; ϵ)  + (  ( (α*z*η)^2 - 2*ν*α*z - (ν^2)*t ) / (2*(η^2)*t+2)  ) - log(sqrt((η^2)*t+1)) + ν*α*z + (ν^2)*t*0.5
    # end
    # return pdf(SequentialSamplingModels.DDM(ν, α, τ, z), t; ϵ)  + (  ( (α*(1-z)*η)^2 + 2*ν*α*(1-z) - (ν^2)*t ) / (2*(η^2)*t+2)  ) - log(sqrt((η^2)*t+1)) - ν*α*(1-z) + (ν^2)*t*0.5
    return pdf(SequentialSamplingModels.DDM(ν, α, τ, z), rt; ϵ)  + (  ( (α*z*η)^2 - 2*ν*α*z - (ν^2)*(rt-τ) ) / (2*(η^2)*(rt-τ)+2)  ) - log(sqrt((η^2)*(rt-τ)+1)) + ν*α*z + (ν^2)*(rt-τ)*0.5
end

function pdf(d::RatcliffDDM, choice, rt; ϵ::Real = 1.0e-12)
    if choice == 1
        (ν, α, τ, z, η, sz, st, σ) = params(d)
        return pdf(RatcliffDDM(-ν, α, τ, 1-z, η, sz, st, σ), rt; ϵ)
    end
    return pdf(d, rt; ϵ)
end

#use numerical integration for variability in non-decision time and bias (Ratcliff and Tuerlinckx, 2002)
function pdf(d::RatcliffDDM{T}, rt; ϵ::Real = 1.0e-12, n_st::Int=2, n_sz::Int=2)  where {T<:Real}
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
            return pdf_sv(d, rt; ϵ)
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

    S =  pdf_sv(RatcliffDDM(ν, α, lb_t, lb_z, η, 0, 0, 1), x; ϵ)
    
    y = 0 
    z_tag = 0 
    t_tag = 0 

    for i in 1:n
        z_tag = lb_z + hz * i
        t_tag = lb_t + ht * i
       
        y = pdf_sv(RatcliffDDM(ν, α, t_tag, z_tag, η, 0, 0, 1), x; ϵ)
           
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

# """
# cdf_full()

#  The orignial algorithm was written on 09/01/06 by Joachim Vandekerckhove
#  Then converted from c to julia by Kianté Fernandez
 
#  Computes Cumulative Distribution Function for the Diffusion model with random trial to trial mean drift (normal), 
#  starting point and non-decision (ter) time (both rectangular). Uses 6 quadrature points for drift and 6 for the others.

#  Based on methods described in:
#     Tuerlinckx, F. (2004). The efficient computation of the
#     cumulative distribution and probability density functions
#     in the diffusion model, Behavior Research Methods,
#     Instruments, & Computers, 36 (4), 702-716.

# """