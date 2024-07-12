"""
    DDM{T<:Real} <: SSM2D

Model object for the standard Drift Diffusion Model.

# Parameters

- `ν`: drift rate. Average slope of the information accumulation process. The drift gives information about the speed and direction of the accumulation of information. Typical range: -5 < ν < 5
- `α`: boundary threshold separation. The amount of information that is considered for a decision. Typical range: 0.5 < α < 2
- `z`: starting point. Indicator of an an initial bias towards a decision. The z parameter is relative to a (i.e. it ranges from 0 to 1).
- `τ`: non-decision time. The duration for a non-decisional processes (encoding and response execution). Typical range: 0.1 < τ < 0.5 
- `η`:  across-trial-variability of drift rate. Typical range: 0 < η < 2. Default is 0.
- `sz`: across-trial-variability of starting point. Typical range: 0 < sz < 0.5. Default is 0.
- `st`: across-trial-variability of non-decision time. Typical range: 0 < st < 0.2. Default is 0.
- `σ`: diffusion noise constant. Default is 1.

# Constructors 

Two constructors are defined below. The first constructor uses positional arguments, and is therefore order dependent:

    DDM(ν, α, z, τ, η, sz, st, σ)

The second constructor uses keywords with default values, and is not order dependent: 

    DDM(; ν = 1.00, α = 0.80, τ = 0.30, z = 0.50, η = 0.16, sz = 0.05, st = 0.10, σ = 1.0)

# Example 

```julia
using SequentialSamplingModels
dist = DDM(ν = 1.00, α = 0.80, τ = 0.30, z = 0.50, η = 0.16, sz = 0.05, st = 0.10, σ = 1.0)
choice,rt = rand(dist, 10)
like = pdf.(dist, choice, rt)
loglike = logpdf.(dist, choice, rt)
```

# References
    
Ratcliff, R., & McKoon, G. (2008). The Diffusion Decision Model: Theory and Data for Two-Choice Decision Tasks. Neural Computation, 20(4), 873–922.
Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59–108.
"""
mutable struct DDM{T <: Real} <: AbstractDDM
    ν::T
    α::T
    z::T
    τ::T
    η::T
    sz::T
    st::T
    σ::T
end

function DDM(ν, α, z, τ, η, sz, st, σ)
    return DDM(promote(ν, α, z, τ, η, sz, st, σ)...)
end

function params(d::DDM)
    (d.ν, d.α, d.z, d.τ, d.η, d.sz, d.st, d.σ)    
end

function DDM(; ν = 1.00, α = 0.80, τ = 0.30, z = 0.50, η = 0.16, sz = 0.05, st = 0.10, σ = 1.0)
    return DDM(ν, α, z, τ, η, sz, st, σ)
end

function pdf(d::DDM, choice, rt; ϵ::Real = 1.0e-12)
    if choice == 1
        (ν, α, z, τ, η, sz, st, σ) = params(d)
            # Transform ν, z if rt is upper bound response
        return _pdf_Full(DDM(-ν, α, 1 - z, τ,  η, sz, st, σ), rt; ϵ)
    end
    return _pdf_Full(d, rt; ϵ)
end

"""
    _pdf_Full(d::DDM{T}, rt; ϵ::Real = 1.0e-12, simps_err::Real = 1e-6) where {T<:Real}

Calculate the probability density function (PDF) for a Diffusion Decision Model (DDM) object. This 
function applies numerical integration to account for variability in non-decision time and bias, as suggested 
by Ratcliff and Tuerlinckx (2002).

# Arguments
- `d::DDM{T}`: a DDM distribution object
- `rt`: reaction time.

# Optional arguments
- `ϵ::Real`: a small constant to prevent divide by zero errors, default is 1.0e-12.
- `simps_err::Real = 1e-3`: Error tolerance for the numerical integration.

"""
function _pdf_Full(d::DDM{T}, rt; ϵ::Real = 1.0e-12, simps_err::Real = 1e-6) where {T<:Real}
    (ν, α, z, τ, η, sz, st, σ) = params(d)

    # Check if parameters are valid
    if (z < 0 || z > 1 || α <= 0 || τ < 0 || st < 0 || η < 0 || sz < 0 || sz > 1 || 
        (rt - (τ - st/2)) < 0 || (z + sz/2) > 1 || (z - sz/2) < 0 || (τ - st/2) < 0)
        return zero(T)
    end

    rt = abs(rt)

    if st < 1e-6
        st = 0
    end
    if sz < 1e-6
        sz = 0
    end

    if sz == 0
        if st == 0  # η=0, sz=0, st=0
            return _pdf_sv(DDM(ν, α, z, τ, η, 0, 0, σ), rt - τ; ϵ)
        else  # η=0, sz=0, st≠0
            f = y -> _pdf_sv(DDM(ν, α, z, y[1], η, 0, 0, σ), rt - y[1]; ϵ)
            result, _ = hcubature(f, (τ-st/2,), (τ+st/2,); rtol=simps_err, atol=simps_err)
            return result / st  # Normalize by the integration range
        end
    else  # sz≠0
        if st == 0  # η=0, sz≠0, st=0
            f = y -> _pdf_sv(DDM(ν, α, y[1], τ, η, 0, 0, σ), rt - τ; ϵ)
            result, _ = hcubature(f, (z-sz/2,), (z+sz/2,); rtol=simps_err, atol=simps_err)
            return result / sz  
        else  # η=0, sz≠0, st≠0
            f = y -> _pdf_sv(DDM(ν, α, y[1], y[2], η, 0, 0, σ), rt - y[2]; ϵ)
            result, _ = hcubature(f, (z-sz/2, τ-st/2), (z+sz/2, τ+st/2); rtol=simps_err, atol=simps_err)
            return result / (sz * st)  
        end
    end
end

"""
    _pdf_sv(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T <: Real}

Computes the Probability Density Function (PDF) for a given Diffusion Decision Model
with across-trial variability in drift-rate. This function uses analytic integration of the likelihood function 
for variability in drift-rate. 

# Arguments
- `d::DDM`: a DDM distribution constructor object
- `rt`: Reaction time for which the PDF is to be computed.

# Returns
- Returns the computed PDF value.

"""
function _pdf_sv(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T <: Real}
    (ν, α, z, τ, η, σ) = params(d)

    if t <= 0
        return zero(T)
    end

    if η == 0
        return _pdf(DDM(ν, α, z, τ, 0, 0, 0, σ), t; ϵ)
    end

    XX = t / (α^2) #use normalized time
    p = _pdf(DDM(ν, α, z, τ, 0, 0, 0, σ), XX; ϵ) #get f(t|0,1,w)
    # convert to f(t|v,a,w)
    return exp(log(p) + ((α*z*η)^2 - 2*α*ν*z - (ν^2)*XX)/(2*(η^2)*XX+2)) / sqrt((η^2)*XX+1) / (α^2)

end

################################################################################
#  Converted from WienerDiffusionModel.jl repository orginally by Tobias Alfers#
#  See https://github.com/t-alfers/WienerDiffusionModel.jl                     #
################################################################################

#####################################
# Probability density function      #
# Navarro & Fuss (2009)             #
# Wabersich & Vandekerckhove (2014) #
#####################################

"""
    _pdf(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T <: Real}

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
function _pdf(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T <: Real}
    (ν, α, z) = params(d)

    if t <= 0
        return zero(T)
    end
    u = t

    K_s = 2.0
    K_l = 1 / (π * sqrt(u))
    # number of terms needed for large-time expansion
    if (π * u * ϵ) < 1
        K_l = max(sqrt((-2 * log(π * u * ϵ)) / (π^2 * u)), K_l)
    end
    # number of terms needed for small-time expansion
    if (2 * sqrt(2 * π * u) * ϵ) < 1
        K_s = max(2 + sqrt(-2u * log(2ϵ * sqrt(2 * π * u))), sqrt(u) + 1)
    end

    p = exp((-α * z * ν) - (0.5 * (ν^2) * (t))) / (α^2)

    # decision rule for infinite sum algorithm
    if K_s < K_l
        return p * _small_time_pdf(u, z, ceil(Int, K_s))
    end

    return p * _large_time_pdf(u, z, ceil(Int, K_l))
end

# small-time expansion
function _small_time_pdf(u::T, z::T, K::Int) where {T <: Real}
    inf_sum = zero(T)

    k_series = (-floor(Int, 0.5 * (K - 1))):ceil(Int, 0.5 * (K - 1))
    for k in k_series
        inf_sum += ((2k + z) * exp(-((2k + z)^2 / (2u))))
    end

    return inf_sum / sqrt(2π * u^3)
end

# large-time expansion
function _large_time_pdf(u::T, z::T, K::Int) where {T <: Real}
    inf_sum = zero(T)

    for k = 1:K
        inf_sum += (k * exp(-0.5 * (k^2 * π^2 * u)) * sin(k * π * z))
    end

    return π * inf_sum
end

logpdf(d::DDM, choice, rt; ϵ::Real = 1.0e-12) = log(pdf(d, choice, rt; ϵ))
#logpdf(d::DDM, t::Real; ϵ::Real = 1.0e-12) = log(pdf(d, t; ϵ))

function logpdf(d::DDM, data::T) where {T <: NamedTuple}
    return sum(logpdf.(d, data...))
end

function logpdf(dist::DDM, data::Array{<:Tuple, 1})
    LL = 0.0
    for d in data
        LL += logpdf(dist, d...)
    end
    return LL
end

logpdf(d::DDM, data::Tuple) = logpdf(d, data...)

#########################################
# Cumulative density function           #
# Blurton, Kesselmeier, & Gondan (2012) #
#########################################

# function cdf(d::DDM, choice::Int, rt::Real = 10; ϵ::Real = 1.0e-12)
#     if choice == 1
#         (ν, α, z, τ) = params(d)
#         return _cdf(DDM(-ν, α, 1 - z, τ), rt; ϵ)
#     end

#     return _cdf(d, rt; ϵ)
# end

# # cumulative density function over the lower boundary
# function _cdf(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T <: Real}
#     if d.τ ≥ t
#         return T(NaN)
#     end

#     K_l = _K_large(d, t; ϵ)
#     K_s = _K_small(d, t; ϵ)

#     if K_l < 10 * K_s
#         return _Fl_lower(d, K_l, t)
#     end
#     return _Fs_lower(d, K_s, t)
# end

# # Large time representation of lower subdistribution
# function _Fl_lower(d::DDM{T}, K::Int, t::Real) where {T <: Real}
#     (ν, α, z, τ) = params(d)
#     F = zero(T)
#     K_series = K:-1:1
#     for k in K_series
#         F -= (
#             k / (ν^2 + k^2 * π^2 / (α^2)) *
#             exp(-ν * α * z - 0.5 * ν^2 * (t - τ) - 0.5 * k^2 * π^2 / (α^2) * (t - τ)) *
#             sin(π * k * z)
#         )
#     end
#     return _P_upper(ν, α, z) + 2 * π / (α^2) * F
# end

# # Small time representation of the upper subdistribution
# function _Fs_lower(d::DDM{T}, K::Int, t::Real) where {T <: Real}
#     (ν, α, z, τ) = params(d)
#     if abs(ν) < sqrt(eps(T))
#         return _Fs0_lower(d, K, t)
#     end

#     sqt = sqrt(t - τ)

#     S1 = zero(T)
#     S2 = zero(T)
#     K_series = K:-1:1

#     for k in K_series
#         S1 += (
#             _exp_pnorm(2 * ν * α * k, -sign(ν) * (2 * α * k + α * z + ν * (t - τ)) / sqt) - _exp_pnorm(
#             -2 * ν * α * k - 2 * ν * α * z,
#             sign(ν) * (2 * α * k + α * z - ν * (t - τ)) / sqt
#         )
#         )

#         S2 += (
#             _exp_pnorm(-2 * ν * α * k, sign(ν) * (2 * α * k - α * z - ν * (t - τ)) / sqt) - _exp_pnorm(
#             2 * ν * α * k - 2 * ν * α * z,
#             -sign(ν) * (2 * α * k - α * z + ν * (t - τ)) / sqt
#         )
#         )
#     end

#     return _P_upper(ν, α, z) +
#            sign(ν) * (
#         (
#             cdf(Normal(), -sign(ν) * (α * z + ν * (t - τ)) / sqt) -
#             _exp_pnorm(-2 * ν * α * z, sign(ν) * (α * z - ν * (t - τ)) / sqt)
#         ) +
#         S1 +
#         S2
#     )
# end

# # Zero drift version
# function _Fs0_lower(d::DDM{T}, K::Int, t::Real) where {T <: Real}
#     (_, α, z, τ) = params(d)
#     F = zero(T)
#     K_series = K:-1:0
#     for k in K_series
#         F -= (
#             cdf(Distributions.Normal(), (-2 * k - 2 + z) * α / sqrt(t - τ)) +
#             cdf(Distributions.Normal(), (-2 * k - z) * α / sqrt(t - τ))
#         )
#     end
#     return 2 * F
# end
# # Number of terms required for large time representation
# function _K_large(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T <: Real}
#     (ν, α, z, τ) = params(d)
#     x = t - τ
#     sqrtL1 = sqrt(1 / x) * α / π
#     sqrtL2 = sqrt(
#         max(
#         1,
#         -2 / x * α * α / π / π *
#         (log(ϵ * π * x / 2 * (ν * ν + π * π / α / α)) + ν * α * z + ν * ν * x / 2)
#     ),
#     )
#     return ceil(Int, max(sqrtL1, sqrtL2))
# end

# # Number of terms required for small time representation
# function _K_small(d::DDM{T}, t::Real; ϵ::Real = 1.0e-12) where {T <: Real}
#     (ν, α, z, τ) = params(d)
#     if abs(ν) < sqrt(eps(T))
#         return ceil(
#             Int,
#             max(
#                 0,
#                 z / 2 -
#                 sqrt(t - τ) / (2 * α) * quantile(Normal(), max(0, min(1, ϵ / (2 - 2 * z))))
#             )
#         )
#     end
#     if ν > 0
#         return _K_small(DDM(-ν, α, z, τ), t; ϵ = exp(-2 * α * z * ν) * ϵ)
#     end
#     S2 = z - 1 + 1 / (2 * ν * α) * log(ϵ / 2 * (1 - exp(2 * ν * α)))
#     S3 = (0.535 * sqrt(2 * (t - τ)) + ν * (t - τ) + α * z) / (2 * α)
#     S4 =
#         z / 2 -
#         sqrt(t - τ) / (2 * α) * quantile(
#             Normal(),
#             max(
#                 0,
#                 min(
#                     1,
#                     ϵ * α / (0.3 * sqrt(2 * π * (t - τ))) *
#                     exp(ν^2 * (t - τ) / 2 + ν * α * z)
#                 )
#             )
#         )
#     return ceil(Int, max(0, S2, S3, S4))
# end
# # Probability for absorption at upper barrier
# function _P_upper(ν::T, α::T, z::T) where {T <: Real}
#     e = exp(-2 * ν * α * (1 - z))
#     if isinf(e)
#         return 1.0
#     end
#     if abs(e - 1) < sqrt(eps(T))
#         return 1 - z
#     end
#     return (1 - e) / (exp(2 * ν * α * z) - e)
# end

# # Calculates exp(a) * pnorm(b) using an approximation by Kiani et al. (2008)
# function _exp_pnorm(a::T, b::T) where {T <: Real}
#     r = exp(a) * cdf(Distributions.Normal(), b)
#     if isnan(r) && b < -5.5
#         r = (1 / sqrt(2)) * exp(a - b^2 / 2) * (0.5641882 / (b^3) - 1 / (b * sqrt(π)))
#     end
#     return r
# end

"""
    rand(dist::DDM)

Generate a random rt for the Diffusion Decision Model (negative coding)

# Arguments
- `dist`: model object for the Diffusion Decision Model. 
"""
function rand(rng::AbstractRNG, d::DDM)
    return _rand_rejection(rng, d)
end

# Rejection-based Method for the Symmetric Wiener Process(Tuerlinckx et al., 2001 based on Lichters et al., 1995)
# adapted from the RWiener R package, note, here σ = 0.1
function _rand_rejection(rng::AbstractRNG, d::DDM)
    (ν, α, z, τ, η, sz, st, σ) = params(d)

    ϵ = 1.0e-15

    D = σ^2 / 2

    bb = z - sz / 2 + sz * rand(rng)
    zz = (bb * α)
    νν = ν + randn(rng) * η

    total_time = 0.0
    start_pos = 0.0
    Aupper = α - zz
    Alower = -zz
    radius = min(abs(Aupper), abs(Alower))
    λ = 0.0
    F = 0.0
    prob = 0.0

    while true
        if νν == 0
            λ = (0.25D * π^2) / (radius^2)
            F = 1.0
            prob = 0.5
        else
            λ = ((0.25 * νν^2) / D) + ((0.25 * D * π^2) / (radius^2))
            F = (D * π) / (radius * νν)
            F = F^2 / (1 + F^2)
            prob = exp((radius * νν) / D)
            prob = prob / (1 + prob)
        end

        r = rand(rng)
        dir = r < prob ? 1 : -1
        l = -1.0
        s1 = 0.0
        s2 = 0.0

        # Tuerlinckx et al. (2001; eq. 16)  
        while s2 > l
            s1 = rand(rng)
            s2 = rand(rng)
            tnew = 0.0
            tδ = 0.0
            uu = zero(Int)

            while (abs(tδ) > ϵ) || (uu == 0)
                uu += 1
                tt = 2 * uu + 1
                tδ = tt * (uu % 2 == 0 ? 1 : -1) * (s1^(F * tt^2))
                tnew += tδ
            end

            l = 1 + (s1^(-F)) * tnew
        end

        total_time += abs(log(s1)) / λ
        dir = start_pos + dir * radius
        ττ = τ - st / 2 + st * rand(rng)

        if (dir + ϵ) > Aupper
            rt = total_time + ττ
            choice = 1
            return (; choice, rt)
        elseif (dir - ϵ) < Alower
            rt = total_time + ττ
            choice = 2
            return (; choice, rt)
        else
            start_pos = dir
            radius = min(abs(Aupper - start_pos), (abs(Alower - start_pos)))
        end
    end
end

"""
    n_options(dist::DDM)

Returns 2 for the number of choice options

# Arguments

- `d::DDM`: a model object for the drift diffusion model
"""
n_options(d::DDM) = 2

function simulate(rng::AbstractRNG, model::DDM; Δt = 0.001)
    (;ν, α, z, η, sz) = model

    zz = (z - sz/2) + ((z + sz/2) - (z - sz/2)) * rand(rng)
    νν = rand(rng, Distributions.Normal(ν, η))
    x = α * zz
    t = 0.0
    evidence = [x]
    time_steps = [t]
    while (x < α) && (x > 0)
        t += Δt
        x += νν * Δt + rand(rng, Normal(0.0, 1.0)) * √(Δt)
        push!(evidence, x)
        push!(time_steps, t)
    end
    return time_steps, evidence
end
