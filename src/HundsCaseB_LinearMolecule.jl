using Parameters
using InfiniteArrays
using CompositeStructs
using HalfIntegers

# Define the spherical tensor T^k_q(ϵ), here for linear and symmetric top molecules
const T_kq = [
    0.0 0.0 0.0
    -2/√3 0.0 -2/√6
    0.0 0.0 0.0
    ]
export T_kq

abstract type HundsCaseB <: BasisState end
export HundsCaseB

@composite Base.@kwdef struct HundsCaseB_LinearMolecule <: HundsCaseB
    E::Float64 = 0.0
    label::String = ""
    v_1::HalfInt = 0
    v_2::HalfInt = 0
    v_3::HalfInt = 0
    S::HalfInt = 0
    I::HalfInt = 0
    Λ::HalfInt = 0
    ℓ::HalfInt = 0
    K::HalfInt = 0
    N::HalfInt = 0
    J::HalfInt = 0
    F::HalfInt = 0
    M::HalfInt = 0
    constraints = (
        K = Λ + ℓ,
        N = abs(K):∞,
        J = abs(N - S):abs(N + S),
        F = abs(J - I):abs(J + I),
        M = -F:F
    )
end
export HundsCaseB_LinearMolecule

function overlap(state::HundsCaseB, state′::HundsCaseB)
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return δ(S,S′) * δ(I,I′) * δ(Λ,Λ′) * δ(N,N′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
end
export overlap

function unpack(state::HundsCaseB_LinearMolecule)
    (; v_1, v_2, v_3, S, I, Λ, ℓ, K, N, J, F, M) = state
    return v_1, v_2, v_3, S, I, Λ, ℓ, K, N, J, F, M
end
export unpack

function Identity(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    return (state == state′)
end
export Identity

function T(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    return state.E * (state == state′)
end
export T

function Rotation(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :K, :Λ, :ℓ, :S, :I, :N, :J, :F, :M)
        return 0.0
    else
        return N * (N + 1) - Λ^2
    end
end
export Rotation

function RotationDistortion(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :K, :Λ, :ℓ, :S, :I, :N, :J, :F, :M)
        return 0.0
    else
        return - (N * (N + 1) - Λ^2)^2
    end
end
export RotationDistortion
    
# Spin-rotation for zero internuclear axis angular momentum, i.e., Λ = 0
function SpinRotation_Λ0(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :K, :Λ, :ℓ, :S, :I, :N, :J, :F, :M)
        return 0.0
    else 
        return (
                (-1)^(N + S + J) 
                * sqrt(S * (S + 1) * (2S + 1) * N * (N + 1) * (2N + 1))
                * wigner6j(S, N, J, N, S, 1)
            )
    end
end
export SpinRotation_Λ0

# Spin-rotation for Λ != 0, reduces to above matrix element for Λ = 0
function SpinRotation(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    # Hirota, eq. (2.3.35)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :J, :F, :M)
        return 0.0
    else
        return (
                (1/2) * (-1)^(J′ + S + N) * (-1)^(N - K) 
                * sqrt( S * (S + 1) * (2S + 1) * (2N + 1) * (2N′ + 1) ) 
                * wigner6j(N′, S ,J′, S, N, 1)
                * sum( sqrt(2k + 1)
                * (
                    (-1)^k *
                    sqrt( N′ * (N′ + 1) * (2N′ + 1) ) * wigner6j(1, 1, k, N, N′, N′) 
                    + 
                    sqrt( N * (N + 1) * (2N + 1) ) * wigner6j(1, 1, k, N′, N, N)
                    )
                    * wigner3j(N, k, N′, -K, q, K′) * T_kq[q+2, k+1]
                for k ∈ 0:2, q ∈ -1:1
            )
        )
    end
end
export SpinRotation

function Hyperfine_IS(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    # Fermi-contact interaction
    # Hirota, pg. 39
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :K, :Λ, :ℓ, :S, :I, :N, :F, :M)
        return 0.0
    else
        return (
                (-1)^(N′ + S + J) * (-1)^(J′ + I + F′ + 1)
                * sqrt( (2J′ + 1) * (2J + 1) * S * (S + 1) * (2S + 1) * I * (I + 1) * (2I + 1) )
                * wigner6j(I, J, F′, J′, I, 1)
                * wigner6j(S, J, N′, J′, S, 1)
            )
    end
end
export Hyperfine_IS

function Hyperfine_Dipolar(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    # Dipolar interaction term, from c(Iz ⋅ Sz)
    # Hirota, pg. 39
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :Λ, :ℓ, :F, :M)
        return 0.0
    else
        return (
                sqrt(30) * (-1)^(N - K) * (-1)^(J′ + I + F + 1) *
                wigner6j(I, J, F′, J′, I, 1) * 
                wigner9j(N, N′, 2, S, S, 1, J, J′, 1) * 
                wigner3j(N, 2, N′, -K, 0, K′) *
                sqrt( S * (S + 1) * (2S + 1) * I * (I + 1) * (2I + 1) * (2J + 1) * (2J′ + 1) * (2N + 1) * (2N′ + 1) )
            )
    end
end
export Hyperfine_Dipolar

function ℓDoubling(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :Λ, :S, :I, :N, :J, :F, :M)
        return 0.0
    else
        return δ(abs(ℓ′-ℓ), 2) * (
                (-1)^(N - K) *
                (1 / (2 * sqrt(6))) *
                sqrt( (2N - 1) * (2N) * (2N + 1) * (2N + 2) * (2N + 3) ) *
                sum(
                    wigner3j(N, 2, N′, -K, 2q, K′)
                    for q ∈ (-1,1)
                )                   
            )
    end
end
export ℓDoubling

# function Hyperfine_IK(state::HundsCaseB, state′::HundsCaseB)
#     S, I, N, Λ, J, F, M = unpack(state)   
#     S′, I′, N′, Λ′, J′, F′, M′ = unpack(state′)
#     return (-1)^(F + I + N′ + S + 2J + 1 + N - Λ) * 
#         sqrt( I * (I + 1) * (2I + 1) * (2J + 1) * (2J′ + 1) * (2N + 1) * (2N′ + 1) ) *
#         wigner6j(I, J, F, J′, I, 1) * 
#         wigner6j(N, J, S, J′, N′, 1) * 
#         wigner3j(N′, 1, N, -Λ, 0, Λ) *
#         δ(Λ, Λ′) * δ(F, F′) * δ(M, M′)
# end
# export Hyperfine_IK

# function Hyperfine_SK(state::HundsCaseB, state′::HundsCaseB)
#     S, I, N, Λ, J, F, M = unpack(state)   
#     S′, I′, N′, Λ′, J′, F′, M′ = unpack(state′)
#     return (-1)^(2N + J + S - Λ) * sqrt(S * (S + 1) * (2S + 1) * (2N + 1) * (2N′ + 1)) *
#         wigner6j(S, N, J, N′, S, 1) * 
#         wigner3j(N′, 1, N, -Λ, 0, Λ) *
#         δ(Λ, Λ′) * δ(J, J′) * δ(F, F′) * δ(M, M′)
# end
# export Hyperfine_SK

function Stark(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule)
    # Hirota, equation (2.5.35)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)
    if ~delta(state, state′, :ℓ)
        return 0.0
    else
        return (
                - (-1)^(F - M) * wigner3j(F, 1, F′, -M, 0, M′)
                * (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
                * (-1)^(N + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) ) * wigner6j(N′, J′, S, J, N, 1)
                * (-1)^(N - K) * sqrt( (2N + 1) * (2N′ + 1) ) * wigner3j(N, 1, N′, -K, 0, K′) 
        )
    end
end
export Stark

function Zeeman(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule, p::Int64)
    # Hirota, equation (2.5.16) and (2.5.19)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)
    if ~delta(state, state′, :ℓ, :Λ, :K, :N)
        return 0.0
    else
        return (
                  (-1)^p * (-1)^(F - M) * wigner3j(F, 1, F′, -M, p, M′)
                * (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
                * (-1)^(S + N′ + J + 1) * sqrt( (2J + 1) * (2J′ + 1) * S * (S + 1) * (2S + 1) ) * wigner6j(S, J′, N, J, S, 1)
        )
    end
end
export Zeeman

# function Zeeman(state::HundsCaseB, state′::HundsCaseB, B::Vector{Float64})
#     # Hirota, equation (2.5.16) and (2.5.19)
#     S,  I,  Λ,  N,  J,  F,  M  = unpack(state)
#     S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
#     if ~δ(Λ, Λ′) || ~δ(N, N′)
#         return 0.0
#     else
#         return (
#                   (-1)^(J′ + I + F + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
#                 * (-1)^(N + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) * S * (S + 1) * (2S + 1) ) * wigner6j(S, J′, N, J, S, 1)
#             ) * 
#             sum(
#                 B[p+2] * (-1)^p * (-1)^(F′ - M′) * wigner3j(F′, 1, F, -M′, -p, M) for p ∈ -1:1
#             )
#     end
# end
# export Zeeman

function Σ(state::HundsCaseB_LinearMolecule)
    @unpack Λ, N, S, J = state
    val = zero(Float64)
    for Σ ∈ -S:S
        Ω = Λ + Σ
        val += Σ * (2N + 1) * wigner3j(J, S, N, Ω, -Σ, -Λ)^2
    end
    return val
end
# function Σ(state::State)
#     val = zero(Float64)
#     for i ∈ eachindex(state.basis), j ∈ eachindex(state.basis)
#         val += conj(state.coeffs[i]) * Σ(state.basis[j]) * state.coeffs[j]
#     end
#     return val
# end
Σ(state::State) = sum(Σ(state.basis[i]) * state.coeffs[i] * conj(state.coeffs[i]) for i ∈ eachindex(state.basis))
export Σ

function TDM_magnetic(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule, p::Int64)
    # Assumes magnetic moment aligned along z-axis of molecule-fixed axis
    S,  I,  Λ,  N,  J,  F,  M  = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    if ~δ(Λ, Λ′) || ~δ(N, N′)
        return 0.0
    else
        return (
                (-1)^p * (-1)^(F′ - M′) * wigner3j(F′, 1, F, -M′, -p, M)
                * (-1)^(J′ + I + F + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
                * (-1)^(N + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) * S * (S + 1) * (2S + 1) ) * wigner6j(S, J′, N, J, S, 1)
            )
    end
end
TDM_magnetic(state::State, state′::State, p::Int64) = extend_operator(TDM_magnetic, state, state′, p)
export TDM_magnetic

function TDM_vibrational(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule, p::Int64)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)
    return (
        - (-1)^p * (-1)^(F - M) * wigner3j(F, 1, F′, -M, p, M′)
        * (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
        * (-1)^(N + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) ) * wigner6j(N′, J′, S, J, N, 1)
        * (-1)^(N - K) * sqrt( (2N + 1) * (2N′ + 1) ) * sum(wigner3j(N, 1, N′, -K, q, K′) for q ∈ -1:1)
    )
end

function TDM(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule, p::Int64)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)
    if ~delta(state, state′, :ℓ)
        return 0.0
    else
        return (
            - (-1)^p * (-1)^(F - M) * wigner3j(F, 1, F′, -M, p, M′)
            * (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
            * (-1)^(N + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) ) * wigner6j(N′, J′, S, J, N, 1)
            * (-1)^(N - K) * sqrt( (2N + 1) * (2N′ + 1) ) * sum(wigner3j(N, 1, N′, -K, q, K′) for q ∈ -1:1)
        )
    end
end
TDM(state, state′) = extend_operator(TDM, state, state′, p)
TDM_vibrational(state, state′, p) = extend_operator(TDM_vibrational, state, state′, p)
export TDM
export TDM_vibrational

# d(state, state′) = extend_operator(TDM, state, state′, 0)
# export d

function 𝒫(K,P,ϵ)
    val = 0.0
    ϵm1, ϵ0, ϵp1 = ϵ
    if P == 0
        if K == 0
            val += 1.0
        elseif K == 1
            val += ϵp1 * conj(ϵp1) - ϵm1 * conj(ϵm1)
        elseif K == 2
            val += -(1/2) * (1 - 3 * ϵ0 * conj(ϵ0))
        end
    elseif P == +1
        if K == 1
            val += - (ϵ0 * conj(ϵm1) + conj(ϵ0) * ϵp1)
        elseif K == 2
            val += sqrt(3/2) * (-ϵ0 * conj(ϵm1) + conj(ϵ0) * ϵp1)
        end
    elseif P == -1
        if K == 1
            val += (ϵ0 * conj(ϵp1) + conj(ϵ0) * ϵm1)
        elseif K == 2
            val += sqrt(3/2) * (-ϵ0 * conj(ϵp1) + conj(ϵ0) * ϵm1)
        end
    elseif P == +2
        if K == 2
            val += -sqrt(3/2) * conj(ϵm1) * ϵp1
        end
    elseif P == -2
        if K == 2
            val += -sqrt(3/2) * conj(ϵp1) * ϵm1
        end
    end
    return val
end             
export 𝒫
                                                            
function polarizability(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule, α, ϵ)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)
    val = 0.0
    for L in 0:2
        for P in -L:L
            val += (
                -(-1)^P
                * (-1)^(F - M)
                * wigner3j(F, L, F′, -M, P, M′)
                * (-1)^(J + I + F′ + L) * sqrt( (2F + 1) * (2F′ + 1) )
                * wigner6j(J, F, I, F′, J′, L)
                * (-1)^(N + S + J′ + L) * sqrt( (2J + 1) * (2J′ + 1) )
                * wigner6j(N, J, S, J′, N′, L)
                * (-1)^(N - K) * sqrt( (2N + 1) * (2N′ + 1) )
                * wigner3j(N, L, N′, -K, 0, K′)
                * α[L+1] * 𝒫(L, -P, ϵ)
            ) 
        end
    end
    return val
end
export polarizability

function polarizability_parity(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule, α, ϵ)
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, J′, F′, M′ = unpack(state′)
    val = 0.0
    if delta(state, state′, :S, :I, :ℓ)
        for k ∈ 0:2
            for p ∈ -k:k
                val += -(
                    (-1)^(F - M) * wigner3j(F, k, F′, -M, p, M′)
                    * (-1)^(F′ + J + I + k) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J, F, I, F′, J′, k)
                    * (-1)^(N + N′) * sqrt( (2N + 1) * (2N′ + 1) )
                    * sqrt( (2J + 1) * (2J′ + 1) )
                    * sum(
                        wigner3j(J, N, S, K+Σ, -K, -Σ)
                        * wigner3j(J′, N′, S′, K′+Σ, -K′, -Σ)
                        * (-1)^(J - Σ) * wigner3j(J, k, J′, -Σ, 0, Σ)
                        for Σ ∈ -S:S
                    )
                ) * α[k+1] * 𝒫(k, -p, ϵ)
            end
        end
    end
    return val
end
export polarizability_parity