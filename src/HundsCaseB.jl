using Parameters
using InfiniteArrays
using CompositeStructs

# Define the spherical tensor T^k_q(ϵ), here for linear and symmetric top molecules
const T = [
    0 0 0
    -1/√3 0 -1/√6
    0 0 0
    ]

abstract type HundsCaseB <: BasisState end
export HundsCaseB

@composite Base.@kwdef struct HundsCaseB_Rot <: HundsCaseB
    E::Float64 = 0.0
    S::Rational 
    I::Rational
    Λ::Rational
    N::Rational
    J::Rational 
    F::Rational
    M::Rational
    constraints = (
        N = abs(Λ):∞,
        J = abs(N - S):abs(N + S),
        F = abs(J - I):abs(J + I),
        M = -F:F
    )
end
export HundsCaseB_Rot

function unpack(state::HundsCaseB)
    return (state.S, state.I, state.Λ, state.N, state.J, state.F, state.M)
end
export unpack

function Rotation(state::HundsCaseB, state′::HundsCaseB)
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return (N * (N + 1) - Λ^2) * δ(Λ, Λ′) * δ(N, N′) * δ(J, J′) * δ(F, F′) * δ(M, M′)
end
export Rotation

function RotationDistortion(state::HundsCaseB, state′::HundsCaseB)
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return - (N * (N + 1) - Λ^2)^2 * δ(Λ, Λ′) * δ(N, N′) * δ(J, J′) * δ(F, F′) * δ(M, M′)
end
export RotationDistortion
    
# Spin-rotation for zero internuclear axis angular momentum, i.e., Λ = 0
function SpinRotation_Λ0(state::HundsCaseB, state′::HundsCaseB)
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return (-1)^(N + S + J) * 
        sqrt(S * (S + 1) * (2S + 1) * N * (N + 1) * (2N + 1)) * 
        wigner6j_(S, N, J, N, S, 1) *
        δ(Λ, Λ′) * δ(N, N′) * δ(J, J′) * δ(F, F′) * δ(M, M′)
end
export SpinRotation_Λ0

# Spin-rotation for Λ != 0, reduces to above matrix element for Λ = 0
function SpinRotation(state::HundsCaseB, state′::HundsCaseB)
    # Hirota, eq. (2.3.35)
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return (-1)^(J + S + N) * (-1)^(N - Λ) *
        sqrt( S * (S + 1) * (2S + 1) * (2N + 1) * (2N′ + 1) ) *
        wigner6j_(N′, S ,J, S, N, 1) *
        sum( sqrt(2k + 1) *
            (
                (-1)^k * 
                sqrt( N′ * (N′ + 1) * (2N′ + 1) ) * 
                wigner6j_(1, 1, k, N, N′, N′) +
                sqrt( N * (N + 1) * (2N + 1) ) *
                wigner6j_(1, 1, k, N′, N, N)
            ) *
                wigner3j_(N, k, N′, -Λ, q, Λ′) * T[q+2, k+1]
            for k in 0:2, q in -1:1
        ) *
        δ(J, J′) * δ(F, F′) * δ(M, M′)
end
export SpinRotation

function Hyperfine_IS(state::HundsCaseB, state′::HundsCaseB)
    # Fermi-contact interaction
    # Hirota, pg. 39
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return (-1)^(J′ + F + I + J + N + S + 1) *
        sqrt( (2J′ + 1) * (2J + 1) * S * (S + 1) * (2S + 1) * I * (I + 1) * (2I + 1) ) *
        wigner6j_(I, J, F, J′, I, 1) *
        wigner6j_(S, J, N, J′, S, 1) *
        δ(Λ, Λ′) * δ(N, N′) * δ(F, F′) * δ(M, M′)
end
export Hyperfine_IS

function Hyperfine_Dipolar(state::HundsCaseB, state′::HundsCaseB)
    # Dipolar interaction term, from c(Iz ⋅ Sz)
    # Hirota, pg. 39
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return - sqrt(30) * (-1)^(J′ + I + F + N) *
        wigner6j_(I, J, F, J′, I, 1) * 
        wigner9j_(N, N′, 2, S, S, 1, J, J′, 1) * 
        wigner3j_(N, 2, N′, -Λ, 0, Λ′) *
        sqrt( S * (S + 1) * (2S + 1) * I * (I + 1) * (2I + 1) * (2J + 1) * (2J′ + 1) * (2N + 1) * (2N′ + 1) ) *
        δ(F, F′) * δ(M, M′)
end
export Hyperfine_Dipolar

function ℓDoubling(state::HundsCaseB, state′::HundsCaseB)
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return (-1)^(N - Λ) *
        (1 / (2 * sqrt(6))) *
        sqrt( (2N - 1) * (2N) * (2N + 1) * (2N + 2) * (2N + 3) ) *
        sum(
            wigner3j_(N, 2, N, -Λ, 2q, Λ′)
            for q in [-1,1]
        ) *
        δ(abs(Λ′ - Λ), 2) *
        δ(J, J′) * δ(F, F′) * δ(M, M′)
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

function Stark(state::HundsCaseB, state′::HundsCaseB)
    # Hirota, equation (2.5.35)
    S,  I,  Λ,  N,  J,  F,  M  = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return -(-1)^(F - M) * 
        wigner3j_(F, 1, F′, -M, 0, M′) * 
        (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) *
        wigner6j_(J, F, I, F′, J′, 1) *
        (-1)^(N + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) ) *
        wigner6j_(N, J, S, J′, N′, 1) *
        (-1)^(N - Λ) * sqrt( (2N + 1) * (2N′ + 1) ) *
        wigner3j_(N, 1, N′, -Λ, 0, Λ′)
end
export Stark

function Zeeman(state::HundsCaseB, state′::HundsCaseB)
    # Hirota, equation (2.5.16) and (2.5.19)
    S, I, Λ, N, J, F, M = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return (-1)^(F - M) *
        wigner3j_(F, 1, F′, -M, 0, M) *
        (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * 
        wigner6j_(J′, F′, I, F, J, 1) *
        (-1)^(N + S + J + 1) * 
        sqrt( (2J + 1) * (2J′ + 1) * S * (S + 1) * (2S + 1) ) *
        wigner6j_(S, J′, N, J, S, 1) * 
        δ(Λ, Λ′) * δ(N, N′) * δ(M, M′)
end
export Zeeman

function TDM(state::HundsCaseB, state′::HundsCaseB, p::Int64)
    S,  I,  Λ,  N,  J,  F,  M  = unpack(state)
    S′, I′, Λ′, N′, J′, F′, M′ = unpack(state′)
    return -(-1)^p * (-1)^(F - M) * wigner3j_(F, 1, F′, -M, p, M′) *
#     return (-1)^(F - M) * wigner3j_(F, 1, F′, -M, p, M′) *
        (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j_(J, F, I, F′, J′, 1) *
        (-1)^(N + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) ) * wigner6j_(N, J, S, J′, N′, 1) *
        (-1)^(N - Λ) * sqrt( (2N + 1) * (2N′ + 1) ) * sum(wigner3j_(N, 1, N′, -Λ, q, Λ′) for q in -1:1)
end
TDM(state, state′) = sum(TDM(state, state′, p) for p ∈ -1:1)

function TDM(state::State, state′::State, args...)
    tdm = 0.0
    for (i, basis_state) in enumerate(state.basis)
        for (j, basis_state′) in enumerate(state′.basis)
            tdm += state.coeffs[i] * conj(state′.coeffs[j]) * TDM(basis_state, basis_state′, args...)
        end
    end
    return tdm
end
export TDM
    
    
