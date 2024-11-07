using Parameters
using InfiniteArrays
using CompositeStructs
using HalfIntegers

Base.@kwdef struct AtomicState <: BasisState
    E::Float64 = 0.0
    L::HalfInt
    S::HalfInt
    J::HalfInt
    I::HalfInt
    F::HalfInt
    M::HalfInt
    constraints = (
        J = abs(L - S):abs(L + S),
        F = abs(J - I):abs(J + I),
        M = -F:F
    )
end
export AtomicState

function unpack(state::AtomicState)
    return (state.L, state.S, state.J, state.I, state.F, state.M)
end
export unpack

function identity(state::AtomicState, state′::AtomicState)
    L,  S,  J,  I,  F,  M  = unpack(state)
    L′, S′, J′, I′, F′, M′ = unpack(state′)
    val = zero(Float64)
    if δ(L,L′) && δ(J,J′) && δ(F,F′) && δ(M,M′)
        val += 1
    end
    return val
end
export zero_point_energy

function hyperfine_magnetic_dipole(state::AtomicState, state′::AtomicState)
    L,  S,  J,  I,  F,  M  = unpack(state)
    L′, S′, J′, I′, F′, M′ = unpack(state′)
    val = zero(Float64)
    if δ(L,L′) && δ(J,J′) && δ(F,F′) && δ(M,M′)
        K = F*(F+1) - I*(I+1) - J*(J+1)
        val += K/2
    end
    return val
end
export hyperfine_magnetic_dipole

function hyperfine_electric_quadrupole(state::AtomicState, state′::AtomicState)
    L,  S,  J,  I,  F,  M  = unpack(state)
    L′, S′, J′, I′, F′, M′ = unpack(state′)
    val = zero(Float64)
    if δ(L,L′) && δ(J,J′) && δ(F,F′) && δ(M,M′)
        K = F*(F+1) - I*(I+1) - J*(J+1)
        numer = (3/2) * K * (K+1) - 2I * (I+1) * J * (J+1)
        denom = 4I * (2I - 1) * J * (2J - 1)
        val += numer / denom
    end
    return val
end
export hyperfine_electric_quadrupole

function zeeman_BdotS(state::AtomicState, state′::AtomicState, p::Int64)
    L,  S,  J,  I,  F,  M  = unpack(state)
    L′, S′, J′, I′, F′, M′ = unpack(state′)
    val = zero(Float64)
    if δ(L,L′)
        val += (
            (-1)^p * (-1)^(F′ - M′) * wigner3j(F′, 1, F, -M′, p, M)
        * (-1)^(J′ + I + F + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
        * (-1)^(L + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) * S * (S + 1) * (2S + 1) ) * wigner6j(S, J′, L, J, S, 1)
        )
    end
    return val
end
export zeeman_BdotS

function zeeman_BdotL(state::AtomicState, state′::AtomicState, p::Int64)
    L,  S,  J,  I,  F,  M  = unpack(state)
    L′, S′, J′, I′, F′, M′ = unpack(state′)
    val = zero(Float64)
    if δ(L,L′)
        val += (
            (-1)^p * (-1)^(F′ - M′) * wigner3j(F′, 1, F, -M′, p, M)
        * (-1)^(J′ + I + F + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J′, F′, I, F, J, 1)
        * (-1)^(L + S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) * L * (L + 1) * (2L + 1) ) * wigner6j(L′, J′, S, J, L, 1)
        )
    end
    return val
end
export zeeman_BdotL

function TDM(state::AtomicState, state′::AtomicState, p::Int64)
    L,  S,  J,  I,  F,  M = unpack(state)
    L′, S′, J′, I′, F′, M′ = unpack(state′)
    val = zero(Float64)
    if ~δ(J,J′)
        val += (-1)^p * (-1)^(F - M) * wigner3j(F, 1, F′, -M, p, M′) * 
        (-1)^(J + I + F′ + 1) * sqrt( (2F + 1) * (2F′ + 1) ) * wigner6j(J, F, I, F′, J′, 1)
    end
    return val
end
