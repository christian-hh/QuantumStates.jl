using Parameters
using InfiniteArrays
using CompositeStructs
using HalfIntegers

Base.@kwdef mutable struct AngularMomentumState_withSpinRotation <: BasisState
    E::Float64 = 0.0
    N::HalfInt
    S::HalfInt
    J::HalfInt
    M::HalfInt
    constraints = (
        J = abs(N - S):abs(N + S),
        M = -J:J,
    )
end
export AngularMomentumState_withSpinRotation

function unpack(state::AngularMomentumState_withSpinRotation)
    return (state.J, state.N, state.S, state.M)
end
export unpack

function Identity(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation)
    J,  N,  S,  M  = unpack(state)
    J′, N′, S′, M′ = unpack(state′)
    if ~δ(J, J′) || ~δ(N, N′) || ~δ(M, M′)
        return 0.0
    else
        return 1.0
    end
end
export Identity

function Rotation(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation)
    J,  N,  S,  M  = unpack(state)
    J′, N′, S′, M′ = unpack(state′)
    if ~δ(J, J′) || ~δ(N, N′) || ~δ(M, M′)
        return 0.0
    else
        return N * (N + 1)
    end
end
export Rotation

function SpinRotation(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation)
    J,  N,  S,  M  = unpack(state)
    J′, N′, S′, M′ = unpack(state′)
    if ~δ(N, N′) || ~δ(J, J′) || ~δ(M, M′)
        return 0.0
    else 
        return (
                (-1)^(N + S + J) 
                * sqrt(S * (S + 1) * (2S + 1) * N * (N + 1) * (2N + 1))
                * wigner6j(S, N, J, N, S, 1)
            )
    end
end
export SpinRotation

function TDM(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation, p::Int64)
    J,  N,  S,  M  = unpack(state)
    J′, N′, S′, M′ = unpack(state′)
    return (
        (-1)^p * (-1)^(N - M) * wigner3j_(N, 1, N′, M, p, -M′) * sqrt(2N + 1)
    )
end

function TDM_magnetic(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation, p::Int64)
    # Assumes magnetic moment aligned along z-axis of molecule-fixed axis
    J,  N,  S,  M  = unpack(state)
    J′, N′, S′, M′ = unpack(state′)
    return -(
        (-1)^p * (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, M, p, -M′)
    )
end
# TDM_magnetic(state::State, state′::State, p::Int64) = extend_operator(TDM_magnetic, state, state′, p)

function d_moment(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation, p)
    J,  N,  S,  M  = unpack(state)
    J′, N′, S′, M′ = unpack(state′)
    return (
        -(-1)^p * (-1)^(J-M) * wigner3j(J, 1, J′, -M, p, M′) 
        * (-1)^(S + J′ + 1) * sqrt( (2J + 1) * (2J′ + 1) ) * wigner6j(N, J, S, J′, N′, 1)
        * sqrt( (2N + 1) * (2N′ + 1) ) * wigner3j(N, 1, N′, 0, 0, 0)
    )
end
d⁰(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation) = d_moment(state, state′, 0)
d⁺(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation) = d_moment(state, state′, +1)
d⁻(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation) = d_moment(state, state′, -1)
export d⁰, d⁺, d⁻

function SplitMStates(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation)
    J,  N,  S,  M  = unpack(state)
    J′, N′, S′, M′ = unpack(state′)
    if ~δ(J, J′) || ~δ(M, M′) || ~δ(N, N′)
        return 0.0
    else
        return M
    end
end
export SplitMStates

# function H(state::AngularMomentumState_withSpinRotation, state′::AngularMomentumState_withSpinRotation)
#     δ(state, state′) * state.E
# end
# export H