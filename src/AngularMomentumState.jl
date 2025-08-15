using Parameters
using InfiniteArrays
using CompositeStructs
using HalfIntegers

Base.@kwdef mutable struct AngularMomentumState <: BasisState
    E::Float64 = 0.0
    label::String = ""
    N::HalfInt
    M::HalfInt
    constraints = (
        M = -N:N,
    )
end
export AngularMomentumState

function unpack(state::AngularMomentumState)
    return (state.N, state.M)
end
export unpack

function Identity(state::AngularMomentumState, state′::AngularMomentumState)
    N,  M  = unpack(state)
    N′, M′ = unpack(state′)
    if ~δ(N, N′) || ~δ(M, M′)
        return 0.0
    else
        return 1.0
    end
end
export Identity

function Rotation(state::AngularMomentumState, state′::AngularMomentumState)
    N,  M  = unpack(state)
    N′, M′ = unpack(state′)
    return I(state, state′) * N * (N + 1)
end
export Rotation

function TDM(state::AngularMomentumState, state′::AngularMomentumState, p::Int64)
    N,  M  = unpack(state)
    N′, M′ = unpack(state′)
    return (state.label != state′.label) * (
        (-1)^p * (-1)^(N - M) * wigner3j(N, 1, N′, -M, -p, M′) * sqrt(2N + 1)
    )
end

function TDM_magnetic(state::AngularMomentumState, state′::AngularMomentumState, p::Int64)
    # Assumes magnetic moment aligned along z-axis of molecule-fixed axis
    N,  M  = unpack(state)
    N′, M′ = unpack(state′)
    return (
        (-1)^p * (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, -p, M′)
    )
end
# TDM_magnetic(state::State, state′::State, p::Int64) = extend_operator(TDM_magnetic, state, state′, p)

function d₀(state::AngularMomentumState, state′::AngularMomentumState)
    N,  M  = state.N,  state.M
    N′, M′ = state′.N, state′.M
    (-1)^M * sqrt((2N+1)*(2N′+1)) * wigner3j(N, 1, N′, -M, 0, M′) * wigner3j(N, 1, N′, 0, 0, 0)
end

function Zeeman(state::AngularMomentumState, state′::AngularMomentumState)
    N,  M  = state.N,  state.M
    N′, M′ = state′.N, state′.M
    if ~δ(M, M′) || ~δ(N, N′)
        return 0.0
    else
        return (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, 0, M′)
    end
end
export Zeeman

# function H(state::AngularMomentumState, state′::AngularMomentumState)
#     δ(state, state′) * state.E
# end
# export H

function overlap(state::AngularMomentumState, state′::AngularMomentumState)
    N, M = state.N, state.M
    N′, M′ = state′.N, state′.M
    overlap = zero(Float64)
    if ~δ(M, M′) || ~δ(N, N′)
    else
        overlap += 1.0
    end
    return overlap
end