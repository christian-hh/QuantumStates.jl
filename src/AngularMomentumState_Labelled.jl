using Parameters
using InfiniteArrays
using CompositeStructs
using HalfIntegers

Base.@kwdef mutable struct AngularMomentumState_Labelled <: BasisState
    E::Float64 = 0.0
    L::Float64 = 0.0
    N::HalfInt = 0.0
    M::HalfInt = 0.0
    constraints = (
        M = -N:N,
    )
end
export AngularMomentumState_Labelled

function unpack(state::AngularMomentumState_Labelled)
    return (state.L, state.N, state.M)
end
export unpack

function L(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    return δ(L, L′) * δ(N, N′) * δ(M, M′) * L
end

function Identity(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    if ~δ(L, L′) || ~δ(N, N′) || ~δ(M, M′)
        return 0.0
    else
        return 1.0
    end
end
export I

function Rotation(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    return I(state, state′) * N * (N + 1)
end
export Rotation

function TDM(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled, p::Int64)
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    if L′ <= L || (L < 1 && L′ < 1) || (L > 1 && L′ > 1) 
    # if (L′ == 0 && L == 2) || (L == 0 && L′ == 2) || (L == 0 && L′ == 0) || (L == 2 && L′ == 2)
        return 0.0
    else
        return (
            (-1)^p * (-1)^(N - M) * wigner3j(N, 1, N′, -M, -p, M′) * sqrt(2N′ + 1)
            # (-1)^(N - M) * wigner3j(N, 1, N′, -M, -p, M′) * sqrt(2N′ + 1)
        )
    end
end

function TDM_magnetic(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled, p::Int64)
    # Assumes magnetic moment aligned along z-axis of molecule-fixed axis
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    return -(
        (-1)^p * (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, M, p, -M′)
    )
end
# TDM_magnetic(state::State, state′::State, p::Int64) = extend_operator(TDM_magnetic, state, state′, p)

function d₀(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    (-1)^M * sqrt((2N+1)*(2N′+1)) * wigner3j(N, 1, N′, -M, 0, M′) * wigner3j(N, 1, N′, 0, 0, 0)
end

function Zeeman_L0(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
    L,  N,  M  = unpack(state) 
    L′, N′, M′ = unpack(state′)
    if ~δ(L, L′) || ~δ(N, N′) || L >= 1
        return 0.0
    else
        return (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, 0, M′)
    end
end

excited(state, state′) = (state.L == 1) && (state′.L == 1) && (state.M == state′.M)
# identity1(state, state′) = (state.N == state′.N == 1) * (state.L == state′.L == 2) && (state.M == state′.M)

# function Zeeman_L2(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
#     L,  N,  M  = unpack(state) 
#     L′, N′, M′ = unpack(state′)
#     if ~δ(L, L′) || ~δ(N, N′) || ~δ(L, 2)
#         return 0.0
#     else
#         return (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, 0, M′)
#     end
# end

# function Zeeman_L0_N2(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
#     L,  N,  M  = unpack(state) 
#     L′, N′, M′ = unpack(state′)
#     if ~δ(L, L′) || ~δ(N, N′) || ~δ(L, 0) || ~δ(N, 2)
#         return 0.0
#     else
#         return (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, 0, M′)
#     end
# end

# function Zeeman_L0_N1(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
#     L,  N,  M  = unpack(state) 
#     L′, N′, M′ = unpack(state′)
#     if ~δ(L, L′) || ~δ(N, N′) || ~δ(L, 0) || ~δ(N, 1)
#         return 0.0
#     else
#         return -(-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, 0, M′)
#     end
# end

function Zeeman_L0(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled, p::Int64)
    L,  N,  M  = unpack(state) 
    L′, N′, M′ = unpack(state′)
    if ~δ(L, L′) || ~δ(N, N′) || L >= 1
        return 0.0
    else
        return (-1)^p * (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, -p, M′)
        # return (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, -p, M′)
    end
end

export Zeeman_L0

function Zeeman_Lx(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled, p::Int64, x::Float64)
    L,  N,  M  = unpack(state) 
    L′, N′, M′ = unpack(state′)
    if ~δ(L, L′) || ~δ(N, N′) || ~δ(L,x)
        return 0.0
    else
        return (-1)^p * (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, -p, M′)
        # return (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, -p, M′)
    end
end

export Zeeman_Lx

function Zeeman_L1(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled)
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    if ~δ(L, L′) || ~δ(N, N′) || L < 1
        return 0.0
    else
        return (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, 0, M′)
    end
end

function Zeeman_L1(state::AngularMomentumState_Labelled, state′::AngularMomentumState_Labelled, p::Int64)
    L,  N,  M  = unpack(state)
    L′, N′, M′ = unpack(state′)
    if ~δ(L, L′) || ~δ(N, N′) || L < 1
        return 0.0
    else
        return (-1)^p * (-1)^(N - M) * sqrt(N * (N + 1) * (2N + 1)) * wigner3j(N, 1, N′, -M, -p, M′)
    end
end

export Zeeman_L1

# function H(state::AngularMomentumState, state′::AngularMomentumState)
#     δ(state, state′) * state.E
# end
# export H