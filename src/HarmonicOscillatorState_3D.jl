Base.@kwdef struct HarmonicOscillatorState_3D <: BasisState
    E = 0.0
    m::Float64
    ωx::Float64
    ωy::Float64
    ωz::Float64
    nx::Int64
    ny::Int64
    nz::Int64
    constraints = ()
end
export HarmonicOscillatorState_3D

function Identity(state::HarmonicOscillatorState_3D, state′::HarmonicOscillatorState_3D)
    return (state.nx == state′.nx) * (state.ny == state′.ny) * (state.nz == state′.nz)
end
export Identity

function T(state::HarmonicOscillatorState_3D, state′::HarmonicOscillatorState_3D)
    nx, ny, nz = state.nx, state.ny, state.nz 
    ωx, ωy, ωz = state.ωx, state.ωy, state.ωz
    m = state.m
    return I(state, state′) * (ωx * (nx + 1/2) + ωy * (ny + 1/2) + ωz * (nz + 1/2))
end
export T
