Base.@kwdef struct HarmonicOscillatorState <: BasisState
    E = 0.0
    m::Float64
    ω::Float64
    n::Int64
    constraints = ()
end
export HarmonicOscillatorState

function Identity(state::HarmonicOscillatorState, state′::HarmonicOscillatorState)
    return δ(state.n, state′.n)
end
export Identity

function T(state::HarmonicOscillatorState, state′::HarmonicOscillatorState)
    n, m, ω = state.n, state.m, state.ω
    return I(state, state′) * ω * (n + 1/2)
end
export T
