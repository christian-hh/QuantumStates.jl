module QuantumStates

using UnitsToValue
using StaticArrays
using Parameters

include("WignerSymbols_Simple.jl")
include("Basis.jl")
include("States.jl")
include("Hamiltonian.jl")
include("Transitions.jl")
include("ProductState.jl")

# Various coupling schemes and other state definitions
include("HundsCaseA_LinearMolecule.jl")
include("HundsCaseB_LinearMolecule.jl")
include("UncoupledCaseB.jl")
include("AngularMomentumState.jl")
include("AngularMomentumState_Labelled_v1.jl")
include("AngularMomentumState_Labelled.jl")
include("AngularMomentumState_withSpinRotation.jl")
include("AngularMomentumState_withSpinRotation_Uncoupled.jl")
include("AsymmetricTopMolecule.jl")
include("HarmonicOscillatorState.jl")
include("HarmonicOscillatorState_3D.jl")
include("AtomicState.jl")
include("TriatomicVibrationalState.jl")

include("StateOverlaps.jl")

include("Printing.jl")
include("Operators.jl")
include("Saving.jl")

end
