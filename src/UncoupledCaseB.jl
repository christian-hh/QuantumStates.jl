# abstract type HundsCaseB_Decoupled <: BasisState end
# export HundsCaseB_Decoupled

@composite Base.@kwdef struct HundsCaseB_LinearMolecule_Decoupled <: BasisState
    E::Float64 = 0.0
    v_1::HalfInt = 0
    v_2::HalfInt = 0
    v_3::HalfInt = 0
    S::HalfInt = 0
    I::HalfInt = 0
    Λ::HalfInt = 0
    ℓ::HalfInt = 0
    K::HalfInt = 0
    N::HalfInt = 0
    M_N::HalfInt
    M_S::HalfInt
    M_I::HalfInt
    constraints = (
        K = Λ + ℓ,
        N = abs(K):∞,
        M_N = -N:N,
        M_S = -S:S,
        M_I = -I:I
    )
end
export HundsCaseB_LinearMolecule_Decoupled

function unpack(state::HundsCaseB_LinearMolecule_Decoupled)
    (; v_1, v_2, v_3, S, I, Λ, ℓ, K, N, M_N, M_S, M_I) = state
    return v_1, v_2, v_3, S, I, Λ, ℓ, K, N, M_N, M_S, M_I
end
export unpack

function overlap(state::HundsCaseB_LinearMolecule, state′::HundsCaseB_LinearMolecule_Decoupled)
    """
    See eq. (5.187) in Brown & Carrington
    """
    v_1,  v_2,  v_3,  S,  I,  Λ,  ℓ,  K,  N,  J,  F,  M_F  = unpack(state)
    v_1′, v_2′, v_3′, S′, I′, Λ′, ℓ′, K′, N′, M_N′, M_S′, M_I′ = unpack(state′)
    if ~delta(state, state′, :v_1, :v_2, :v_3, :Λ, :ℓ, :N)
        return 0.0
    else
        return sum(
                  (-1)^(J - I + M_F) * sqrt(2F + 1) * wigner3j(J, I, F, M_J, M_I′, -M_F)
                * (-1)^(N - S + M_J) * sqrt(2J + 1) * wigner3j(N, S, J, M_N′, M_S′, -M_J)
                for M_J ∈ -J:J
            )
    end
end
overlap(state::HundsCaseB_LinearMolecule_Decoupled, state′::HundsCaseB_LinearMolecule) = overlap(state′, state)
export overlap