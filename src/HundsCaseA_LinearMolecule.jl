abstract type HundsCaseA <: BasisState end
export HundsCaseA

Base.@kwdef struct HundsCaseA_LinearMolecule <: HundsCaseA
    E::Float64 = 0.0
    label=""
    v_1::HalfInt = 0
    v_2::HalfInt = 0
    ℓ::HalfInt = 0
    v_3::HalfInt = 0
    Λ::HalfInt = 0
    K::HalfInt = 0 # K = Λ + ℓ
    I::HalfInt = 0
    S::HalfInt = 0
    Σ::HalfInt = 0
    J::HalfInt = 0
    P::HalfInt = 0 # P = Λ + ℓ + Σ
    F::HalfInt = 0
    M::HalfInt = 0
    constraints = (
        K = Λ + ℓ,
        Σ = -S:S,
        # ℓ = unique(sign * (x + v2 % 2) for x ∈ 0:2:v_2 for sign ∈ (-1,1)),
        P = max(-J, Λ + ℓ + Σ):min(J, Λ + ℓ + Σ),
        F = abs(J - I):abs(J + I),
        M = -F:F
    )
end
export HundsCaseA_LinearMolecule

function T(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    return state.E * (state == state′)
end
export T

function unpack(state::HundsCaseA_LinearMolecule)
    (; v_1, v_2, ℓ, v_3, Λ, K, I, S, Σ, J, P, F, M) = state
    return v_1, v_2, ℓ, v_3, Λ, K, I, S, Σ, J, P, F, M
end
export unpack

function Rotation(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    term1 = (J * (J + 1) + S * (S + 1) - 2 * P * Σ - K^2) * δ(Σ,Σ′) * δ(P,P′) # diagonal term
    term2 = ( # spin uncoupling
        - 2 * (-1)^(J - P + S - Σ) * 
            sqrt(J * (J + 1) * (2J + 1) * S * (S + 1) * (2S + 1)) * 
            sum(
                wigner3j(J, 1, J, -P, q, P′) *
                wigner3j(S, 1, S, -Σ, q, Σ′)
                for q ∈ (-1,1)
            )
    )
    return (term1 + term2) * δ(ℓ,ℓ′) * δ(Λ,Λ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
end
export Rotation

function Rotation_Σ(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return Rotation(state, state′) * δ(abs(Λ+ℓ), 0)
end
export Rotation_Σ

function Rotation_Δ(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return Rotation(state, state′) * δ(abs(Λ+ℓ), 2)
end
export Rotation_Δ

function SpinOrbit(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return Λ * Σ * 
        δ(ℓ,ℓ′) * δ(Λ,Λ′) * δ(Σ,Σ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
end
export SpinOrbit

function SpinUncoupling(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    # Brown & Carrington, eq. (8.364)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return -2 * (-1)^(J - P + S - Σ) * 
        sqrt( J * (J + 1) * (2J + 1) * S * (S + 1) * (2S + 1) ) *
        sum(
            wigner3j_(J, 1, J, -P, q, P′) * 
            wigner3j_(S, 1, S, -Σ, q, Σ′)
            for q ∈ (-1,1)
        ) *
        δ(Λ,Λ′) * δ(ℓ,ℓ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
end
export SpinUncoupling

function ΛDoubling_q(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    """
    (Λ_+^2 J_-^2) term
    See Li & Coxon (1995)
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return (
        (-1)^(J - P) *
        (1 / (2 * sqrt(6))) *
        sqrt( (2J - 1) * (2J) * (2J + 1) * (2J + 2) * (2J + 3) ) *
        sum(
            δ(Λ′, Λ + 2q) *
            wigner3j_(J, 2, J′, -P, -2q, P′)
            for q ∈ (-1,1)
        )
    ) * δ(ℓ,ℓ′) * δ(Σ,Σ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
end
export ΛDoubling_q

function ΛDoubling_p2q(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    """
    (Λ⁺ Λ⁺ J⁻ S⁻) term
    Brown and Carrington (eq. 9.66)
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return (-1)^(J - P) * (-1)^(S - Σ) * 
        sqrt( J * (J + 1) * (2J + 1) * S * (S + 1) * (2S + 1) ) *
        sum(
            δ(Λ′, Λ + 2q) *
            wigner3j_(J, 1, J′, -P, -q, P′) *
            wigner3j_(S, 1, S, -Σ, q, Σ′)
            for q ∈ (-1,1)
        ) *
        δ(ℓ,ℓ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
    end
export ΛDoubling_p2q

function ℓDoubling(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    """
    H_ℓ = (1/2)q^v[(1/2)(G₊²J₋² + G₋²J₊²) - (G₊²J₋S₋ + G₋²J₊S₊)]
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    term1 = (
        (-1)^(J - P) *
        (1 / (2 * sqrt(6))) *
        sqrt( (2J - 1) * (2J) * (2J + 1) * (2J + 2) * (2J + 3) ) *
        sum(
            δ(ℓ′, ℓ + 2q) *
            wigner3j_(J, 2, J′, -P, -2q, P′)
            for q ∈ (-1,1)
        )
    ) * δ(Σ,Σ′)
    # Add factors for G₊²???
    term2 = (
        (-1)^(J - P + S - Σ) * 
        sqrt( J * (J + 1) * (2J + 1) * S * (S + 1) * (2S + 1) ) *
        sum(
            δ(ℓ′, ℓ + 2q) *
            wigner3j_(J, 1, J′, -P, -q, P′) *
            wigner3j_(S, 1, S′, -Σ, q, Σ′)
            for q ∈ (-1,1)
        )
    )
    return (term1 - term2) * δ(Λ,Λ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
end

function gK_nonadiabatic(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    """
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    if ~delta(state, state′, :ℓ, :Λ, :K, :Σ, :J, :P, :F, :M)
        return 0.0
    else
        return K * Λ
    end
end
        
function Hyperfine_IL(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    # Brown and Carrington, eq. (8.372)
    # Hirota, eq. (2.3.66)
    # Orbital hyperfine interaction
    # Only the term diagonal in q is retained here
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    if ~delta(state, state′, :ℓ, :Σ, :Λ, :K, :Σ, :P, :F, :M)
        return 0.0
    else
        return (
            Λ′ * (-1)^(I + J + F′) * (-1)^(J - P) * 
            sqrt(I * (I + 1) * (2I + 1) * (2J + 1) * (2J′ + 1)) *
            wigner6j(I, J, F′, J′, I, 1) * 
            wigner3j(J, 1, J′, -P, 0, P′)
        )
    end
end
export Hyperfine_IL

function Hyperfine_IF(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    # Fermi contact interaction
    # Hirota, eq. (2.3.67)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    if ~delta(state, state′, :ℓ, :F, :M)
        return 0.0
    else
        return (
            (-1)^(I + J + F′) * (-1)^(S - Σ) * (-1)^(J - P) * 
            sqrt(I * (I + 1) * (2I + 1) * (2J + 1) * (2J′ + 1) * S * (S + 1) * (2S + 1)) *
            wigner6j(I, J, F′, J′, I, 1) *
            sum(
                wigner3j(J, 1, J′, -P, q, P′) *
                wigner3j(S, 1, S, -Σ, q, Σ′)
                for q ∈ -1:1
            )
        )
    end
end
export Hyperfine_IF
    
function Hyperfine_Dipolar_c(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return sqrt(30) * (1/3) * (-1)^(I + J′ + F) * (-1)^(J - P) * (-1)^(S - Σ) *
        wigner6j_(I, J′, F, J, I, 1) *
        sqrt( I * (I + 1) * (2I + 1) ) *
        sqrt( (2J + 1) * (2J′ + 1) ) *
        sqrt( S * (S + 1) * (2S + 1) ) *
        sum(
            (-1)^q * 
            wigner3j_(J, 1, J′, -P, q, P′) *
            sum(
                wigner3j_(1, 2, 1, q′, 0, -q) *
                wigner3j_(S, 1, S, -Σ, q′, Σ′)
                for q′ ∈ -1:1
            ) for q ∈ -1:1
        ) * 
        δ(F, F′) * δ(M, M′)
end
export Hyperfine_Dipolar_c

function Hyperfine_Dipolar_d(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return sqrt(30) * (1/2) * sqrt(3/2) * (2/3) * (-1)^(I + J′ + F) * (-1)^(J - P) * (-1)^(S - Σ) *
        wigner6j_(I, J′, F, J, I, 1) *
        sqrt( I * (I + 1) * (2I + 1) ) *
        sqrt( (2J + 1) * (2J′ + 1) ) *
        sqrt( S * (S + 1) * (2S + 1) ) *
        sum(
            (-1)^q * 
            wigner3j_(J, 1, J′, -P, q, P) *
            sum(
                (
                    wigner3j_(1, 2, 1, q′, 2, -q) +
                    wigner3j_(1, 2, 1, q′, -2, -q)
                ) *
                wigner3j_(S, 1, S, -Σ, q′, Σ′)
                for q′ ∈ -1:1
            ) for q ∈ -1:1
        ) * 
        δ(F, F′) * δ(M, M′)
end
export Hyperfine_Dipolar_d

function RennerTeller(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule)
    """
    See Coxon & Li (1995).
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    # return (
    #     sum(((v_2 + 1)^2 - K^2)^(1/2) * δ(Λ,Λ′+2q) * δ(ℓ,ℓ′-2q) for q ∈ (-1,1))
    # ) *
    # δ(Σ,Σ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
    return (1/2) * (
        sum(((v_2 + 1)^2 - K^2)^(1/2) * δ(Λ,Λ′+2q) * δ(ℓ,ℓ′-2q) for q ∈ (-1,1))
    ) *
    δ(Σ,Σ′) * δ(J,J′) * δ(F,F′) * δ(M,M′)
end
export RennerTeller

function TDM(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule, p::Integer)
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)
    return (-1)^(F′ - M′) * wigner3j(F′, 1, F, -M′, p, M) * (-1)^(F + J′ + I + 1) * sqrt((2F′ + 1) * (2F + 1)) *
        δ(Σ, Σ′) * (-1)^(J′ - P′) * wigner6j(J, F, I, F′, J′, 1) * sqrt((2J′ + 1) * (2J + 1)) *
        sum(
            wigner3j(J′, 1, J, -P′, q, P) for q ∈ -1:1
        )
end
# TDM(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule) = sum(TDM(state, state′, p) for p ∈ -1:1)
export TDM

function Zeeman_L(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule, p::Int64)
    """
    Electron orbital Zeeman interaction
    See Brown & Carrington, eq. (9.57)
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :ℓ, :Λ, :K, :I, :S, :Σ, :P)
        return 0.0
    else
        return (
            (-1)^p * Λ * (-1)^(F-M+F′+J+I+1+J-P) * wigner6j(J,F,I,F′,J′,1) * 
            wigner3j(F,1,F′,-M,p,M′) * sqrt((2F+1)*(2F′+1)*(2J+1)*(2J′+1)) * 
            wigner3j(J,1,J′,-P,0,P′)
        )
    end
end

function Zeeman_S(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule, p::Int64)
    """
    Electron spin Zeeman interaction
    See Brown & Carrington, eq. (9.58)
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :ℓ, :I, :S)
        return 0.0
    else
        return (
                sum(
                    (-1)^p * (-1)^(F-M+J+I+F′+1+J-P+S-Σ) * wigner6j(J,F,I,F′,J′,1) * 
                    wigner3j(F,1,F′,-M,p,M′) * sqrt((2F+1)*(2F′+1)) * 
                    wigner3j(J,1,J′,-P,q,P′) * sqrt((2J+1)*(2J′+1)) *
                    wigner3j(S,1,S,-Σ,q,Σ′) * sqrt(S*(S+1)*(2S+1))
                    for q ∈ -1:1
                )
            )
    end
end

function Zeeman_gl′(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule, p::Int64)
    """
    
    """
    v_1,  v_2,  ℓ,  v_3,  Λ,  K,  I,  S,  Σ,  J,  P,  F,  M  = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)

    if delta(state, state′)
        return 0.0
    else
        return (
            (-1)^p * (-1)^(F-M) * wigner3j_(F,1,F′,-M,p,M′) * (-1)^(F′+J+I+1) * sqrt((2F+1)*(2F′+1)) *
            wigner6j_(J′,F′,I,F,J,1) * 
            sum(
                δ(K′, K - 2q) * (-1)^(J-P+S-Σ) *
                (-1)^(J-P) * wigner3j_(J,1,J′,-P,q,P′) * sqrt((2J+1)*(2J′+1)) *
                (-1)^(S-Σ) * wigner3j_(S,1,S,-Σ,-q,Σ′) * sqrt(S*(S+1)*(2S+1))
                for q ∈ (-1,1)
            )
        )
    end
end
export Zeeman_gl′

# def ZeemanParityZ_even_aBJ(K0,Sigma0,P0,J0,F0,M0,K1,Sigma1,P1,J1,F1,M1,S=1/2,I=1/2):
#     if kronecker(K0,K1)*(not kronecker(M0,M1)):
#         return 0
#     else:
#         return (-1)*(-1)**(F0-M0)*wigner_3j(F0,1,F1,-M0,0,M1)*\
#             (-1)**(F1+J0+I+1)*np.sqrt((2*F0+1)*(2*F1+1))*wigner_6j(J1,F1,I,F0,J0,1)*\
#             np.sqrt((2*J0+1)*(2*J1+1)*S*(S+1)*(2*S+1))*\
#             sum([kronecker(K1,K0-2*q)*(-1)**(J0-P0+S-Sigma0)*wigner_3j(J0,1,J1,-P0,q,P1)*wigner_3j(S,1,S,-Sigma0,-q,Sigma1) for q in [-1,1]])