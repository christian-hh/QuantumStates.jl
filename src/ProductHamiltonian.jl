@with_kw struct ProductHamiltonian{T1,T2}

    basis::Vector{ProductState{T1,T2}}
    H1::Hamiltonian{T1}
    H2::Hamiltonian{T2}

    states::Vector{State{ProductState{T1,T2}}} = states_from_basis(basis)
    matrix::Matrix{ComplexF64} = zeros(ComplexF64, length(H1.basis)*length(H2.basis), length(H1.basis)*length(H2.basis))

end
export ProductHamiltonian

import Kronecker: kronecker
import LinearAlgebra: I
function evaluate!(H::ProductHamiltonian)

    basis1, basis2 = H.H1.basis, H.H2.basis
    H1, H2 = H.H1.matrix, H.H2.matrix

    I1 = Matrix{ComplexF64}(I, length(basis1), length(basis1))
    I2 = Matrix{ComplexF64}(I, length(basis2), length(basis2))

    H.matrix .= kronecker(H1, I2) .+ kronecker(I1, H2)

    # idx1 = 1
    # for i ∈ eachindex(basis1)
    #     for j ∈ eachindex(basis2)
    #         idx2 = 1
    #         for k ∈ eachindex(basis1)
    #             for l ∈ eachindex(basis2)

    #                 # if idx1 == 2 && idx2 == 2
    #                 #     display((i,j))
    #                 #     display((k,l))
    #                 # end

    #                 H.matrix[idx1,idx2] = H1[i,j] * H2[k,l]
    #                 idx2 += 1
    #             end
    #         end
    #         idx1 += 1
    #     end
    # end

    return nothing

end


