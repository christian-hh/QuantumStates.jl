# import WignerSymbols_Simple: wigner3j, wigner6j
import WignerSymbols
import LinearAlgebra: norm

using CompositeStructs
using DataFrames
using PartialWaveFunctions

using HalfIntegers
using NamedTupleTools
# using LinearAlgebra
using LoopVectorization
using NamedTupleTools

import ProgressMeter: Progress, next!, update!

mutable struct State{T<:BasisState}
    E::Float64
    basis::Vector{T}
    coeffs::Vector{ComplexF64}
    idx::Int64
end
export State

⊕(f1::T1, f2::T2) where {T1,T2} = (bs, bs′) -> f1(bs.basis_state1, bs′.basis_state1) + f2(bs.basis_state2, bs′.basis_state2)
export ⊕

⊗(f1::T1, f2::T2) where {T1,T2} = (bs, bs′) -> f1(bs.basis_state1, bs′.basis_state1) * f2(bs.basis_state2, bs′.basis_state2)
export ⊗

function expectation(state::State, s::Symbol)
    exp = zero(ComplexF64)
    for i ∈ eachindex(state.basis)
        exp += getfield(state.basis[i], s) * state.coeffs[i] * conj(state.coeffs[i])
    end
    return exp
end
export expectation 

energy(s::State) = s.E
export energy

"""
    Returns a vector of states based on a basis.
"""
function states_from_basis(basis::Vector{<:BasisState})
    states = [State(0.0, basis, zeros(ComplexF64, length(basis)), i) for i in 1:length(basis)]
    for i in eachindex(basis)
        states[i].coeffs[i] = 1.0
    end
    return states
end
export states_from_basis

function states_to_basis(states::Vector{State{T}}) where T
    basis = states[1].basis
    basis_states_include = zeros(Bool, length(basis))
    for i ∈ eachindex(states)
        state = states[i]
        normed_state_coeffs = norm.(state.coeffs)
        nonzero_idxs = findall(x -> x > 0, normed_state_coeffs)
        basis_states_include[nonzero_idxs] .|= true
    end
    return findall(>(false), basis_states_include), basis[basis_states_include]
end
export states_to_basis

import LinearAlgebra: dot
dot(state::State, state′::State) = state.coeffs ⋅ state′.coeffs
export ⋅

norm(state::State) = norm(state.coeffs)
export norm

function order(basis::Vector{<:BasisState}, ordering_QN)
    all_QN = [getfield(state, ordering_QN) for state in basis]
    idxs_sorted = sortperm(all_QN)
    return basis[idxs_sorted]
end
export order

function is_good_quantum_number(state::State, QN::Symbol)
    vals = Rational[]
    for (i, coeff) in enumerate(state.coeffs)
        val = getfield(state.basis[i], QN)
        if norm(coeff) > 1e-5
            push!(vals, val)
        end
    end
    unique_vals = unique(vals)
    return (length(unique_vals) == 1, unique_vals)
end

# function minimize_basis(states::Vector{State})
#     basis_idxs = zeros(Bool, length(states[1].basis))
#     for state in states
#         for (i, coeff) in enumerate(state.coeffs)
#             if norm(coeff)^2 > 1e-5
#                 basis_idxs[i] = true
#             end
#         end
#     end
#     return

function states_table(states::Vector{<:State}; threshold=1e-3, dominant_state_only=false)
    QNs = fieldnames(typeof(states[1].basis[1]))[3:end-1]
    # QN_types = [typeof(getfield(states[1].basis[1], QN)) for QN ∈ QNs]
    # QN_columns = NamedTuple(QN => QN_types[i][] for (i,QN) ∈ enumerate(QNs))
    QN_columns = NamedTuple(QN => Rational{Int64}[] for QN in QNs)
    all_columns = merge((State = Integer[], c = Float64[],), QN_columns, (E = Float64[],))
    df = DataFrame(all_columns)

    vals = zeros(Float64, length(QNs)+3)
    for (i, state) ∈ enumerate(states)
        basis_states, coeffs = contributing_basis_states(state, threshold)
        vals[1] = state.idx
        vals[end] = state.E
        
        cs = norm.(coeffs).^2
        if dominant_state_only
            max_c, dominant_state_idx = findmax(cs)
            basis_state = basis_states[dominant_state_idx]
            vals[2] = max_c
            for (i, QN) ∈ enumerate(QNs)
                vals[i+2] = getfield(basis_state, QN)
            end
            push!(df, vals)
        else
            for (basis_state, coeff) ∈ zip(basis_states, coeffs)
                c = norm(coeff)^2
                vals[2] = c
                for (i, QN) ∈ enumerate(QNs)
                    vals[i+2] = getfield(basis_state, QN)
                end
                push!(df, vals)
            end
        end
    end
    return df
end

states_table(states::Vector{<:State}, relabelling_states::Vector{<:State}; threshold=1e-3, dominant_state_only=false) =
    states_table(relabelling_states, threshold=threshold, dominant_state_only=dominant_state_only)
export states_table
        
# Extend the base "+" function such that we can simply add matrix elements before calling them
import Base: +, -, *, /, ^

+(f::Function, g::Function) =
    (args...; kwargs...) -> f(args...; kwargs...) + g(args...; kwargs...)
-(f::Function, g::Function) =
    (args...; kwargs...) -> f(args...; kwargs...) - g(args...; kwargs...)
*(f::Function, g::Function) =
    (args...; kwargs...) -> f(args...; kwargs...) * g(args...; kwargs...)
*(c::Number, f::Function) =
    (args...; kwargs...) -> c * f(args...; kwargs...)
/(f::Function, c::Number) =
    (args...; kwargs...) -> f(args...; kwargs...) / c
^(f::Function, c::Real) = 
    (args...; kwargs...) -> f(args...; kwargs...)^c
# +(f::Function, c::Real) =
#     (args...; kwargs...) -> f(args...; kwargs...) + c
#     +(a, b) = +(b, a)`

*(c, state::State) = State(0., state.basis, c .* state.coeffs, 0)

+(state::State, state′::State) = State(0., state.basis, state.coeffs .+ state′.coeffs, 0)

δ(x,y) = ==(x,y)
export δ

struct ParameterList
    param_dict::Dict{Symbol, ComplexF64}
end
export ParameterList

mutable struct Operator{F}
    param::Symbol
    operator::F
    matrix::Matrix{ComplexF64}
end
export Operator

function create_block_diagonal_matrix(m1, m2)
    n = size(m1, 1)
    m = size(m2, 1)
    block_m = zeros(ComplexF64, n + m, n + m)
    block_m[1:n, 1:n] .= m1
    block_m[(n+1):(n+m), (n+1):(n+m)] .= m2
    return block_m
end
export create_block_diagonal_matrix

function create_block_diagonal_matrix(ms)
    ns = (size(m,1) for m ∈ ms) # sizes of matrices
    block_matrix = zeros(ComplexF64, sum(ns), sum(ns))
    idx1 = 1
    idx2 = 0
    for (n,m) ∈ zip(ns,ms)
        idx2 += n
        block_matrix[idx1:idx2,idx1:idx2] .= m
        idx1 += n
    end
    return block_matrix
end
export create_block_diagonal_matrix

function extend_basis(states, basis′)
    states′ = deepcopy(states)
    extended_basis = unique([states′[1].basis; basis′])
    for state′ ∈ states′
        state′.basis = extended_basis
        state′.coeffs = [state′.coeffs; zeros(length(basis′))]
    end
    return states′
end
export extend_basis

function get_basis_tdms(basis::Vector{<:BasisState}, tdm_func::F) where F
    basis_tdms = zeros(ComplexF64, length(basis), length(basis), 3)
    for (i, bstate) ∈ enumerate(basis)
        for (j, bstate′) ∈ enumerate(basis)
            for p ∈ -1:1
                if i < j
                    basis_tdms[i,j,p+2] = tdm_func(bstate, bstate′, p)
                    basis_tdms[j,i,p+2] = conj(basis_tdms[i,j,p+2])
                elseif i == j
                    basis_tdms[i,j,p+2] = tdm_func(bstate, bstate′, p)
                end
            end
        end
    end
    return basis_tdms
end
export get_basis_tdms

function get_tdms_two_bases(basis::Vector{<:BasisState}, basis′::Vector{<:BasisState}, tdm_func::F) where F
    basis_tdms = zeros(ComplexF64, length(basis), length(basis′), 3)
    for (i, bstate) ∈ enumerate(basis)
        for (j, bstate′) ∈ enumerate(basis′)
            for p ∈ -1:1
                basis_tdms[i,j,p+2] = tdm_func(bstate, bstate′, p)
            end
        end
    end
    return basis_tdms
end
export get_tdms_two_bases

"""
    `states` and `states′` must share the same basis.
"""
function tdms_between_states!(d::Array{T, 3}, basis_tdms::Array{T, 3}, states::Vector{<:State}, states′::Vector{<:State}) where T
    for i ∈ eachindex(states), j ∈ eachindex(states′)
        state  = states[i]
        state′ = states′[j]
        basis  = state.basis
        basis′ = state′.basis
        for p ∈ -1:1
            d[i,j,p+2] = 0.0
            for m ∈ eachindex(basis), n ∈ eachindex(basis′)
                d[i,j,p+2] += conj(state.coeffs[m]) * state′.coeffs[n] * basis_tdms[m,n,p+2]
            end
        end
    end
    return nothing
end
export tdms_between_states!

function tdms_between_states(states::Vector{<:State}, states′::Vector{<:State})
    d = zeros(ComplexF64, length(states), length(states′), 3)
    for i ∈ eachindex(states), j ∈ eachindex(states′)
        state  = states[i]
        state′ = states′[j]
        basis  = state.basis
        basis′ = state′.basis
        for p ∈ -1:1
            d[i,j,p+2] = 0.0
            for m ∈ eachindex(basis), n ∈ eachindex(basis′)
                if norm(state.coeffs[m]) > 1e-6 && norm(state′.coeffs[n]) > 1e-6
                    d[i,j,p+2] += conj(state.coeffs[m]) * state′.coeffs[n] * TDM(basis[m], basis′[n], p)
                end
            end
        end
    end
    return d
end
export tdms_between_states

macro params(arg)
    dict = Dict{Symbol, ComplexF64}()
    for x in arg.args
        if x isa Expr
            param_name = x.args[1]
            param_value = x.args[2]
            dict[param_name] = eval(param_value)
        end
    end
    return ParameterList(dict)
end
export @params

using MutableNamedTuples

macro params_mnt(ex)
    Expr(:call, :MutableNamedTuple, (:($(Expr(:kw, Symbol(arg.args[1]), arg.args[2]))) for arg in ex.args if arg isa Expr)...)
end
export @params_mnt

import Base: getproperty
function getproperty(p::ParameterList, s::Symbol)
    param_dict = getfield(p, :param_dict)
    if s ∈ keys(param_dict)
        return getindex(param_dict, s)
    else
        return getfield(p, s)
    end
end
export getproperty

import Base: setproperty!
function setproperty!(p::ParameterList, s::Symbol, v)
    setindex!(getfield(p, :param_dict), v, s)
end
export setproperty!

macro make_operator(basis, operator_expr)
    return quote
        operators = NamedTuple()
        basis = $(esc(basis))
        for expr ∈ $(operator_expr.args)
            if expr isa Expr
                param = expr.args[2]
                println(param)
                operator = $(esc(operator_expr.args[2].args[3]))
                matrix = matrix_from_operator(basis, operator)
                Operator(param, operator, matrix)
                if param isa Symbol
                    operators = (param => 1,)
                end
            end
        end
        operators
    end
end
export make_operator

function unpack_operator(operator::Expr, basis::Vector{<:BasisState})
    operators = NamedTuple()
    if length(operator.args) > 0
        if operator.args[1] != :+ # case if only one term is included in the Hamiltonian
            expr = operator
            param = expr.args[2]
            operator = eval(expr.args[3])
            matrix = matrix_from_operator(basis, operator)
            if param isa Symbol
                operators = (; operators..., param => Operator(param, operator, matrix))
            end
        else
            for expr ∈ operator.args
                if expr isa Expr
                    param = expr.args[2]
                    operator = eval(expr.args[3])
                    matrix = matrix_from_operator(basis, operator)
                    if param isa Symbol
                        operators = (; operators..., param => Operator(param, operator, matrix))
                    end
                end
            end
        end
    end
    return operators
end
export unpack_operator

function matrix_from_operator(basis, operator::F) where {F}
    m = zeros(ComplexF64, length(basis), length(basis))
    for i ∈ eachindex(basis), j ∈ eachindex(basis)
        m[i,j] = operator(basis[i], basis[j])
    end
    return m
end

# function evaluate_operator!(basis, o::Operator{F}) where F
#     for i ∈ eachindex(basis), j ∈ i:length(basis)
#         o.matrix[i,j] = o.operator(basis[i], basis[j])
#         o.matrix[j,i] = conj(o.matrix[i,j])
#     end
#     return nothing
# end
function evaluate_operator!(basis, o::Operator{F}) where F
    for i ∈ eachindex(basis), j ∈ eachindex(basis)
        o.matrix[i,j] = o.operator(basis[i], basis[j])
    end
    return nothing
end
# export evaluate_operator!

function findindex(QNs, QN::Symbol)
    exists = false
    i = 1
    for name in QNs
        if name == QN
            exists = true
            break
        else
            i += 1
        end
    end
    return (exists, i)
end

# import Base.==
# function ==(state::State, state′::State)
#     state_type = typeof(state)
#     for field in fieldnames(state_type)
#         if getfield(state, field) != getfield(state′, field)
#             return false
#         end
#     end
#     return true
# end
# export ==

function DiagonalOperator(state::BasisState, state′::BasisState)
    if state == state′
        return 1.0
    else
        return 0.0
    end
end
export DiagonalOperator

import LinearAlgebra: ⋅
function calculate_state_overlaps!(states, states′, overlaps)
    for i ∈ eachindex(states)
        for j ∈ eachindex(states′)
            state = states[i]
            state′ = states′[j]
            overlaps[i,j] = norm(state ⋅ state′)
        end
    end
    return nothing
end
export calculate_state_overlaps!

function calculate_state_overlaps(states, states′)
    overlaps = zeros(Float64, length(states), length(states))
    calculate_state_overlaps!(states, states′, overlaps)
    return overlaps
end
export calculate_state_overlaps

function tracked_idxs!(overlaps, states, prev_states, tracked_idxs)
    for i ∈ axes(overlaps, 2)
        overlaps_row = @view overlaps[:,i] # overlap for previous state `i` with all states of current iteration 
        max_idx = argmax(overlaps_row)
        tracked_idxs[i] = max_idx
        overlaps[max_idx,:] .= 0.0
        # found_idx = false
        # j = 1
        # while !found_idx && j <= size(overlaps,1)
        #     max_idx = argmax(overlaps_row)
        #     # println(abs(energy(prev_states[i]) - energy(states[tracked_idxs[i]])), " ", abs(energy(prev_states[i]) - energy(states[max_idx])))
        #     # println(energy(prev_states[i]) - energy(states[tracked_idxs[i]]))
        #     # println(energy(prev_states[i]), " ", energy(prev_states[i]) - energy(states[max_idx]), " ", j)
        #     # println(" ")
        #     # Don't allow the tracked index to change if the change is energy is too large
        #     # (This is meant to avoid discrete jumps between states far from each other that may look similar close to an avoided crossing.)
        #     if abs(energy(prev_states[i]) - energy(states[max_idx])) < (10 * abs(energy(prev_states[i]) - energy(states[i])))
        #         tracked_idxs[i] = max_idx
        #         overlaps[max_idx,:] .= 0.0
        #         found_idx = true
        #     else
        #         overlaps[max_idx,i] = 0.0
        #     end
        #     j += 1
        # end
        # println(abs(energy(prev_states[i]) - energy(states[i])), " ", abs(energy(prev_states[i]) - energy(states[tracked_idxs[i]])))
        # if !found_idx
            # println(found_idx, i)
        # end
    end
    return nothing
end
export tracked_idxs!

function subspace(states::Vector{State{T}}, QN_bounds, threshold=0.01) where {T<:BasisState}
    subspace = State{T}[]
    subspace_idxs = Int64[]
    QNs = keys(QN_bounds)
    add_to_subspace = ones(Bool, length(states))
    for QN ∈ QNs
        for (i, state) ∈ enumerate(states)
            for (j, coeff) ∈ enumerate(state.coeffs)
                # println(getfield(state.basis[j], QN))
                # println(QN_bounds[QN])
                if getfield(state.basis[j], QN) ∉ QN_bounds[QN]
                    if norm(coeff)^2 > threshold
                        add_to_subspace[i] = false
                    end
                end
            end
        end
    end
    for i ∈ eachindex(states)
        if add_to_subspace[i]
            push!(subspace, states[i])
            push!(subspace_idxs, i)
        end
    end
    return (subspace_idxs, subspace)
end

function subspace(basis::Vector{<:BasisState}, QN_bounds)
    subspace_basis = typeof(basis[1])[]
    subspace_idxs = Int64[]
    QNs = keys(QN_bounds)
    add_to_subspace = ones(Bool, length(basis))
    for QN ∈ QNs
        for (i, b_state) ∈ enumerate(basis)
            if getfield(b_state, QN) ∉ QN_bounds[QN]
                add_to_subspace[i] = false
            end
        end
    end
    for i ∈ eachindex(basis)
        if add_to_subspace[i]
            push!(subspace_basis, basis[i])
            push!(subspace_idxs, i)
        end
    end
    return subspace_idxs, subspace_basis
end
export subspace

function extend_operator(operator::T, state::State, state′::State, args...) where {T}
    val = zero(ComplexF64)
    for (i, basis_state) in enumerate(state.basis)
        for (j, basis_state′) in enumerate(state′.basis)
            val += conj(state.coeffs[i]) * state′.coeffs[j] * operator(basis_state, basis_state′, args...)
        end
    end
    return val
end
export extend_operator

function TDM(state::State, state′::State, args...)
    tdm = zero(ComplexF64)
    for (i, basis_state) in enumerate(state.basis)
        for (j, basis_state′) in enumerate(state′.basis)
            tdm += conj(state.coeffs[i]) * state′.coeffs[j] * TDM(basis_state, basis_state′, args...)
            # coeff1 = state.coeffs[i]
            # coeff2 = state′.coeffs[i]
            # if (norm(coeff1)^2 > 1e-5) && (norm(coeff2)^2 > 1e-5)
                # tdm += state.coeffs[i] * state′.coeffs[j] * TDM(basis_state, basis_state′, args...)
            # end
        end
    end
    return tdm
end
export TDM

function convert_basis(states::Vector{State{T}}, basis′) where {T}

    new_states = State{typeof(basis′[1])}[]
    basis = states[1].basis

    # Calculate change-of-basis matrix
    P = zeros(Float64, length(basis′), length(basis))
    for (i, state) in enumerate(basis′)
        for (j, state′) in enumerate(basis)
            P[i,j] = overlap(state, state′)
        end
    end

    # for state ∈ states
    #     new_state = State(state.E, basis′, P * state.coeffs, state.idx)
    #     push!(new_states, new_state)
    # end
    for state ∈ states
        new_state = State(state.E, basis′, P * state.coeffs, state.idx)
        push!(new_states, new_state)
    end


#     # Check that the basis provided included all necessary states to make the conversion
#     for state in new_states
#         if norm(state.coeffs) 
    
    return new_states
end