using LaTeXStrings

function print_nice(state::State)
    for field in fieldnames(typeof(state))
        println(field, ": ", getfield(state, field))
    end
    return nothing
end
export print_nice

function format_float(value::Number)
    formatted_str = ""
    value_str = string(value)
    value_str = split(value_str, "e")
    if length(value_str) > 1 
        formatted_str *= string(round(parse(Float64, value_str[1]), sigdigits=3))
        formatted_str *= " \\times 10^{"
        formatted_str *= value_str[2]
        formatted_str *= "}"
    else
        formatted_str *= value_str[1]
    end
    return formatted_str
end
export format_float

function print_basis_state(basis_state::BasisState)
    str = "\\left|"
    fields = fieldnames(typeof(basis_state))
    for (i, field) in enumerate(fields)
        if 2 <= i < length(fields)
            val = getfield(basis_state, field)
            if typeof(val) != Float64
                val_str = string(val)
        #         if isinteger(val)  ### used when QNs were Rational rather than HalfInteger
        #             val_str = string(val)
        #         else
        #             val_str = "\\frac{$(val.num)}{$(val.den)}"
        #         end
                str *= string(field) * "=" * val_str
                if i < length(fields)-1
                    str *= string(", ")
                end
            end
        end
    end
    str *= "\\right\\rangle"
    return str
end

function contributing_basis_states(state::State, threshold=1e-3)
    basis_states = typeof(state.basis[1])[]
    coeffs = ComplexF64[]
    for (i, coeff) in enumerate(state.coeffs)
        if norm(coeff) > threshold
            push!(basis_states, state.basis[i])
            push!(coeffs, coeff)
        end
    end
    return basis_states, coeffs
end
export contributing_basis_states

function Base.show(io::IO, m::MIME"text/latex", state::State{T}) where T
    basis = state.basis
    coeffs = state.coeffs
    printed = false
    
    for (i, coeff) in enumerate(coeffs)
        str = "\$\$"
        basis_state = basis[i]
        if norm(coeff)^2 > 1e-2
            plus_sign = true
            state_str = ""
            real_val = string(round(real(coeff), digits=3))
            imag_val = string(round(imag(coeff), digits=3))
            if (abs(real(coeff)) > 1e-3) && (abs(imag(coeff)) < 1e-5)
                state_str *= real_val
                if real(coeff) < 0
                    plus_sign = false
                end
            elseif (abs(imag(coeff)) > 1e-3) && abs((real(coeff)) < 1e-5)
                state_str *= imag_val * "i"
                if imag(coeff) < 0
                    plus_sign = false
                end
            else
                state_str *= "(" * real_val * " + " * imag_val * "i" * ")"
            end
            if plus_sign && printed
                str *= " + "
            end
            str *= state_str
            str *= print_basis_state(basis_state)
            str *= "\$\$"
            latex_str = latexstring(str)
            println(io, latex_str)
            printed = true
        end
    end
    return nothing
end

function print_matrix(A)
    println("Real part:")
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            A_ij = real(A[i,j])
            if abs(A_ij) < 1e-5
                A_ij = convert(Int64, floor(A_ij))
            else
                A_ij = round(A_ij, digits=2)
            end
            print(A_ij, " ")
        end
        println()
    end
    println()
    println("Imaginary part:")
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            A_ij = imag(A[i,j])
            if abs(A_ij) < 1e-5
                A_ij = convert(Int64, floor(A_ij))
            else
                A_ij = round(A_ij, digits=2)
            end
            print(A_ij, " ")
        end
        println()
    end
end
export print_matrix

# using PyPlot, PyCall
# PyPlot.svg(true)

# function broken_plot(states, breaks, plotted_QNs)
    
#     axes_grid1 = pyimport("mpl_toolkits.axes_grid1")
#     fig = figure()
#     axes = fig.subplots(nrows=length(breaks), sharex=true)
    
#     d = 0.015 # thickness of break markers
    
#     es = [state.E for state in states]
#     max_M = maximum(basis_state.M for basis_state in states[1].basis)
    
#     prev_break_val = -Inf
#     for (i, break_val) in enumerate(breaks)
#         ax = axes[end-i+1]
#         ax.set_xticks([])
#         ax.spines["bottom"].set_visible(false)
#         ax.spines["top"].set_visible(false)
#         ax.spines["right"].set_visible(false)
#         ax.ticklabel_format(useOffset=false)
        
#         fig.add_axes(ax)
        
#         if i < length(breaks)
#             ax.plot((-d, +d), (1 - d, 1 + d), transform=ax.transAxes, color="k", clip_on=false,linewidth=0.8) 
#         end
#         if i != 1
#             ax.plot((-d, +d), (-d, +d), transform=ax.transAxes, color="k", clip_on=false, linewidth=0.8) 
#         end
        
#         plotted_QN_vals = Rational[]
        
#         M_vals = Rational[]
#         break_es = Float64[]
#         for j ∈ eachindex(states)
#             state = states[j]; e = es[j]
#             if prev_break_val < e < break_val
#                 M = is_good_quantum_number(state, :M)[2][1]
#                 push!(M_vals, M)
#                 push!(break_es, e)
#                 for QN in plotted_QNs
#                     is_QN_good, QN_val = is_good_quantum_number(state, QN)
#                     if !is_QN_good
#                         error("Quantum number is not well-defined.")
#                     else
#                         push!(plotted_QN_vals, QN_val[1])
#                     end
#                 end
#             end
#         end

#         xs = [M_vals' .+ 0.1; M_vals' .+ 0.9]
#         ys = [break_es'; break_es']
#         ax.plot(xs, ys, color="black")
        
#         min_e = minimum(ys)
#         max_e = maximum(ys)
#         scale = max_e - min_e
#         ax.set_ylim([min_e - scale * 0.25, max_e + scale * 0.25])
        
#         j = 1
#         prev_y = -Inf
        
#         for i ∈ 1:size(xs,2)
#             x = xs[1,i]
#             y = ys[1,i]
#             M = M_vals[i]
            
#             state_string = ""
#             for i ∈ 1:length(plotted_QNs)
#                 QN_val = plotted_QN_vals[j]
#                 QN_val_string = string(plotted_QNs[i]) * " = "
#                 if isinteger(QN_val)
#                     QN_val_string *= string(Int(QN_val))
#                 else
#                     QN_val_string *= string(plotted_QN_vals[j].num) * "/" * string(plotted_QN_vals[j].den)
#                 end
                
#                 M_string = "M" * " = "
#                 if isinteger(M)
#                     M_string *= string(Int(M))
#                 else
#                     M_string *= string(M[j].num) * "/" * string(M[j].den)
#                 end
                
#                 ax.text(M + 0.52, y - scale * 0.2, M_string, ha="center")
#                 if prev_y != y
#                     ax.text(max_M + 1.5, y - scale * 0.04, QN_val_string, ha="center")
#                 end
                
#                 prev_y = y

#                 j += 1
#             end
#         end
        
#         prev_break_val = break_val 
#     end

#     return nothing
# end
# export broken_plot