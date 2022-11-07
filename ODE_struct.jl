#----------------------#
using LinearAlgebra
#----------------------#


#---------------------------------------------------#
# Storing the inital conditions:
mutable struct ODE_initial
    t0::Int
    n_states::Int
    h::Float64
    tn::Int
    P::Array

    # Initialize struct and create array
    function ODE_initial(t0, n_states, h, tn)
        P0 = Matrix(1.0 * I, n_states, n_states)

        if t0 == tn
            new(t0, n_states, h, tn, P0)
        end

        N = Int((tn - t0) / h)
        D = Int(size(P0)[1])
        P = zeros(Float64, D, D, N + 1)
        P[:, :, 1] = P0

        new(t0, n_states, h, tn, P)
    end
end

# Pretty print of ODE_initial struct
function Base.show(io::IO, z::ODE_initial)
    print(io, "t0 = $(z.t0), n_states = $(z.n_states), h = $(z.h), tn = $(z.tn)")
end
#---------------------------------------------------#

#states: 
# 0: alive, 1: disabled, 2: deceased

# Transition rate matrix Λ
function Λ(t)
    #state0:
    μ01(t) = 0.0004 + 10^(0.06*t-5.46)
    μ02(t) = 0.0005 + 10^(0.038*t-4.12)
    μ00(t) = -(μ01(t) + μ02(t))
    #state1:
    μ10(t) = 0.05
    μ12(t) = μ02(t)
    μ11(t) = -(μ10(t)+μ12(t))
    #state2:
    # transition rates in the deceased state are zero

    L = [μ00(t) μ01(t) μ02(t)
         μ10(t) μ11(t) μ12(t)
         0       0     0    ]

    return L
end

# Forward Kolmogorov
function f(t,M)
    return M*Λ(t)
end

#-----------------------------------------------------------------------------------------------#
# Methods
function Euler(p::ODE_initial)
    N = Int((p.tn-p.t0)/p.h)
    P = copy(p.P)

    h = p.h
    t0 = p.t0

    for n in 1:N
        P[:,:, n+1] = P[:,:,n] + h*f(t0+n*h, P[:,:,n])
    end

    return P
end


function Taylor(p::ODE_initial)
    N = Int((p.tn-p.t0)/p.h)
    P = copy(p.P)

    h = p.h
    t0 = p.t0 
    Id = P[:,:,1]

    for n in 1:N
        P[:,:, n+1] = P[:,:,n]*(Id + (h/2)*Λ(t0 + n*h) + (h/2)*Λ(t0 + n*h + h) + (h^(2)/2)*(Λ(t0+n*h))^2)
    end

    return P
end

function RK4(p::ODE_initial)
    N = Int((p.tn-p.t0)/p.h)
    P = copy(p.P)

    h = p.h
    t0 = p.t0

    function k1(t,M)
        return f(t,M)
    end
    
    function k2(t,M)
        return f(t+h/2, M +h*k1(t,M)/2)
    end
    
    function k3(t,M)
        return f(t+h/2, M+ h*k2(t, M)/2)
    end
    
    function k4(t,M)
        return f(t+h, M + h*k3(t,M))
    end

    for n in 1:N
        P[:,:,n+1] = P[:,:,n] + (h/6)*(k1(t0 + n*h, P[:,:,n]) + 2*k2(t0 +n*h, P[:,:,n]) +
                                       2*k3(t0 +n*h, P[:,:, n]) + k4(t0 +n*h, P[:,:,n]))
    end

    return P
end
#-----------------------------------------------------------------------------------------------#


t0 = 30       # inital age
n_states = 3  # number of states 
h = 1/12      # stepsize
tn = 120      # max age

initial = ODE_initial(t0, n_states, h, tn)
println(Euler(initial)[1, 1, 1:10])
println(Taylor(initial)[1,1,1:10])
println(RK4(initial)[1, 1, 1:10])
