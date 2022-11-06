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
end

function Matrix_inital(p::ODE_initial)
    P0 = Matrix(1.0*I, p.n_states,p.n_states)
    
    if p.t0 == p.tn
        return P0
    end

    N = Int((p.tn-p.t0)/p.h)
    D = Int(size(P0)[1])
    P = zeros(Float64,D,D,N+1)
    P[:,:, 1] = P0
    return P
end

#---------------------------------------------------#

#states: 
# 0: alive, 1: disabeld, 2: deceased

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

#-----------------------------------------------#
# Rungel-Kutta functions
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
#-----------------------------------------------#


#-----------------------------------------------------------------------------------------------#
# Differenet methods
function Euler(p::ODE_initial)
    N = Int((p.tn-p.t0)/p.h)
    P = Matrix_inital(p)

    h = p.h
    t0 = p.t0

    for n in 1:N
        P[:,:, n+1] = P[:,:,n] + h*f(p.t0+n*h, P[:,:,n])
    end
    return P
end

function RK4(p::ODE_initial)
    N = Int((p.tn-p.t0)/p.h)
    P = Matrix_inital(p)

    h = p.h
    t0 = p.t0
    
    for n in 1:N
        P[:,:,n+1] = P[:,:,n] + (h/6)*(k1(t0 + n*h, P[:,:,n]) + 2*k2(t0 +n*h, P[:,:,n]) +
                                       2*k3(t0 +n*h, P[:,:, n]) + k4(t0 +n*h, P[:,:,n]))
    end

    return P
end
#-----------------------------------------------------------------------------------------------#


#---------------------------------------------#
t0 = 30       # inital age
n_states = 3  # number of states 
h = 1/12      # stepsize
tn = 120      # max age
#---------------------------------------------#

#---------------------------------------------#
initial = ODE_initial(t0, n_states, h, tn)
Euler(initial)[1,1,1:10]
RK4(initial)[1,1,1:10]
#---------------------------------------------#
