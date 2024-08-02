## loading environment 
using Pkg;
Pkg.activate(@__DIR__);
Pkg.instantiate();

## importing packages
import ForwardDiff as FD
using LinearAlgebra
using Plots
include(joinpath(@__DIR__,"helpers/DEL_dynamics.jl"))
include(joinpath(@__DIR__,"helpers/quaternions.jl"))

## constraints
function c(states, params_rbs)
	x1, q1 = states[1:3,1], states[4:7,1]
	cons_res = [x1 + quat2rot(q1) * params_rbs[1].joints[1];]
	return cons_res
end

function Dc(states, params_rbs)
	cons_jac = FD.jacobian(s -> c(s, params_rbs), states) * attitude_jacobian_block_matrix(states, params_rbs)
	return cons_jac
end 

## simulation parameters
h = 0.01 # 100 Hz
T_final = 5 # final time
T_hist = Array(0:h:T_final)
N = length(T_hist)

# model parameters
params_link1 = (m=1.0, J=Diagonal([0.1, 1.0, 1.0]), g=0, joints = [[0.5, 0, 0],]) # rigidbody parameters

m1 = params_link1.m
J1 = params_link1.J

# random initial orientation
q0 = [1, 0, 0, 0]
# q0 = normalize(rand(4))
x0 = [0, 0, 0]
v0 = [0, 0, 0]
ω0 = [0, 0, 0]

# initial conditions
link1_state0 = [x0; q0] # initial state of the rigidbody 
link1_vel0 = [v0; ω0] # initial velocity of the rigidbody
link1_momentum0 = vel2mom(link1_vel0, params_link1) # initial momentum of the rigidbody

# forcing terms
forces = [0, 0, 0]
torques = [0, 0, 0]
no_forcing = zeros(6) # no external forces

forcing = [forces; torques]

forcing_matrix = zeros(6, N)

# forcing_matrix[:, 5:15] .= forcing

# integrate
state_hist = zeros(7, N)
momentum_hist = zeros(6, N)

state_hist[:, 1] = link1_state0
momentum_hist[:, 1] = link1_momentum0

for i in 2:N

  λ = zeros(3)
  @show i
  state_hist[:, i], momentum_hist[:, i] = integrator_step(
    hcat(momentum_hist[:, i-1]),
    hcat(state_hist[:, i-1]),
    hcat(forcing_matrix[:, i-1]),
    hcat(forcing_matrix[:, i]),
    λ,
    [params_link1],
    h, 
    max_iters=50)
end