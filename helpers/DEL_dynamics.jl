function mom2vel(
  mom::Vector,
  params_rb::NamedTuple)
  """
  # Calculate velocity from momentum

  # Arguments
  - `mom`: Vector containing momentum of the body
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `vel`: velocity of the body
  """
  m = params_rb.m
  J = params_rb.J
  vel = [mom[1:3] / m; J \ mom[4:6]]
  return vel
end

function vel2mom(
  vel::Vector,
  params_rb::NamedTuple)
  """
  # Calculate momentum from velocity

  # Arguments
  - `vel`: Vector containing velocity of the body
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `mom`: momentum of the body
  """
  m = params_rb.m
  J = params_rb.J
  mom = [m * vel[1:3]; J * vel[4:6]]
  return mom
end

function potential_energy(
  state::Vector,
  params_rb::NamedTuple)
  """
  # Calculate potential energy of the body

  # Arguments
  - `state`: Vector containing states of the body
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `U`: potential energy of the system
  """
  q = state[1:3]
  z = q[3]
  m, g = params_rb.m, params_rb.g
  U = m * g * z
  return U
end

# left momentum term of the linear discrete euler lagrange equation
function D2Ll(
  state1::Vector,
  state2::Vector,
  h::Float64,
  params_rb::NamedTuple)
  """
  # Calculate left momentum term of the discrete euler lagrange equation

  # Arguments
  - `state1`: Vector containing states of the body at time t
  - `state2`: Vector containing states of the body at time t + h
  - `h`: time step
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `p⁺`: left momentum term
  """
  q1, q2 = state1[1:3], state2[1:3]
  m = params_rb.m
  q̄ = (q1 + q2) / 2 # midpoint position
  v̄ = (q2 - q1) / h # average velocity
  ∇U = FD.gradient(state -> potential_energy(state, params_rb), q̄)
  p⁺ = m * v̄ - h / 2 * ∇U # left momentum term
  return p⁺
end

# right momentum term of the linear discrete euler lagrange equation
function D1Ll(
  state1::Vector,
  state2::Vector,
  h::Float64,
  params_rb::NamedTuple)
  """
  # Calculate right momentum term of the discrete euler lagrange equation

  # Arguments
  - `state1`: Vector containing states of the body at time t
  - `state2`: Vector containing states of the body at time t + h
  - `h`: time step
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `_p⁻`: right momentum term
  """
  q1, q2 = state1[1:3], state2[1:3]
  m = params_rb.m
  q̄ = (q1 + q2) / 2 # midpoint position
  v̄ = (q2 - q1) / h # average velocity
  ∇U = FD.gradient(state -> potential_energy(state, params_rb), q̄)
  _p⁻ = -m * v̄ - h / 2 * ∇U # negative of right momentum term
  return _p⁻
end

function linear_momentum_DEL(
  momentum1::Vector,
  state1::Vector,
  state2::Vector,
  cg_force1::Vector,
  cg_force2::Vector,
  params_rb::NamedTuple,
  h::Float64)
  """
  # Calculate discrete Lagrangian of the body

  # Arguments
  - `momentum1`: Vector containing momentum of the body at time t
  - `state1`: Vector containing states of the body at time t
  - `state2`: Vector containing states of the body at time t + h
  - `cg_force1`: Vector containing forcing terms at time t - h/2
  - `cg_force2`: Vector containing forcing terms at time t + h/2
  - `params_rb`: NamedTuple containing rigidbody parameters
  - `h`: time step

  # Returns
  - `lm_DEL`: residual of the discrete linear momentum Lagrangian of the body using midpoint integration
  """
  linear_momentum = momentum1[1:3]
  lm_DEL = linear_momentum + D1Ll(state1, state2, h, params_rb) + h / 2 * (cg_force1 + cg_force2)
  return lm_DEL
end

function D2Lr(
  state1::Vector,
  state2::Vector,
  h::Float64,
  params_rb::NamedTuple)
  """
  # Calculate left momentum term of the discrete euler lagrange equation

  # Arguments
  - `state1`: Vector containing states of the body at time tₖ
  - `state2`: Vector containing states of the body at time tₖ + h
  - `h`: time step
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `l⁺`: left momentum term
  """
  Q1, Q2 = state1[4:7], state2[4:7]
  J = params_rb.J
  # calculate the right momentum term
  l⁺ = (2.0/h) * G(Q2)' * L(Q1) * H * J * H' * L(Q1)' * Q2
  return l⁺
end

function D1Lr(
  state1::Vector,
  state2::Vector,
  h::Float64,
  params_rb::NamedTuple)
  """
  # Calculate right momentum term of the discrete euler lagrange equation

  # Arguments
  - `state1`: Vector containing states of the body at time t
  - `state2`: Vector containing states of the body at time t + h
  - `h`: time step
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `_l⁻`: negative of the right momentum term
  """
  Q1, Q2 = state1[4:7], state2[4:7]
  J = params_rb.J
  # calculate the right momentum term
  _l⁻ = (2.0/h) * (G(Q1)' * T * R(Q2)' * H * J * H' * L(Q1)' * Q2) # negative of right momentum term
  return _l⁻
end

function rotational_momentum_DEL(
  momentum1::Vector,
  state1::Vector,
  state2::Vector,
  torque1::Vector,
  torque2::Vector,
  params_rb::NamedTuple,
  h::Float64)
  """
  # Calculate discrete Lagrangian of the body

  # Arguments
  - `momentum1`: Vector containing momentum of the body at time t
  - `state1`: Vector containing states of the body at time t
  - `state2`: Vector containing states of the body at time t + h
  - `torque1`: Vector containing torques at time t - h/2
  - `torque2`: Vector containing torques at time t + h/2
  - `params_rb`: NamedTuple containing rigidbody parameters
  - `h`: time step

  # Returns
  - `rm_DEL`: residual of the discrete Lagrangian of the body using midpoint integration
  """
  # calculate the discrete lagrangian
  J = params_rb.J
  velocity1 = mom2vel(momentum1, params_rb)
  ω1 = velocity1[4:6]
  # Q1 = state1[4:7]
  # Q2 = L(Q1) * [1; (h/2.0) * ω1]
  # Q3 = state2[4:7]
  # rm_DEL =  (2.0/h) * G(Q2)' * H * J * (G(Q2)' * Q3) + (h/2.0) * (torque1 + torque2)
  rm_DEL = J*ω1 + D1Lr(state1, state2, h, params_rb) + h / 2 * (torque1 + torque2)
  return rm_DEL
end

# function angular_momentum_update(
#   state1::Vector, 
#   state2::Vector, 
#   h::Float64, 
#   params_rb::NamedTuple)
#   """
#   # Calculate the updated angular momentum

#   # Arguments
#   - `state1`: Vector containing states of the body at time t
#   - `state2`: Vector containing states of the body at time t + h
#   - `h`: time step
#   - `params_rb`: NamedTuple containing rigidbody parameters

#   # Returns
#   - `angular_momentum`: updated angular momentum
#   """
#   Q1, Q2 = state1[4:7], state2[4:7]
#   J = params_rb.J
#   # calculate the updated angular momentum between 2 knot points
#   angular_momentum = (2.0/h) * J * H' * L(Q1)' * Q2 
#   return angular_momentum
# end

# function angular_momentum_update(
#   momentum1::Vector,
#   state1::Vector, 
#   state2::Vector, 
#   h::Float64, 
#   params_rb::NamedTuple)
#   """
#   # Calculate the updated angular momentum

#   # Arguments
#   - `state1`: Vector containing states of the body at time t
#   - `state2`: Vector containing states of the body at time t + h
#   - `h`: time step
#   - `params_rb`: NamedTuple containing rigidbody parameters

#   # Returns
#   - `angular_momentum`: updated angular momentum
#   """
#   J = params_rb.J
#   velocity1 = mom2vel(momentum1, params_rb)
#   ω1 = velocity1[4:6]
#   Q1 = state1[4:7]
#   Q2 = normalize(L(Q1) * [1; (h/2.0) * ω1])
#   Q3 = state2[4:7]

#   # calculate the updated angular momentum between 2 knot points
#   angular_momentum = (2.0/h) * J * H' * L(Q2)' * Q3
#   return angular_momentum
# end

function complete_DEL(
  momenta1::Matrix,
  states1::Matrix,
  states2::Matrix,
  forcing1::Matrix,
  forcing2::Matrix,
  params_rbs::Vector{<:NamedTuple},
  h::Float64)
  """
  # Calculate discrete Lagrangian of the system

  # Arguments
  - `momenta1`: Matrix containing momenta of the system at time t (each column is a rigidbody)
  - `state1`: Matrix containing states of the system at time t
  - `state2`: Matrix containing states of the system at time t + h
  - `forcing1`: Matrix containing forcing (Fx,Fy,Fz,τx,τy,τz) terms at time t - h/2 
  - `forcing2`: Matrix containing forcing terms at time t + h/2
  - `params_rb`: NamedTuple containing rigidbody parameters
  - `h`: time step

  # Returns
  - `DEL`: residual of the discrete Lagrangian of the system using midpoint integration
  """
  DEL = []
  for i in 1:size(momenta1, 2)
    append!(DEL, linear_momentum_DEL(
      momenta1[:, i],
      states1[:, i],
      states2[:, i],
      forcing1[1:3, i],
      forcing2[1:3, i],
      params_rbs[i],
      h))
    append!(DEL, rotational_momentum_DEL(
      momenta1[:, i],
      states1[:, i],
      states2[:, i],
      forcing1[4:6, i],
      forcing2[4:6, i],
      params_rbs[i],
      h))
  end
  return DEL
end

function complete_DEL_jacobian(
  momenta1::Matrix,
  states1::Matrix,
  states2::Matrix,
  forcing1::Matrix,
  forcing2::Matrix,
  params_rbs::Vector{<:NamedTuple},
  h::Float64)
  """
  # Calculate jacobian of the discrete Lagrangian of the system

  # Arguments
  - `momenta1`: Matrix containing momenta of the system at time t (each column is a rigidbody)
  - `state1`: Matrix containing states of the system at time t
  - `state2`: Matrix containing states of the system at time t + h
  - `forcing1`: Matrix containing forcing (Fx,Fy,Fz,τx,τy,τz) terms at time t - h/2
  - `forcing2`: Matrix containing forcing terms at time t + h/2
  - `params_rb`: NamedTuple containing rigidbody parameters
  - `h`: time step

  # Returns
  - `complete_jacobian`: jacobian of the discrete Lagrangian of the system using midpoint integration
  """
  vanilla_jacobian = FD.jacobian(s2 -> complete_DEL(momenta1, states1, s2, forcing1, forcing2, params_rbs, h), states2)
  Ḡ = attitude_jacobian_block_matrix(states2, params_rbs)
  complete_jacobian = vanilla_jacobian * Ḡ
  return complete_jacobian
end

function attitude_jacobian_block_matrix(
  states::Matrix,
  params_rbs::Vector{<:NamedTuple})
  """
  # Generate attitude jacobian block matrix for the system

  # Arguments
  - `states`: Matrix containing states of the system
  - `params_rb`: NamedTuple containing rigidbody parameters

  # Returns
  - `attitude_jacobian_matrix`: attitude jacobian block matrix
  """
  bodies = length(params_rbs)
  attitude_jacobian_matrix = zeros(7 * bodies, 6 * bodies)
  for i in 1:bodies
    # multiply rows of rotational DEL with attitude jacobian, the rest with identity
    attitude = states[4:7, i]
    attitude_jacobian_matrix[7*i-6:7*i-4, 6*i-5:6*i-3] = I(3)
    attitude_jacobian_matrix[7*i-3:7*i, 6*i-2:6*i] = G(attitude)
  end
  return attitude_jacobian_matrix
end

# TODO: add constraint terms

function integrator_step(
  momenta1::Matrix,
  states1::Matrix,
  forcing1::Matrix,
  forcing2::Matrix,
  params_rbs::Vector{<:NamedTuple},
  h::Float64;
  tol=1e-6,
  max_iters=100)
  """
  # Integrate the system by doing Newton root-finding on the discrete euler lagrange equations

  # Arguments
  - `momenta1`: Matrix containing momenta of the system at time t (each column is a rigidbody)
  - `state1`: Matrix containing states of the system at time t
  - `forcing1`: Matrix containing forcing (Fx,Fy,Fz,τx,τy,τz) terms at time t - h/2
  - `forcing2`: Matrix containing forcing terms at time t + h/2
  - `params_rb`: NamedTuple containing rigidbody parameters
  - `h`: time step
  - `tol`: tolerance for convergence
  - `max_iters`: maximum number of iterations

  # Returns
  - `states2`: Matrix containing states of the system at time t + h
  - `momenta2`: Matrix containing momenta of the system at time t + h
  """
  states2 = states1 # initial guess
  bodies = length(params_rbs)

  for i in 1:max_iters
    residual = complete_DEL(momenta1, states1, states2, forcing1, forcing2, params_rbs, h)
    
    if norm(residual) < tol
      momenta2 = 0 * momenta1 # initialize momenta2
      for j in 1:bodies
        # calculate momenta at t + h
        momenta2[:, j] .= [
          D2Ll(states1[:, j], states2[:, j], h, params_rbs[j]);
          # angular_momentum_update(states1[:, j], states2[:, j], h, params_rbs[j])]
          D2Lr(states1[:, j], states2[:, j], h, params_rbs[j])]
      end
      return states2, momenta2
    end

    DEL_jacobian = complete_DEL_jacobian(momenta1, states1, states2, forcing1, forcing2, params_rbs, h)
    Δstates = -real(DEL_jacobian) \ real(residual) # wrapping in real helped with linear solver issue ??
    Δstates = reshape(Δstates, 6, bodies)

    new_states = 0 * states2 # why do I need another variable to hold state2 ??
    for k in 1:bodies
      δ, ϕ = Δstates[1:3, k], Δstates[4:6, k] # linear and rotational Newton steps
      linear_state2 = states2[1:3, k] + δ
      rotation_state2 = L(states2[4:7, k]) * [sqrt(1 - ϕ' * ϕ); ϕ] # quaternion update using axis angle newton step
      new_states[:, k] .= [linear_state2; normalize(rotation_state2)]
    end

    states2 = new_states
  end
  throw("Integration did not converge")
end