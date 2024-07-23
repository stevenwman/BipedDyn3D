function states2tuple(states::Vector)
    """
    # Separate state vectors into position, and attitude
  
    # Arguments
    - `states`: Vector containing states of the body
  
    # Returns
    - `states_tuple`: NamedTuple containing separated states
    """
    qᴺ = states[1:3] # position of the rigidbody in world frame
    @show states[1]
  
    if length(states) == 3
      states_tuple = (qᴺ = qᴺ)
    elseif length(states) == 7
      ᴺQᴮ = states[4:7] # orientation of the rigidbody in inertial frame
      states_tuple = (qᴺ=qᴺ, ᴺQᴮ=ᴺQᴮ) # separated states
    else
      throw("Invalid state vector")
    end
    return states_tuple
  end
  
  function potential_energy(state::Vector, params_rb::NamedTuple)
    """
    # Calculate potential energy of the body
  
    # Arguments
    - `state`: Vector containing states of the body
    - `params_rb`: NamedTuple containing rigidbody parameters
  
    # Returns
    - `U`: potential energy of the system
    """
    # q = states2tuple(state).qᴺ
    q = state[1:3]
    z = q[3]
    m, g = params_rb.m, params_rb.g
    U = m * g * z
    return U
  end
  
  # left momentum term of the linear discrete euler lagrange equation
  function D2Ll(state1::Vector, state2::Vector, h::Float64, params_rb::NamedTuple)
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
    # q1, q2 = states2tuple(state1).qᴺ, states2tuple(state2).qᴺ
    q1, q2 = state1[1:3], state2[1:3]
    m = params_rb.m
    q̄ = (q1 + q2) / 2 # midpoint position
    v̄ = (q2 - q1) / h # average velocity
    ∇U = FD.gradient(state -> potential_energy(state, params_rb), q̄)
    p⁺ = m * v̄ - h / 2 * ∇U # left momentum term
    return p⁺
  end
  
  # right momentum term of the linear discrete euler lagrange equation
  function D1Ll(state1::Vector, state2::Vector, h::Float64, params_rb::NamedTuple)
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
    # q1, q2 = states2tuple(state1).qᴺ, states2tuple(state2).qᴺ
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

  function D2Lr(state1::Vector, state2::Vector, h::Float64, params_rb::NamedTuple)
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
    # Q1, Q2 = states2tuple(state1).ᴺQᴮ, states2tuple(state2).ᴺQᴮ
    Q1, Q2 = state1[4:7], state2[4:7]
    J = params_rb.J
    # calculate the right momentum term
    l⁺ = 2 / h * G(Q2)' * L(Q1) * H * J * H' * L(Q1)' * Q2
    return l⁺
  end
  
  function D1Lr(state1::Vector, state2::Vector, h::Float64, params_rb::NamedTuple)
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
    # Q1, Q2 = states2tuple(state1).ᴺQᴮ, states2tuple(state2).ᴺQᴮ
    Q1, Q2 = state1[4:7], state2[4:7]
    J = params_rb.J
    # calculate the right momentum term
    _l⁻ = 2 / h * (G(Q1)' * T * R(Q2)' * H * J * H' * L(Q1)' * Q2) # negative of right momentum term
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
    angular_momentum = momentum1[4:6]
    rm_DEL = angular_momentum + D1Lr(state1, state2, h, params_rb) + (h / 2) * (torque1 + torque2)
    return rm_DEL
  end

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
  
  function attitude_jacobian_block_matrix(states::Matrix, params_rbs::Vector{<:NamedTuple})
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
    residual = complete_DEL(momenta1, states1, states2, forcing1, forcing2, params_rbs, h)
    bodies = length(params_rbs)
  
    for i in 1:max_iters
      if norm(residual) < tol
        momenta2 = 0 * momenta1
        for j in 1:bodies
          # calculate momenta at t + h
          momenta2[:, j] = [
            D2Ll(states1[:, j], states2[:, j], h, params_rbs[j]);
            D2Lr(states1[:, j], states2[:, j], h, params_rbs[j])]
        end
        return states2, momenta2
      end
      DEL_jacobian = complete_DEL_jacobian(momenta1, states1, states2, forcing1, forcing2, params_rbs, h)
      # wrapping in real helped fixed some random matrix solve error
      Δstates = -real(DEL_jacobian) \ real(residual)
      # TODO: fix so generalizable to multiple bodies
      states2[1:3] .= states2[1:3] + Δstates[1:3]
      ϕ = Δstates[4:6]
      states2[4:7] .= L(states2[4:7]) * [sqrt(1 - ϕ' * ϕ); ϕ]
      residual = complete_DEL(momenta1, states1, states2, forcing1, forcing2, params_rbs, h)
    end
    throw("Integration did not converge")
  end
  
  function mom2vel(mom::Vector, params_rb::NamedTuple)
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
    vel = [mom[1:3] / m; J \ mom[4:end]]
    return vel
  end
  
  function vel2mom(vel::Vector, params_rb::NamedTuple)
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
    mom = [m * vel[1:3]; J * vel[4:end]]
    return mom
  end