function hat(v::Vector)::Matrix
  # generate skew symmetric matrix from vector
  return [0 -v[3] v[2];
    v[3] 0 -v[1];
    -v[2] v[1] 0]
end

function L(Q::Vector)::Matrix
  # left multiply matrix of quaternion Q
  s = Q[1] # scalar part
  v = Q[2:end] # vector part
  return [s -v'; v hat(v)+s*I]
end

function R(Q::Vector)::Matrix
  # right multiply matrix of quaternion Q
  s = Q[1] # scalar part
  v = Q[2:end] # vector part
  return [s -v'; v hat(v)-s*I]
end

H = [zeros(1, 3); I]; # hat map for vectors

T = Diagonal([1; -ones(3)]); # quaternion conjugation matrix

function G(Q::Vector)::Matrix
  # generate attitude jacobian from quaternion Q
  return L(Q) * H
end

function quat2rot(Q::Vector)::Matrix
  # convert quaternion to rotation matrix
  return H' * L(Q) * R(Q) * H
end