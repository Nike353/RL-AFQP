import torch
import numpy as np


@torch.jit.script
def quat_to_rot_mat(q):
  n = q.shape[0]

  x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  Nq = w * w + x * x + y * y + z * z
  s = 2.0 / Nq
  X, Y, Z = x * s, y * s, z * s
  wX, wY, wZ = w * X, w * Y, w * Z
  xX, xY, xZ = x * X, x * Y, x * Z
  yY, yZ = y * Y, y * Z
  zZ = z * Z

  rotation_matrix = torch.stack([
      torch.stack([1.0 - (yY + zZ), xY - wZ, xZ + wY], dim=-1),
      torch.stack([xY + wZ, 1.0 - (xX + zZ), yZ - wX], dim=-1),
      torch.stack([xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=-1)
  ],
                                dim=-2)

  return rotation_matrix

def quat_to_rot_mat_np(q):
  n = q.shape[0]
  x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  Nq = w * w + x * x + y * y + z * z
  s = 2.0 / Nq
  X, Y, Z = x * s, y * s, z * s
  wX, wY, wZ = w * X, w * Y, w * Z
  xX, xY, xZ = x * X, x * Y, x * Z
  yY, yZ = y * Y, y * Z
  zZ = z * Z

  rotation_matrix = np.stack([
      np.stack([1.0 - (yY + zZ), xY - wZ, xZ + wY], axis=-1),
      np.stack([xY + wZ, 1.0 - (xX + zZ), yZ - wX], axis=-1),
      np.stack([xZ - wY, yZ + wX, 1.0 - (xX + yY)], axis=-1)
  ],axis=-2)
  return rotation_matrix
@torch.jit.script
def copysign(a, b):
  # type: (float, Tensor) -> Tensor
  a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
  return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz_from_quaternion(q):
  qx, qy, qz, qw = 0, 1, 2, 3
  # roll (x-axis rotation)
  sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
  cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
      q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
  roll = torch.atan2(sinr_cosp, cosr_cosp)

  # pitch (y-axis rotation)
  sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
  pitch = torch.where(
      torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

  # yaw (z-axis rotation)
  siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
  cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
      q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
  yaw = torch.atan2(siny_cosp, cosy_cosp)

  return torch.stack(
      (roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)), dim=1)

def get_euler_xyz_from_quaternion_np(q):
  qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

  # roll (x-axis rotation)
  sinr_cosp = 2.0 * (qw * qx + qy * qz)
  cosr_cosp = qw**2 - qx**2 - qy**2 + qz**2
  roll = np.arctan2(sinr_cosp, cosr_cosp)

  # pitch (y-axis rotation)
  sinp = 2.0 * (qw * qy - qz * qx)
  pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))

  # yaw (z-axis rotation)
  siny_cosp = 2.0 * (qw * qz + qx * qy)
  cosy_cosp = qw**2 + qx**2 - qy**2 - qz**2
  yaw = np.arctan2(siny_cosp, cosy_cosp)

  # Wrap angles into [0, 2π)
  euler = np.stack([(roll % (2 * np.pi)),
                    (pitch % (2 * np.pi)),
                    (yaw % (2 * np.pi))], axis=1)
  return euler