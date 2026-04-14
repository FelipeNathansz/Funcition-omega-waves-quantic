import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================================
# GRID 3D (Δ)
# =====================================
N = 40
x = np.linspace(-3, 3, N)
y = np.linspace(-3, 3, N)
z = np.linspace(-3, 3, N)

X, Y, Z = np.meshgrid(x, y, z)

# =====================================
# FUNÇÃO DE ONDA ψ(Δ)
# =====================================
k = 1.9
psi = np.exp(1j * k * (X + Y + Z))

# =====================================
# CAMPO λ(Δ)
# =====================================
lambda_field = np.sin(X + Y + Z)

# =====================================
# PARÂMETROS DA MATRIZ Ohm
# =====================================
phi = 2.0
theta = 1.5
omega_val = 0.8

# =====================================
# MATRIZ Ohm (3x3)
# =====================================
Ohm = np.array([
    [phi, 0, 0],
    [0, np.exp(-omega_val), 0],
    [0, 0, theta]
], dtype=complex)

# =====================================
# MATRIZ gamma (tipo Dirac simplificada)
# =====================================
gamma = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=complex)

# =====================================
# PRÉ-CÁLCULO: ln(Ohm)
# =====================================
log_Ohm = logm(Ohm)

# =====================================
# SIMULAÇÃO
# =====================================
result_field = np.zeros((N, N, N))

for i in range(N):
    for j in range(N):
        for k_idx in range(N):

            # ψ no ponto Δ
            psi_val = psi[i, j, k_idx]

            # λ no ponto Δ
            lambda_val = lambda_field[i, j, k_idx]

            # ψ^γ = e^(γ ln ψ)
            psi_gamma = expm(gamma * np.log(psi_val))

            # Ohm^λ = e^(λ ln Ohm)
            Ohm_lambda = expm(lambda_val * log_Ohm)

            # Φ(Δ)
            Phi = Ohm_lambda @ psi_gamma

            # intensidade (componente [0,0])
            result_field[i, j, k_idx] = np.abs(Phi[0, 0])

# =====================================
# VISUALIZAÇÃO (slice 2D)
# =====================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# reduzir pontos pra não travar
step = 2

Xr = X[::step, ::step, ::step]
Yr = Y[::step, ::step, ::step]
Zr = Z[::step, ::step, ::step]
Cr = result_field[::step, ::step, ::step]

# plot 3D
sc = ax.scatter(
    Xr.flatten(),
    Yr.flatten(),
    Zr.flatten(),
    c=Cr.flatten(),
    cmap='viridis',
    marker='o',
    s=5
)

fig.colorbar(sc, ax=ax, label="|Φ(Δ)|")

ax.set_title("Onda 3D Φ(Δ)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()