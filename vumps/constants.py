from ncon import ncon
import numpy as np
from scipy import linalg

def get_spin_operators(d):
    """Returns tuple of 3 spin operators and a unit matrix for given value of spin."""
    eye = np.eye(d, dtype=complex)
    s = (d-1)/2.
    # print(s)
    sx = np.zeros([d, d], dtype=complex)
    sy = np.zeros([d, d], dtype=complex)
    sz = np.zeros([d, d], dtype=complex)

    for a in range(d):
        if a != 0:
            sx[a, a - 1] = np.sqrt((s + 1) * (2 * a) - (a + 1) * a) / 2
            sy[a, a - 1] = 1j * np.sqrt((s + 1) * (2 * a) - (a + 1) * a) / 2
        if a != d - 1:
            sx[a, a + 1] = np.sqrt((s + 1) * (2 * a + 2) - (a + 2) * (a + 1)) / 2
            sy[a, a + 1] = -1j * np.sqrt((s + 1) * (2 * a + 2) - (a + 2) * (a + 1)) / 2
        sz[a, a] = s - a
    if d == 2:
        sx *= 2
        sy *= 2
        sz *= 2
    return sx, sy, sz, eye

class Model:
    def __init__(self, model, d, hz_field=0, delta=1.0):
        self.model = model
        self.d = d
        self.hz_field = hz_field
        self.delta = delta

    def get_h_W_E(self):
        model, d, hz_field, delta = self.model, self.d, self.hz_field, self.delta
        sX, sY, sZ, sI = get_spin_operators(d)
        # print(sX)
        # exit()
        sP = sX + 1j * sY
        sM = sX - 1j * sY
        if model == 'XX':
            ## XX model
            hloc = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
            d_w = 4
            W = np.zeros([d_w, d_w, d, d], dtype=complex)
            W[0, 0] = W[3, 3] = sI
            W[1, 0] = W[3, 2] = sP/2**0.5
            W[2, 0] = W[3, 1] = sM/2**0.5
            # W[0, 0] = W[3, 3] = sI
            # W[1, 0] = W[3, 1] = sX/np.sqrt(2)
            # W[2, 0] = W[3, 2] = sY/np.sqrt(2)
            Exact = -4/np.pi ## = -1.27...
        elif model == 'TFIM':
            ## TFIM
            print('hz = ', hz_field)
            hloc = (-np.kron(sX, sX) - (hz_field / 2) * (np.kron(sZ, sI) + np.kron(sI, sZ))).reshape(2, 2, 2, 2)
            # hloc = (-np.kron(sX, sX) - hz_field *np.kron(sI, sZ)).reshape(2, 2, 2, 2)
            d_w = 3
            W = np.zeros([d_w, d_w, d, d], dtype=complex)
            W[0, 0] = W[2, 2] = sI
            W[1, 0] = -sX;
            W[2, 1] = sX
            W[2, 0] = -hz_field * sZ

            # W[0, 0] = W[2, 2] = sI
            # W[1, 0] = -sZ;
            # W[2, 1] = sZ
            # W[2, 0] = -hz_field * sX
            N = 1000000;
            x = np.linspace(0, 2 * np.pi, N + 1)
            y = np.sqrt((hz_field - 1) ** 2 + 4 * hz_field * np.sin(x / 2) ** 2)
            Exact = -0.5 * sum(y[1:(N + 1)] + y[:N]) / N
        elif model == 'XXZ':
            ## XXZ model
            ## data of XXZ model
            ## Ref: "Study of the ground state of the one-dimensionalHeisenberg spin-1 chain 2"; Author: K.R. de Ruiter
            # delta        0.        0.25      0.50      0.75      1.
            # E_infty/LJ  -0.318310 -0.345180 -0.375000 -0.407659 -0.443147
            E_XXZ = {0.:-0.318310, 0.25: -0.345180, -0.50:-0.375000, 0.75:-0.407659, 1.:  -0.443147, 4.: -4.246}
            print('delta = ', delta)
            hloc = np.real(np.kron(sX, sX) +np.kron(sY,sY) +delta*np.kron(sZ,sZ)).reshape(2, 2, 2, 2)
            d_w = 5
            W = np.zeros([d_w, d_w, d, d], dtype=complex)
            W[0, 0] = W[4, 4] = sI
            W[1, 0] = W[4,1] = sX
            W[2, 0] = W[4,2] = sY
            W[3, 0] = sZ; W[4,3] = delta*sZ
            # W = W
            Exact = E_XXZ[delta]*4
        return hloc, W, Exact

def get_AKLT():
    a0 =1; a2 = np.sqrt(6); a1 = np.sqrt(3/2)
    d = 5; D = 2
    A = np.zeros([d,D,D,D,D])
    A[0,0,0,0,0] = A[4,1,1,1,1] = a2
    A[1,0,0,0,1] = A[1,0,0,1,0] = A[1,0,1,0,0] = A[1,1,0,0,0] = a1
    A[3,1,1,1,0] = A[3,1,1,0,1] = A[3,1,0,1,1] = A[3,0,1,1,1] = a1
    A[2,0,0,1,1] = A[2,0,1,1,0] = A[2,1,1,0,0] = A[2,1,0,0,1] = A[2,0,1,0,1] = A[2,1,0,1,0] = a0
    _, sY, _, _ = get_spin_operators(D)
    string = 1j*sY
    A = ncon([A,string,string],
             [[-1,-2,-3,1,2], [1,-4], [2,-5]])
    return A/np.sqrt(6)

def get_RVB():
    D = 3; d = 2
    epsilon = np.zeros([D,D,D])
    P = np.zeros([d,D,D])
    epsilon[2,2,2] = epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1
    epsilon[1,0,2] = epsilon[0,2,1] = epsilon[2,1,0] = -1
    P[0,0,2] = P[0,2,0] = P[1,1,2] = P[1,2,1] = 1
    RVB = ncon([epsilon,P,epsilon,P,P],
               [[1,-4,-5], [-1,1,2], [2,3,4],[-2,3,-6],[-3,4,-7]])

    return RVB

def create_loop_gas_operator(d):
    """Returns loop gas (LG) operator Q_LG for spin=1/2 or spin=1 Kitaev model."""

    tau_tensor = np.zeros((2, 2, 2), dtype=complex)  # tau_tensor_{i j k}

    if d%2 == 0:
        tau_tensor[0][0][0] = - 1j
    elif d%2 == 1:
        tau_tensor[0][0][0] = 1

    tau_tensor[0][1][1] = tau_tensor[1][0][1] = tau_tensor[1][1][0] = 1

    sx, sy, sz, one = get_spin_operators(d)
    d = one.shape[0]

    Q_LG = np.zeros((d, d, 2, 2, 2), dtype=complex)  # Q_LG_{s s' i j k}

    if d == 2:
        u_gamma = (sx, sy, sz)
    elif d%2 == 0:
        u_gamma = tuple(map(lambda x: -1j * linalg.expm(1j * np.pi * x), (sx, sy, sz)))
    elif d%2 == 1:
        u_gamma = tuple(map(lambda x: linalg.expm(1j * np.pi * x), (sx, sy, sz)))

    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = np.eye(d)
                if i == 0:
                    temp = temp @ u_gamma[0]
                if j == 0:
                    temp = temp @ u_gamma[1]
                if k == 0:
                    temp = temp @ u_gamma[2]
                for s in range(d):
                    for sp in range(d):
                        Q_LG[s][sp][i][j][k] = tau_tensor[i][j][k] * temp[s][sp]

    return Q_LG

def dimer_gas_operator(spin, phi):
    """Returns dimer gas operator (or variational ansatz) R for spin=1/2 or spin=1 Kitaev model."""

    zeta = np.zeros((2, 2, 2), dtype=complex)  # tau_tensor_{i j k}

    zeta[0][0][0] = np.cos(phi)
    zeta[1][0][0] = zeta[0][1][0] = zeta[0][0][1] = np.sin(phi)

    sx, sy, sz, one = get_spin_operators(spin)
    d = one.shape[0]
    R = np.zeros((d, d, 2, 2, 2), dtype=complex)  # R_DG_{s s' i j k}
    p = 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = np.eye(d)
                if i == p:
                    temp = temp @ sx
                if j == p:
                    temp = temp @ sy
                if k == p:
                    temp = temp @ sz
                for s in range(d):
                    for sp in range(d):
                        R[s][sp][i][j][k] = zeta[i][j][k] * temp[s][sp]
    return R

if __name__ == '__main__':
    get_RVB()
