import constants
import vumps
from ncon import ncon
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator

########################### Parameters
D = 10;
num_of_p = 10
num_of_excite = 30
rvb = constants.get_RVB()
dim = rvb.shape[-1]
d = dim**2
W = ncon([rvb, np.conj(rvb)],
         [[1,2,3,-1, -3, -5, -7], [1,2,3, -2, -4, -6, -8]])
W = W.reshape(d, d, d, d)
W = W.transpose([0, 2, 1, 3])
print('d*D**2 = ', d*D**2)
############################################## Vumps
print('>'*100)
print('Vumps part begin')
W = W/5.70804057 ## Make the largest eigenvalue equals 1 in vumps
A = np.random.rand(D, d, D)
eta_0, Ac, C, A_L, A_R, L_W, R_W = vumps.vumps_fixed_points(W, A, eta=1e-6)
print('eta_0 = ', eta_0)


for p in [0, np.pi*0.9]:
    print('p = ', p)
    omega, _ = vumps.quasiparticle_mpo(W, p, A_L, A_R, L_W, R_W, num_of_excite=num_of_excite, system='RVB')
    print('omega = ', omega)
    phi = np.angle(omega)/np.pi*180
    print('phi = ', np.array2string(phi, formatter={'float_kind':lambda phi: "%.2f" % phi}))
    min_ln = -np.log(abs(omega / eta_0))
    print('min_ln = ', min_ln)
    print('sorted(min_ln) = ', sorted(list(min_ln)))
exit()