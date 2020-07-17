import constants
import vumps
from ncon import ncon
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
############################################## Parameters
D = 12;
num_of_excite = 10
num_of_p = 10
aklt = constants.get_AKLT()
dim = aklt.shape[1]
W = ncon([aklt, np.conj(aklt)],
         [[1, -1, -3, -5, -7], [1, -2, -4, -6, -8]])
d = dim**2
W = W.reshape(d, d, d, d)
W = W.transpose([0, 2, 1, 3])

############################################## Vumps
print('>'*100)
print('vumps part begin')
W = W/1.30574308 ## Make the largest eigenvalue equals 1 in vumps case
print('d*D**2 = ', d*D**2)
A = np.random.rand(D, d, D)
eta_0, Ac, C, A_L, A_R, L_W, R_W = vumps.vumps_fixed_points(W, A, eta=1e-6)
print('eta_0 = ', eta_0)

# p = 0
omega_kx0 = []
omega_kxpi = []
# for p in [0, np.pi]:
for p in np.linspace(0,np.pi,11):
    print('p = ', p)
    omega, _ = vumps.quasiparticle_mpo(W, p, A_L, A_R, L_W, R_W, num_of_excite=num_of_excite, system='AKLT')
    omega_kx0_tmp = list(omega[omega > 0])
    omega_kxpi_tmp = list(omega[omega < 0])
    omega_kx0.append(omega_kx0_tmp)
    omega_kxpi.append(omega_kxpi_tmp)
    print('omega(+) = ', omega_kx0_tmp)
    print('omega(-) = ', omega_kxpi_tmp)
    min_ln0 = list(-np.log(abs(omega_kx0_tmp / eta_0)))
    min_ln0.sort()
    print('min_ln0 = ', min_ln0)
    min_lnpi = list(-np.log(abs(omega_kxpi_tmp / eta_0)))
    min_lnpi.sort()
    print('min_lnpi = ', min_lnpi)

print(omega_kx0)
np.save('D12kx0.npy', omega_kx0)
print(omega_kxpi)
np.save('D12kxpi.npy', omega_kxpi)
