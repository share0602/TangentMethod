import constants
import vumps
from ncon import ncon
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
############################################## Parameters
D = 10;
d = 2
model = 'TFIM'
hz_field = 0.9
num_of_excite = 10
PBC = True # PBC or OBC for cylinder case
sX, sY, sZ, sI = constants.get_spin_operators(d)
Model = constants.Model(model, d, hz_field)
hloc, W, e_exact = Model.get_h_W_E()
print('We are solving ' + model + ' model!')
print('hz_field = ', hz_field)
print('e_exact = ', e_exact)


############################################## Calculation
#################### Vumps
########## MPO
print('>'*100)
print('Vumps part begin!')
A = np.random.rand(D, d, D)
e_cal, Ac, C, A_L, A_R, L_W, R_W = vumps.vumps_mpo(W,A, eta=1e-8)
e_error = abs((e_cal - e_exact) / e_exact)
print('e_error = ', e_error)
for p in [0, np.pi]:
    print('p = ', p)
    omega, _ = vumps.quasiparticle_mpo(W, p, A_L, A_R, L_W, R_W, num_of_excite=num_of_excite, system = '1D')
    print('omega = ', (omega-e_cal).real)

'''A_L1 = A_L
A_R2 = ncon([A_R, sZ],
            [[-1, 1, -3], [1, -2]])
omega = vumps.quasiparticle_domain(W,p, A_L1, A_R2, L_W, R_W, num_of_excite=5)
print('omega = ', omega-e_cal)'''


'''
#################### 2sites
e_cal, A_L, A_R, Ac,C,L_h, R_h = vumps.vumps_2sites(hloc, A, eta=1e-6)
e_error = abs((e_cal - e_exact) / e_exact)
print('e_error = ', e_error)
e_eye = e_cal * np.eye(d ** 2, d ** 2).reshape(d, d, d, d)
h_tilda = hloc - e_eye
p = 0
omega = vumps.quasiparticle_2sites(h_tilda, p, A_L, A_R, L_h, R_h, num_of_excite=5)
print('omega = ', omega)
'''