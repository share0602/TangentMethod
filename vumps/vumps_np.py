from ncon import ncon
import numpy as np
from numpy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab

import copy


'''
Index Ordering Convention
   
   0--A--2   0--A_L--2   2--A_R--0  0--Ac--2
      |          |           |         |
      1          1           1         1
                                                             2   
 ---1    1---                   3--A_L--1   1--A_R--3        |
 |          |                       |           |         0--W--1
 L          R       0--C--1         |           |            |
 |          |                   2--A_L--0   0--A_R--2        3
 ---0    0---
Three sections:
1.Algorithm that will be used for both 2sites and MPO
2.Functions only for 2sites VUMPS
3.Functions only for MPO VUMPS
'''

'''
########################################################################################################################
Section 1 Algorithm that will be used for both 2sites and MPO
Contain three parts:
1.Change uMPS to canonical form.
2.Find {A_L, A_R} from a given {Ac, C} 
3.Sum infinite transfer matrix (most FREQUENTLY used) 
########################################################################################################################

###########################################################
Part 1-1
Change uMPS to canonical form: method 1
See Algorithm 1 & 2 in arXiv:1810.07006v3
This method seems to be unstable, and doesn't been used now.
For impatience, just SKIP this part
###########################################################
'''

def left_orthonormalize(A, L0, eta=1e-10):
    '''
    Algorithm 1: Gauge transform a uniform MPS A into left-orthogonal form
    0--L--1 0--A--2 ----> 0--A_L--2  0--L--1
               |     QR      |
               1             1
    '''
    print('left_orthonormalize begin!')
    D,d,_ = A.shape
    def transfer_map(X):
        '''
        1--- --A--(-2)           ----1
        |     |                 |
        x     |3        ----->  x
        |     |                 |
        2--- --A_L--(-1)         ----0
        '''
        X = X.reshape(D,D)
        X_out = ncon([X,A,np.conj(A_L)],
                     [[2,1], [1,3,-2], [2,3,-1]])
        return X_out.reshape(D*D)

    L = L0 / linalg.norm(L0)
    L_old = L
    ## QR
    LA = ncon([L,A],
              [[-1,1],[1,-2,-3]])
    A_L, L = linalg.qr(LA.reshape(D*d,D))
    A_L = A_L.reshape(D,d,D)
    lam = linalg.norm(L); L = L / lam
    delta = linalg.norm(L + L_old)
    print('delta = ', delta)
    while not (delta < eta or abs(delta-2) < eta):
        ## Arnoldi
        L = L.reshape(-1)
        _, L = eigs(LinearOperator((D**2, D**2), matvec=transfer_map), k=1, which='LM',
                    v0=L, tol=eta / 10)
        L = L.reshape(D,D)
        _,L = linalg.qr(L)
        L = L / linalg.norm(L)
        L_old = L
        LA = ncon([L, A],
                  [[-1, 1], [1, - 2, -3]])
        A_L, L = linalg.qr(LA.reshape(D * d, D))
        A_L = A_L.reshape(D, d, D)
        lam = linalg.norm(L); L = L/lam;
        delta = linalg.norm(L + L_old)
        print('delta = ', delta)
    return A_L, L, lam

def right_orthonormalize(A, R0, eta=1e-10):
    '''
    Algorithm 1: Gauge transform a uniform MPS A into right-orthogonal form
    0--A--2  1--R--0           1--R--0  2--A_R--0
       |               ----->               |
       1                 QR                 1
    '''
    print('right_orthonormalize begin!')
    D,d,_ = A.shape
    def transfer_map(X):
        X = X.reshape(D,D)
        X_out = ncon([X,A,np.conj(A_R)],
                     [[2,1], [-2,3,1], [2,3,-1]])
        return X_out.reshape(D*D)

    R = R0 / linalg.norm(R0)
    R_old = copy.deepcopy(R)
    ## QR
    RA = ncon([R,A],
              [[-1,1],[-3,-2,1]])
    A_R, R = linalg.qr(RA.reshape(D*d,D))
    A_R = A_R.reshape(D,d,D)
    lam = linalg.norm(R); R = R / lam
    delta = linalg.norm(R - R_old)
    print('delta = ', delta)
    while not (delta < eta or abs(delta-2) < eta):
        ## Arnoldi
        R = R.reshape(-1)
        _, R = eigs(LinearOperator((D**2, D**2), matvec=transfer_map), k=1, which='LM',
                    v0=R, tol=eta/10 )
        R = R.reshape(D,D)
        _,R = linalg.qr(R)
        R = R / linalg.norm(R)
        R_old = copy.deepcopy(R)
        RA = ncon([R, A],
                  [[-1, 1], [-3,-2,1]])
        A_R, R = linalg.qr(RA.reshape(D * d, D))
        A_R = A_R.reshape(D, d, D)
        lam = linalg.norm(R);
        R = R / lam
        delta = linalg.norm(R - R_old)
        print('delta = ', delta)
    return A_R, R, lam

def mixed_canonical(A, eta = 1e-10):
    A_L, _, lam = left_orthonormalize(A,L0,eta)
    A_R, C, _ = right_orthonormalize(A_L,R0,eta)
    U,C,V_dagger = linalg.svd(C, full_matrices=False)
    A_L = ncon([np.conj(U.T), A_L, U],
               [[-1,1], [1,-2,2], [2,-3]])
    V = np.conj(V_dagger.T)
    A_R = ncon([V_dagger, A_R, V],
               [[-3,1], [2,-2,1], [2,-1]])
    return A_L,A_R, np.diag(C), lam

'''
########################################################
Part 1-2
Change uMPS to canonical form: method 2
Ref: R. Orús and G. Vidal. PRB78,155117(2008)􏱮
Three functions:
1.A_to_lam_gamma
2.lam_gamma_to_canonical
3.canonical_to_Al_Ar
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
The final result we want is A_L, A_R, and C
Ac should be calculate using (A_L,C) or (C,A_R)
For instance, most of the time, we use
lam, gamma = A_to_lam_gamma(A)
lam, gamma = lam_gamma_to_canonical(lam, gamma)
A_L, A_R = canonical_to_Al_Ar(lam, gamma)
C = lam
Ac = ncon([A_L, C],
          [[-1, -2, 1], [1, -3]])
########################################################
'''
def A_to_lam_gamma(A):
    '''
    A = U@S@V_da
    return (lam = S; gama = V_da@U)
    '''
    D,d, _ = A.shape
    u,s,v_da = linalg.svd(A.reshape(D*d,D), full_matrices=False)
    lam = np.diag(s)/linalg.norm(s)
    u = u.reshape(D,d,D)
    gamma = ncon([v_da, u],
                 [[-1,1], [1,-2,-3]])
    norm = ncon([lam@lam, gamma, lam@lam, np.conj(gamma)],
                [[1,4], [1,3,2], [2,5], [4,3,5]])**0.5
    gamma = gamma/norm
    return lam, gamma


def lam_gamma_to_canonical(lam, gamma):
    '''
    Get (lam,gamma) to canonical (lam,gamma)
    '''
    def get_L_or_R(l, dtol=1e-16):
        dtemp, utemp = linalg.eigh(l)
        dtemp = abs(dtemp)
        chitemp = sum(dtemp > dtol)
        dtemp = abs(dtemp)
        # print(dtemp)
        d = np.diag(dtemp[-1:-(chitemp + 1):-1])
        U_da = np.conj(utemp[:, -1:-(chitemp + 1):-1].T)
        L = np.sqrt(d) @ U_da
        return L
    def transfer_map_l(l):
        l = l.reshape(D,D)
        l_out = ncon([l,lam,gamma,lam,gamma],
                     [[3,1],[1,2],[2,5,-2],[3,4],[4,5,-1]])
        return l_out
    def transfer_map_r(r):
        r = r.reshape(D,D)
        r_out = ncon([gamma,lam,np.conj(gamma), np.conj(lam), r],
                     [[-2,1,2], [2,3],[-1,1,4],[4,5],[5,3]])
        return r_out
    D,d,_ = gamma.shape
    l0 = np.random.rand(D,D)
    l0 = l0.reshape(-1)
    norm1, l = eigs(LinearOperator((D ** 2, D ** 2), matvec=transfer_map_l), k=1, which='LM',
                v0=l0)
    # print('norm = ', norm1[0])
    l = l.reshape(D,D)
    r0 = np.random.rand(D, D)
    r0 = r0.reshape(-1)
    norm2, r = eigs(LinearOperator((D ** 2, D ** 2), matvec=transfer_map_r), k=1, which='LM',
                v0=r0)
    r = r.reshape(D, D)

    L = get_L_or_R(l)
    R = get_L_or_R(r)

    u_tmp,s_tmp,vda_tmp = linalg.svd(L@lam@R.T, full_matrices=False)
    lam_new = np.diag(s_tmp)

    gamma_new = ncon([vda_tmp@linalg.inv(R.T), gamma, linalg.inv(L)@u_tmp],
                     [[-1,1], [1,-2,2], [2,-3]])
    lam_new = lam_new/linalg.norm(lam_new)
    norm = ncon([lam_new@lam_new, gamma_new, lam_new@lam_new, np.conj(gamma_new)],
                [[1,4], [1,3,2], [2,5], [4,3,5]])**0.5
    gamma_new = gamma_new/norm
    return lam_new, gamma_new
def canonical_to_Al_Ar(lam, gamma):
    A_L = ncon([lam, gamma],
               [[-1,1], [1,-2,-3]])
    A_R = ncon([gamma, lam],
               [[-3,-2,1], [1,-1]])
    return A_L, A_R

'''
##########################################################
Part 2
Find {A_L, A_R} from a given {Ac, C} 
Ref: Algorithm 5 in arXiv:1810.07006v3
##########################################################
'''
def min_Ac_C(Ac, C):
    D,d,_ = Ac.shape

    U_Ac,S_Ac,V_dagger_Ac = linalg.svd(Ac.reshape(D*d,D), full_matrices=False)
    U_c, S_c, V_dagger_c = linalg.svd(C, full_matrices=False)
    A_L = (U_Ac@V_dagger_Ac)@np.conj(U_c@V_dagger_c).T
    A_L = A_L.reshape(D,d,D)
    Ac_r = Ac.transpose([2,1,0])

    C_r = C.T
    U_Ac, S_Ac, V_dagger_Ac = linalg.svd(Ac_r.reshape(D * d, D), full_matrices=False)
    U_c, S_c, V_dagger_c = linalg.svd(C_r, full_matrices=False)
    A_R = (U_Ac@V_dagger_Ac)@np.conj(U_c@V_dagger_c).T
    A_R = A_R.reshape(D,d,D)
    return A_L, A_R

'''
#################################################
Part 3 (most FREQUENTLY used algorithm)
Sum infinite transfer matrix
Constuct T_R/T_L using A_R/A_L,
then solve y_R from (1-T_R)^inv @ x_R = y_R
Ref: PRB 97, 045145 (2018) Appendix D
#################################################
'''
def sum_right_left(x, A_R, C, tol=1e-8):
    def map(y_R): ## eqn(D13) in PRB 97, 045145 (2018)
        y_R = y_R.reshape(D,D)
        term1 = y_R
        term2 = ncon([y_R,A_R,np.conj(A_R)],
                     [[3,1],[1,2,-2],[3,2,-1]])
        term3 = ncon([y_R,L],
                     [[1,2],[1,2]])*np.eye(D,D)
        return (term1-term2+term3).reshape(-1)
    D,d,_ = A_R.shape
    L = ncon([np.conj(C), C],
             [[1,-1],[1,-2]]) # = C@np.conj(C.T)
    x_tilda = x - ncon([x,L],
                       [[1,2],[1,2]])
    y_R, info = bicgstab(LinearOperator((D ** 2, D ** 2), matvec=map), x_tilda.reshape(-1),x0=x_tilda.reshape(-1), tol=tol)
    y_R = y_R.reshape(D,D)
    if info != 0:
        print('bicgstab did not converge')
        exit()
    return y_R

'''
########################################################################################################################
Section 2 Functions only for 2sites VUMPS
Three parts:
1. Evaluate 2sites Energy 
2. Get h_L and h_R (will be used as the input of sum_right_left)
3. Final Algorithm for 2sites VUMPS
########################################################################################################################

##########################################################
Evaluate 2sites Energy
##########################################################
'''
def evaluate_energy_two_sites(A_L, A_R, Ac, h):
    e1 = ncon([A_L,Ac,h,np.conj(A_L), np.conj(Ac)],
              [[1,4,2],[2,5,3],[4,5,6,7],[1,6,8],[8,7,3]])
    e2 = ncon([Ac, A_R,h,np.conj(Ac), np.conj(A_R)],
              [[1,4,2],[3,5,2],[4,5,6,7],[1,6,8],[3,7,8]])
    e = (e1+e2)/2
    # print('abs(e-e1) = ', abs(e-e1))
    if abs(e-e1) > 1e-4:
        pass
        # print('e1 is not close to e2 !')
        # print('abs(e-e1) = ', abs(e - e1))
    return e
'''
##########################################################
Get h_L and h_R (will be used as the input of sum_right_left)
##########################################################
'''

def get_h_L(A_L, h): ## eqn(133) in arXiv:1810.07006v3
    h_L = ncon([A_L,A_L,h,np.conj(A_L), np.conj(A_L)],
              [[1,3,2],[2,4,-2],[3,4,5,6],[1,5,7],[7,6,-1]])
    return h_L

def get_h_R(A_R,h): ## eqn(133) in arXiv:1810.07006v3
    h_R = ncon([A_R, A_R, h, np.conj(A_R), np.conj(A_R)],
               [[2,3,-2],[1,4,2],[3,4,5,6],[7,5,-1],[1,6,7]])
    return h_R


'''
############################################################
Final Algorithm for 2sites VUMPS
This function will largely use the previos defined functions
Ref: Algorithm 4 in arXiv:1810.07006v3
############################################################
'''
def vumps_2sites(h, A, eta=1e-7):
    print('VUMPS for two sites begin!')
    def map_Hac(Ac): ## eqn(131) in arXiv:1810.07006v3
        Ac = Ac.reshape(D,d,D)
        term1 = ncon([A_L,Ac,h,np.conj(A_L)],
                     [[5,2,1],[1,3,-3],[2,3,4,-2],[5,4,-1]])
        term2 = ncon([Ac, A_R,h,np.conj(A_R)],
                     [[-1,2,1],[5,3,1],[2,3,-2,4],[5,4,-3]])
        term3 = ncon([L_h,Ac],
                     [[-1,1],[1,-2,-3]])
        term4 = ncon([Ac,R_h],
                     [[-1,-2,1],[1,-3]])
        final = term1+term2+term3+term4
        return final.reshape(-1)
    def map_Hc(C): ## eqn(132) in arXiv:1810.07006v3
        C = C.reshape(D,D)
        term1 = ncon([A_L,C,A_R,h,np.conj(A_L),np.conj(A_R)],
                     [[1,5,2],[2,3],[4,6,3],[5,6,7,8],[1,7,-1],[4,8,-2]])
        term2 = L_h@C
        term3 = C@R_h.T
        final = term1+term2+term3
        return final.reshape(-1)
    D,d,_ = A.shape
    lam, gamma = A_to_lam_gamma(A)
    lam, gamma = lam_gamma_to_canonical(lam, gamma)
    A_L, A_R = canonical_to_Al_Ar(lam, gamma)
    C = lam
    Ac = ncon([A_L, C],
              [[-1, -2, 1], [1, -3]])
    delta = eta*1000
    e_memory = -1
    e = 0
    count = 0
    while delta > eta and abs(e-e_memory) > eta/10:
        e_memory = e
        e = evaluate_energy_two_sites(A_L, A_R, Ac, h)
        e_eye = e * np.eye(d ** 2, d ** 2).reshape(d, d, d, d)
        h_tilda = h - e_eye
        # h_tilda = h
        h_L = get_h_L(A_L, h_tilda)
        # L_h = sum_left(h_L, A_L, C,tol=delta/10)
        C_r = C.T
        L_h = sum_right_left(h_L, A_L, C_r, tol=delta / 10)
        h_R = get_h_R(A_R, h_tilda)
        R_h = sum_right_left(h_R, A_R, C, tol=delta / 10)
        # print(Ac.shape)
        E_Ac, Ac = eigs(LinearOperator((D ** 2*d, D ** 2*d), matvec=map_Hac), k=1, which='SR',
                v0=Ac.reshape(-1), tol=delta/10)
        Ac= Ac.reshape(D,d,D)
        E_C, C = eigs(LinearOperator((D ** 2 , D ** 2 ), matvec=map_Hc), k=1, which='SR',
                     v0=C.reshape(-1), tol=delta/10)
        C = C.reshape(D,D)
        A_L, A_R = min_Ac_C(Ac,C)
        Al_C =  ncon([A_L, C],
                    [[-1, -2, 1], [1, -3]])
        delta = linalg.norm(Ac-Al_C)
        energy_error = abs((e - Exact)/Exact)

        if count%5 == 0:
            print(50*'-'+'steps',count, 50*'-')
            print('energy = ', e)
            print('delta = ', delta)
            print('Eac = ', E_Ac)
            print('Ec = ',E_C)
        count += 1
    print(50*'-'+' final '+50*'-')
    print('energy = ', e)
    print('energy error = ', energy_error)
        # exit()

'''
########################################################################################################################
Section 3 Functions only for MPO VUMPS
Two parts:
1. Construct left and right fixed points of MPO
2. Final Algorithm for MPO VUMPS
########################################################################################################################
##########################################################
Construct left and right fixed points of MPO
Note that it only includes MPO with no long-range interaction,
which means W[a,a] = 0 except for W[0,0] and W[d_w-1,d_w-1]
Ref: Algorithm 6 in PRB 97, 045145 (2018)
##########################################################
'''
def get_T_O(A_L, O):
    T_O = ncon([A_L, O, np.conj(A_L)],
               [[-4,1,-2], [1,2],[-3,2,-1]])
    return T_O
def get_Lh_Rh_mpo(A_L, A_R, C,W):
    d_w,_,_,_ = W.shape
    D,d,_ = A_L.shape
    L_W = np.zeros([d_w, D,D], dtype=complex)
    L_W[d_w-1] = np.eye(D,D)
    for i in range(d_w-2,-1,-1): # dw-2,dw-3,...,1,0
        for j in range(i+1, d_w): # j>i: i+1,...d_w-1
            L_W[i] += ncon([L_W[j],get_T_O(A_L, W[j,i])],
                           [[1,2],[-1,-2,1,2]]) # Lw[i] = Lw[j]T[j,i]
    C_r = C.T
    R = ncon([np.conj(C_r), C_r],
             [[1, -1], [1, -2]])
    e_Lw = ncon([R, L_W[0]], ## eqn (C27) in PRB 97, 045145 (2018)
                     [[1, 2], [1, 2]])
    # print('e_test_Lw = ', e_test_Lw)
    L_W[0] = sum_right_left(L_W[0], A_L, C_r)
    L_W = L_W.transpose([1,0,2])
    R_W = np.zeros([d_w, D,D], dtype=complex)
    R_W[0] = np.eye(D,D)
    for i in range (1,d_w): # 1,2,...,dw-1
        for j in range(i-1,-1,-1): # j<i: i-1,i-2,...,0
            # print('i=',i,'j=',j)
            R_W[i] += ncon([R_W[j], get_T_O(A_R,W[i,j])],
                        [[1,2], [-1,-2,1,2]]) # Rw[i] = T[i,j]R[j]
    L = ncon([np.conj(C), C],
             [[1,-1],[1,-2]])
    e_Rw = ncon([L, R_W[d_w-1]], ## eqn (C27) in PRB 97, 045145 (2018)
                      [[1,2],[1,2]])
    # print('e_test_Rw = ', e_test_Rw)
    R_W[d_w-1] = sum_right_left(R_W[d_w-1], A_R, C)
    R_W = R_W.transpose([1,0,2])
    # print(e_Rw, e_Lw)
    return L_W, R_W, (e_Lw+e_Rw)/2
'''
##############################################################
Final Algorithm for MPO VUMPS
This function will largely use the previos defined functions
Ref: Hao-Ti Hung's thesis p.26
##############################################################
'''
def vumps_mpo(W,A,eta = 1e-8):
    print('VUMPS for MPO begin!')
    def map_Hac(Ac):
        Ac = Ac.reshape(D,d,D)
        Ac_new = ncon([L_W,Ac,W,R_W],
                      [[-1,3,1],[1,5,2],[3,4,5,-2],[-3,4,2]])
        return Ac_new.reshape(-1)
    def map_Hc(C):
        C= C.reshape(D,D)
        C_new = ncon([L_W,C,R_W],
                     [[-1,3,1],[1,2],[-2,3,2]])
        return C_new.reshape(-1)
    D, d, _ = A.shape
    lam, gamma = A_to_lam_gamma(A)
    lam, gamma = lam_gamma_to_canonical(lam, gamma)
    A_L, A_R = canonical_to_Al_Ar(lam, gamma)
    C = lam
    Ac = ncon([A_L, C],
              [[-1, -2, 1], [1, -3]])
    delta = eta * 1000
    e_memory = -1
    e = 0
    count = 0
    while delta > eta and abs(e - e_memory) > eta / 10:
        L_W, R_W, energy = get_Lh_Rh_mpo(A_L,A_R,C,W)
        E_Ac, Ac = eigs(LinearOperator((D ** 2 * d, D ** 2 * d), matvec=map_Hac), k=1, which='SR',
                        v0=Ac.reshape(-1), tol=delta / 10)
        Ac = Ac.reshape(D, d, D)
        E_C, C = eigs(LinearOperator((D ** 2, D ** 2), matvec=map_Hc), k=1, which='SR',
                      v0=C.reshape(-1), tol=delta / 10)
        C = C.reshape(D, D)
        e_memory = e
        e = energy
        A_L, A_R = min_Ac_C(Ac, C)
        Al_C = ncon([A_L, C],
                    [[-1, -2, 1], [1, -3]])
        delta = linalg.norm(Ac - Al_C)
        if count % 5 == 0:
            print(50 * '-' + 'steps', count, 50 * '-')
            print('energy = ', e)
            print('delta = ', delta)
            # print('Eac = ', E_Ac)
            # print('Ec = ', E_C)
            # print('Eac/Ec = ', E_Ac/E_C)
        count += 1
    print(50 * '-' + ' final ' + 50 * '-')
    print('energy = ', e)
    energy_error = abs((e-Exact)/Exact)
    print('Error', energy_error)

'''
########################################################################################################################
Main Program
########################################################################################################################
'''

if __name__ == '__main__':
    D = 20;
    d = 2
    A = np.random.rand(D, d, D)
    L0 = np.random.randn(D, D)
    R0 = np.random.randn(D, D)
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0, -1]])
    sI = np.array([[1, 0], [0,1]])
    sP = sX + 1j * sY
    sM = sX - 1j * sY
    # print(sP)
    # exit()
    model = 'TFIM'
    hz_field = 0.9
    print('We are solving '+model+' model!')
    print('D = ', D)
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
        d_w = 3
        W = np.zeros([d_w, d_w, d, d], dtype=complex)
        W[0, 0] = W[2, 2] = sI
        W[1, 0] = -sX;
        W[2, 1] = sX
        W[2, 0] = -hz_field * sZ
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
        delta = 0.75
        E_XXZ = {0.:-0.318310, 0.25: -0.345180, -0.50:-0.375000, 0.75:-0.407659, 1.:  -0.443147}
        print('delta = ', delta)
        hloc = np.real(np.kron(sX, sX) +np.kron(sY,sY) +delta*np.kron(sZ,sZ)).reshape(2, 2, 2, 2)/4
        d_w = 5
        W = np.zeros([d_w, d_w, d, d], dtype=complex)
        W[0, 0] = W[4, 4] = sI
        W[1, 0] = W[4,1] = sX
        W[2, 0] = W[4,2] = sY
        W[3, 0] = sZ; W[4,3] = delta*sZ
        W = W/4
        Exact = E_XXZ[delta]
    print('Exact = ', Exact)

    vumps_2sites(hloc, A, eta=1e-7)
    vumps_mpo(W,A, eta=1e-8)
    # print('Exact = ', Exact)
    exit()

    # print(L_W1)
    lam, gamma = A_to_lam_gamma(A)
    lam, gamma = lam_gamma_to_canonical(lam, gamma)
    A_L, A_R = canonical_to_Al_Ar(lam,gamma)
    C = lam
    Ac = ncon([A_L,C],
              [[-1,-2,1], [1,-3]])

