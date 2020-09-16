import constants
from ncon import ncon
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab

'''
#################################################
Sum infinite transfer matrix
Constuct T_R/T_L using A_R/A_L,
then solve y_R from (1-T_R)^inv @ x_R = y_R
Ref: PRB 97, 045145 (2018) Appendix D
#################################################
'''
def sum_right_left(x, A_R, C, tol=1e-8):
    def map_y(y_R): ## eqn(D13) in PRB 97, 045145 (2018)
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
    y_R, info = bicgstab(LinearOperator((D ** 2, D ** 2), matvec=map_y), x_tilda.reshape(-1),x0=x_tilda.reshape(-1), tol=tol)
    y_R = y_R.reshape(D,D)
    if info != 0:
        print('bicgstab did not converge')
        exit()
    return y_R
'''
#################################################
Excitation
#################################################
'''

def Tw_to_rl(T_W):
    '''
    :param T_W: transder matrix with MPO
    :return: left and right dominant eigenvectors of T_W
    Note: If T_W = T_Wr, then l<->r
    '''
    # print('doing get_rl')
    def map_r(r):
        '''If T_W = T_Wr, then it is map_l, for <l|T_Wr = <l|'''
        r = r.reshape(D,d_w,D)
        r_out = ncon([r, T_W],
                     [[1,2,3], [1,2,3,-1,-2,-3]])
        return r_out.reshape(-1)
    def map_l(l):
        '''If T_W = T_Wr, then it is map_r, for T_Wr|r> = |r>'''
        l = l.reshape(D,d_w,D)
        l_out = ncon([T_W,l],
                     [[-1,-2,-3,1,2,3], [1,2,3]])
        return l_out.reshape(-1)
    D,d_w = T_W.shape[0], T_W.shape[1]
    l_val, l = eigs(LinearOperator((D**2*d_w, D**2*d_w), matvec=map_l), k=1, which='LM')
    l = l.reshape(D,d_w,D)
    r_val, r = eigs(LinearOperator((D**2*d_w, D**2*d_w), matvec=map_r), k=1, which='LM')
    r = r.reshape(D,d_w,D)
    # print('norm(l_val) = ', linalg.norm(l_val), 'norm(r_val) = ', linalg.norm(r_val))
    # print(l_val, r_val)
    # exit()
    return r, l

def T_to_rl(T_RL):
    '''
    :param T_RL: transder matrix with A_R up A_L down
    :return: left and right dominant eigenvectors of T_W
    Note: If T = T_LR, then l<->r
    '''
    # print('doing get_rl')
    def map_r(r):
        '''If T = T_LR, then it is map_l, for <l|T_LR = <l|'''
        r = r.reshape(D,D)
        r_out = ncon([r, T_RL],
                     [[1,2],[1,2,-1,-2]])
        return r_out.reshape(-1)
    def map_l(l):
        '''If T_W = T_Wr, then it is map_r, for T_Wr|r> = |r>'''
        l = l.reshape(D,D)
        l_out = ncon([T_RL, l],
                     [[-1,-2,1,2], [1,2]])
        return l_out.reshape(-1)
    D = T_RL.shape[0]
    l_val, l = eigs(LinearOperator((D**2, D**2), matvec=map_l), k=1, which='LM')
    l = l.reshape(D,D)
    r_val, r = eigs(LinearOperator((D**2, D**2), matvec=map_r), k=1, which='LM')
    r = r.reshape(D,D)
    # print('norm(l_val) = ', linalg.norm(l_val), 'norm(r_val) = ', linalg.norm(r_val))
    # print(l_val, r_val)
    # exit()
    return r, l

def quasi_sum_right_left_mpo(T_W, r, l, x):
    '''
    Use (1-T_W)|y> = |x> with pseudo inverse to solve y
    :param T_W: transder matrix with MPO
    :param r: right dominant vector (If T_W = T_Wr, then it is l)
    :param l: left dominant vector (If T_W = T_Wr, then it is r)
    :param x: tensor on which infinite sum we want to apply
    :return: y
    '''
    # print('doing quasi_sum_right_left')
    def trans_map(y):
        y = y.reshape(D,d_w,D)
        term1 = y
        term2 = ncon([T_W, y],
                     [[-1,-2,-3,1,2,3], [1,2,3]])
        term3 = ncon([r,y],
                     [[1,2,3], [1,2,3]])*l
        y_out = term1 + term2 + term3
        return y_out.reshape(-1)
    D,d_w,_ = x.shape
    x_tilda = x - ncon([x,r],[[1,2,3],[1,2,3]])*l
    y, info = bicgstab(LinearOperator((D**2*d_w, D**2*d_w), matvec=trans_map), x_tilda.reshape(-1),
                       x0=x_tilda.reshape(-1))
    # y, info = bicg(LinearOperator((D ** 2 * d_w, D ** 2 * d_w), matvec=trans_map), x_tilda.reshape(-1),
    #                    x0=x_tilda.reshape(-1))
    y = y.reshape(D,d_w,D)
    if info != 0:
        print('bicgstab did not converge!')
        exit()
    return y

def quasi_sum_right_left_2sites(T_RL, r, l, x):
    '''
    Use (1-T_W)|y> = |x> with pseudo inverse to solve y
    :param T_RL: transder matrix with A_R up A_L down
    :param r: right dominant vector (If T = T_LR, then it is l)
    :param l: left dominant vector (If T = T_LR, then it is r)
    :param x: tensor on which infinite sum we want to apply
    :return: y
    '''
    # print('doing quasi_sum_right_left')
    def trans_map(y):
        y = y.reshape(D,D)
        term1 = y
        term2 = ncon([T_RL, y],
                     [[-1,-2,1,2], [1,2]])
        term3 = ncon([r,y],
                     [[1,2], [1,2]])*l
        y_out = term1 + term2 + term3
        return y_out.reshape(-1)
    D,_ = x.shape
    x_tilda = x - ncon([x,r],[[1,2],[1,2]])*l
    y, info = bicgstab(LinearOperator((D**2, D**2), matvec=trans_map), x_tilda.reshape(-1),
                       x0=x_tilda.reshape(-1))
    # y, info = bicg(LinearOperator((D ** 2 * d_w, D ** 2 * d_w), matvec=trans_map), x_tilda.reshape(-1),
    #                    x0=x_tilda.reshape(-1))
    y = y.reshape(D,D)
    if info != 0:
        print('bicgstab did not converge!')
        exit()
    return y