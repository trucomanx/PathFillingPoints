#!/usr/bin/python

import numpy as np

from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q00
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q01
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q10
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q11
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q20
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q21
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverSc import square_curvature

def get_cubic3dsolver_pmatrix(N):
    Nr, Nc = Q00.shape;
    
    P = np.zeros((Nr*N,Nc*(N-1)));
    
    for l in range(N-1):
        P[Nr*l:(Nr*l+Nr),Nc*l:(Nc*l+Nc)]=Q01;
    P[(Nr*(N-1)):(Nr*N),Nc*(N-2):(Nc*(N-1))]=Q00;
    return P;
    
def get_cubic3dsolver_qmatrix(N):
    Nr, Nc = Q00.shape;
    Q = np.zeros((3*Nr*(N-2),Nc*(N-1)));
    
    for l in range(N-2):
        Q[ Nr*(3*l+0):Nr*(3*l+1), Nc*(l+0):Nc*(l+1) ] = Q00;
        Q[ Nr*(3*l+0):Nr*(3*l+1), Nc*(l+1):Nc*(l+2) ] =-Q01;
        
        Q[ Nr*(3*l+1):Nr*(3*l+2), Nc*(l+0):Nc*(l+1) ] = Q10;
        Q[ Nr*(3*l+1):Nr*(3*l+2), Nc*(l+1):Nc*(l+2) ] =-Q11;
        
        Q[ Nr*(3*l+2):Nr*(3*l+3), Nc*(l+0):Nc*(l+1) ] = Q20;
        Q[ Nr*(3*l+2):Nr*(3*l+3), Nc*(l+1):Nc*(l+2) ] =-Q21;
    
    return Q;
    
def get_cubic3dsolver_bmatrix(N):
    P=get_cubic3dsolver_pmatrix(N);
    Q=get_cubic3dsolver_qmatrix(N);
    
    return np.concatenate((P,Q), axis=0);
    
def get_cubic3dsolver_ccolvector(Points):
    Nr, Nc = Q00.shape;
    N=len(Points);
    C=np.zeros((Nr*N + 3*Nr*(N-2),1));
    
    for n in range(N):
        C[3*n  ,0]=Points[n][0];
        C[3*n+1,0]=Points[n][1];
        C[3*n+2,0]=Points[n][2];
    
    return C;
    
def get_cubic3dsolver_dmatrix(  N,
                                weight_pr=1.0,
                                weight_pp=1.0,
                                weight_dpdp=1.0,
                                weight_ddpddp=1.0):
    Nr, Nc = Q00.shape;
    
    wp=[weight_pr]*(Nr*N);
    wq=( [weight_pp]*Nr + [weight_dpdp]*Nr + [weight_ddpddp]*Nr )*(N-2);
    D = np.diag(wp+wq);
    
    return D;

################################################################################

def func(v,v0):
    if v>=v0:
        return v-v0;
    elif v<=-v0:
        return v+v0;
    else:
        return 0.0;

def dfunc(v,v0):
    if v>=v0:
        return 1.0;
    elif v<=-v0:
        return 1.0;
    else:
        return 0.0;

def func_vec(V,v0):
    L=V.size;
    N=1+int((L+6)/12);
    
    Vout=V.copy();
    
    Vout[0,0]    =V[0,0];#V[0,0]*(3*N-1)/6.0;
    Vout[1,0]    =V[1,0];#V[1,0]*(3*N-1)/6.0;
    Vout[2,0]    =V[2,0];#V[2,0]*(3*N-1)/6.0;
    Vout[3*N-3,0]=V[3*N-3,0];#V[3*N-3,0]*(3*N-1)/6.0;
    Vout[3*N-2,0]=V[3*N-2,0];#V[3*N-2,0]*(3*N-1)/6.0;
    Vout[3*N-1,0]=V[3*N-1,0];#V[3*N-1,0]*(3*N-1)/6.0;
    for n in range(3,3*N-3):
        Vout[n,0]=func(V[n,0],v0);
    
    return Vout;

def dfunc_vec(V,v0):
    L=V.size;
    N=1+int((L+6)/12);
    
    J=np.eye(L);
    
    J[0,0]        =1.0;#(3*N-1)/6.0;
    J[1,0]        =1.0;#(3*N-1)/6.0;
    J[2,0]        =1.0;#(3*N-1)/6.0;
    J[3*N-3,3*N-3]=1.0;#(3*N-1)/6.0;
    J[3*N-2,3*N-2]=1.0;#(3*N-1)/6.0;
    J[3*N-1,3*N-1]=1.0;#(3*N-1)/6.0;
    for n in range(3,3*N-3):
        J[n,n]=dfunc(V[n,0],v0);
    
    return J;


################################################################################
def cost_func(  Points,
                w,
                beta=0.001,
                weight_pr=1.0,
                weight_pp=1.0,
                weight_dpdp=1.0,
                weight_ddpddp=1.0,
                func_offset=0.0):
    Nr, Nc = Q00.shape;
    N=1+int(w.size/Nc);
    
    if len(w.shape)>1:
        W=w
    else:
        W=w.reshape((-1,1));
    
    B=get_cubic3dsolver_bmatrix(N);
    
    C=get_cubic3dsolver_ccolvector(Points);
    
    D=get_cubic3dsolver_dmatrix(N,
                                weight_pr=weight_pr,
                                weight_pp=weight_pp,
                                weight_dpdp=weight_dpdp,
                                weight_ddpddp=weight_ddpddp);

    v=B@W-C;
    
    if func_offset==0.0:
        E=(v.T@D@v).mean();
    else:
        E=(func_vec(v,func_offset).T@D@func_vec(v,func_offset)).mean();
    
    if beta>0:
        E=E+beta*square_curvature(W,Nc);
    
    return E;
