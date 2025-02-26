#!/usr/bin/python

"""
Módulo para a construção de matrizes e funções auxiliares para a resolução de splines cúbicas em 3D.

Este módulo contém funções para gerar as matrizes P, Q e B, bem como vetores C e D necessários para
resolver um problema de interpolação cúbica em três dimensões.

Funções principais:
- `get_cubic3dsolver_pmatrix(N)`: Gera a matriz P usada na interpolação cúbica em 3D.
- `get_cubic3dsolver_qmatrix(N)`: Gera a matriz Q usada na interpolação cúbica em 3D.
- `get_cubic3dsolver_bmatrix(N)`: Gera a matriz B combinando as matrizes P e Q.
- `get_cubic3dsolver_ccolvector(Points)`: Gera o vetor coluna C com as coordenadas dos pontos fornecidos.
- `get_cubic3dsolver_dmatrix(N, weight_pr, weight_pp, weight_dpdp, weight_ddpddp)`: Gera a matriz D de pesos.
- `func(v, v0)`: Função auxiliar para calcular um valor modificado com um limiar v0.
- `dfunc(v, v0)`: Derivada da função auxiliar `func`.
- `func_vec(V, v0)`: Aplica a função `func` em um vetor.
- `dfunc_vec(V, v0)`: Aplica a derivada da função `func` em um vetor.
- `cost_func(Points, w, beta, weight_pr, weight_pp, weight_dpdp, weight_ddpddp, func_offset)`: Função de custo para otimização.
"""

import numpy as np

from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q00
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q01
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q10
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q11
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q20
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q21
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverSc import square_curvature

def get_cubic3dsolver_pmatrix(N):
    """Gera a matriz P usada na interpolação cúbica em 3D."""
    Nr, Nc = Q00.shape
    
    P = np.zeros((Nr * N, Nc * (N - 1)))
    
    for l in range(N - 1):
        P[Nr * l:(Nr * l + Nr), Nc * l:(Nc * l + Nc)] = Q01
    
    P[(Nr * (N - 1)):(Nr * N), Nc * (N - 2):(Nc * (N - 1))] = Q00
    return P

def get_cubic3dsolver_qmatrix(N):
    """Gera a matriz Q usada na interpolação cúbica em 3D."""
    Nr, Nc = Q00.shape
    Q = np.zeros((3 * Nr * (N - 2), Nc * (N - 1)))
    
    for l in range(N - 2):
        Q[ Nr*(3*l+0):Nr*(3*l+1), Nc*(l+0):Nc*(l+1) ] = Q00;
        Q[ Nr*(3*l+0):Nr*(3*l+1), Nc*(l+1):Nc*(l+2) ] =-Q01;
        
        Q[ Nr*(3*l+1):Nr*(3*l+2), Nc*(l+0):Nc*(l+1) ] = Q10;
        Q[ Nr*(3*l+1):Nr*(3*l+2), Nc*(l+1):Nc*(l+2) ] =-Q11;
        
        Q[ Nr*(3*l+2):Nr*(3*l+3), Nc*(l+0):Nc*(l+1) ] = Q20;
        Q[ Nr*(3*l+2):Nr*(3*l+3), Nc*(l+1):Nc*(l+2) ] =-Q21;
    
    return Q

def get_cubic3dsolver_bmatrix(N):
    """Gera a matriz B combinando as matrizes P e Q."""
    P = get_cubic3dsolver_pmatrix(N)
    Q = get_cubic3dsolver_qmatrix(N)
    
    return np.concatenate((P, Q), axis=0)

def get_cubic3dsolver_ccolvector(Points):
    """Gera o vetor coluna C com as coordenadas dos pontos fornecidos."""
    Nr, Nc = Q00.shape
    N = len(Points)
    C = np.zeros((Nr * N + 3 * Nr * (N - 2), 1))
    
    for n in range(N):
        C[3*n  ,0]=Points[n][0]
        C[3*n+1,0]=Points[n][1]
        C[3*n+2,0]=Points[n][2]
    
    return C
    
def get_cubic3dsolver_dmatrix(  N,
                                weight_pr=1.0,
                                weight_pp=1.0,
                                weight_dpdp=1.0,
                                weight_ddpddp=1.0):
    """Gera a matriz D de pesos para a interpolação cúbica."""
    Nr, Nc = Q00.shape
    
    wp=[weight_pr]*(Nr*N)
    wq=( [weight_pp]*Nr + [weight_dpdp]*Nr + [weight_ddpddp]*Nr )*(N-2)
    D = np.diag(wp+wq)
    
    return D;

################################################################################

def func(v, v0):
    """Função auxiliar para calcular um valor modificado com um limiar v0."""
    if v >= v0:
        return v - v0
    elif v <= -v0:
        return v + v0
    else:
        return 0.0

def dfunc(v, v0):
    """Derivada da função auxiliar func."""
    if v>=v0:
        return 1.0
    elif v<=-v0:
        return 1.0
    else:
        return 0.0

def func_vec(V, v0):
    """Aplica a função auxiliar em um vetor."""
    L=V.size;
    N=int((L+18)/12);# number of points
    
    Vout=V.copy();
    
    for n in range(3,3*N-3):
        Vout[n,0]=func(V[n,0],v0);
    
    return Vout;

def dfunc_vec(V, v0):
    """Aplica a derivada da função auxiliar em um vetor."""
    L = V.size;
    N = int((L + 18)/12);# number of points
    
    J = np.eye(L)
    
    for n in range(3,3*N-3):
        J[n, n] = dfunc(V[n, 0], v0)
    
    return J


################################################################################
def cost_func(  Points,
                w,
                beta=0.001,
                weight_pr=1.0,
                weight_pp=1.0,
                weight_dpdp=1.0,
                weight_ddpddp=1.0,
                func_offset=0.0):
    """Função de custo para otimização da interpolação cúbica em 3D."""
    Nr, Nc = Q00.shape
    N=1+int(w.size/Nc)
    
    if len(w.shape)>1:
        W=w
    else:
        W=w.reshape((-1,1))
    
    B = get_cubic3dsolver_bmatrix(N)
    
    C = get_cubic3dsolver_ccolvector(Points)
    
    D = get_cubic3dsolver_dmatrix(N,
                                weight_pr=weight_pr,
                                weight_pp=weight_pp,
                                weight_dpdp=weight_dpdp,
                                weight_ddpddp=weight_ddpddp);

    v = B @ W - C
    
    if func_offset==0.0:
        E=(v.T@D@v).mean();
    else:
        E=(func_vec(v,func_offset).T@D@func_vec(v,func_offset)).mean();
    
    if beta > 0:
        E = E + beta * square_curvature(W,Nc)
    
    return E
