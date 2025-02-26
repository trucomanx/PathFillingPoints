#  
import sys
import numpy as np
import PathFillingPoints.Splines.Linear3DSolver.Linear3DSolver as linsolver
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q00
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverPoly import DPoly
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverPoly import DDPoly

import PathFillingPoints.Splines.Cubic3DSolverMethod1.Cubic3DSolverDsc as pscca
import PathFillingPoints.Splines.Cubic3DSolverMethod1.Cubic3DSolverDscNumeric as psccn
import PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverCost as psccost
from tqdm import tqdm



def round2(number):
    """
    Arredonda um número para a notação científica com três casas decimais.
    
    Parâmetros:
    number (float): Número a ser arredondado.
    
    Retorna:
    float: Número arredondado.
    """
    formatted_number_str = "{:.3e}".format(number)
    return float(formatted_number_str)

def to_get_cubic3d_weight_list(
    Points: list,
    w0=None,
    alpha=0.01,
    max_iter=20000,
    min_mse=1e-5,
    beta=0.001,
    weight_pr=1.0,
    weight_pp=1.0,
    weight_dpdp=1.0,
    weight_ddpddp=1.0,
    func_offset=0.0,
    cnumeric=False,
    show=False
):
    """
    Resolve o problema de ajuste de splines cúbicas tridimensionais por meio de um método iterativo.
    
    Parâmetros:
    Points (list): Lista de pontos tridimensionais.
    w0 (array, opcional): Pesos iniciais. Se None, utiliza uma solução linear inicial.
    alpha (float, opcional): Taxa de aprendizado inicial. Padrão é 0.01.
    max_iter (int, opcional): Número máximo de iterações. Padrão é 20000.
    min_mse (float, opcional): Erro quadrático médio mínimo para parada. Padrão é 1e-5.
    beta (float, opcional): Peso do termo de suavização. Padrão é 0.001.
    weight_pr, weight_pp, weight_dpdp, weight_ddpddp (float, opcionais): Pesos dos diferentes termos da função de custo.
    func_offset (float, opcional): Offset da função de custo. Padrão é 0.0.
    cnumeric (bool, opcional): Se True, usa cálculos numéricos para suavização. Padrão é False.
    show (bool, opcional): Se True, exibe barra de progresso. Padrão é False.
    
    Retorna:
    tuple: (W, E), onde W são os pesos ajustados e E é a evolução do erro ao longo das iterações.
    """
    N = len(Points)
    
    Nr, Nc = Q00.shape
    
    if w0 is None:
        w0 = linsolver.to_get_linear3d_weight_list(Points)
        w0 = linsolver.change_order_of_weight_list(w0, to_order=3)
    
    B = psccost.get_cubic3dsolver_bmatrix(N)
    C = psccost.get_cubic3dsolver_ccolvector(Points)
    W = w0.reshape((-1, 1))
    
    ## Pesos 
    D = psccost.get_cubic3dsolver_dmatrix(N,
                                        weight_pr=weight_pr,
                                        weight_pp=weight_pp,
                                        weight_dpdp=weight_dpdp,
                                        weight_ddpddp=weight_ddpddp)
    
    j = 0
    if cnumeric:
        psccn.ModDeltaW.Wold = np.zeros(W.shape)
    
    if show:
        pbar = tqdm(total=max_iter)
    
    v = B @ W - C
    if func_offset == 0.0:
        E=[(v.T@(D@v)).mean()]
    else:
        E=[(psccost.func_vec(v,func_offset).T@D@psccost.func_vec(v,func_offset)).mean()]
    
    Alpha=alpha
    
    while j<max_iter and E[-1]>=min_mse:
    
        if func_offset==0.0:
            dW=2*B.T@(D@v)
        else:
            dW=2*B.T@psccost.dfunc_vec(v,func_offset)@D@psccost.func_vec(v,func_offset)
        
        if beta>0:
            if cnumeric:
                delta=psccn.ModDeltaW(W) ###
                dK=psccn.d_square_curvature(W,Nc,h=delta*0.5) ###
            else:
                dK=pscca.d_square_curvature(W,Nc)
            
            dW=dW+beta*dK
        
        W=W-Alpha*dW
        
        #Update error
        v=B@W-C
        if func_offset==0.0:
            E.append( (v.T@(D@v)).mean() )
        else:
            E.append( (psccost.func_vec(v,func_offset).T@D@psccost.func_vec(v,func_offset)).mean() )
        j=j+1
        

        
        # Modifying learning rate
        if len(E)>4:
            if round2(E[-1])<=round2(E[-2]) and round2(E[-2])<=round2(E[-3]) :
                Alpha=Alpha*1.01
            elif E[-1]>E[-2]:
                Alpha=Alpha*0.95
        if Alpha>10*alpha:
            Alpha=10*alpha
        if Alpha<alpha/10.0:
            Alpha=alpha/10.0
            
        # Show progress bar
        if show:
            pbar.set_description("err:%10.3E alpha:%10.3E" % (E[-1], Alpha))
            pbar.update(1)
    
    if show:
        pbar.close()
    
    
    return W.reshape((-1,)), E

    
