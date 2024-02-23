#  
import sys
import numpy as np

import PathFillingPoints.Splines.Linear3DSolver as linsolver


from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q00
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q01
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q10
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q11
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q20
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverQ import Q21

from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverPoly import DPoly
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverPoly import DDPoly

import PathFillingPoints.Splines.Cubic3DSolverMethod1.Cubic3DSolverDsc as pscca
import PathFillingPoints.Splines.Cubic3DSolverMethod1.Cubic3DSolverDscNumeric as psccn

from tqdm import tqdm

################################################################################
    
def to_get_cubic3d_weight_list(Points:list,
                                w0=None,
                                alpha=0.01,
                                max_iter=20000,
                                min_mse=1e-5,
                                beta=0.001,
                                weight_pr=1.0,
                                weight_pp=1.0,
                                weight_dpdp=1.0,
                                weight_ddpddp=1.0,
                                cnumeric=False,
                                show=False):
    N = len(Points);
    
    Nr, Nc = Q00.shape;
    
    if w0==None:
        w0=linsolver.to_get_linear3d_weight_list(Points);
        w0=linsolver.change_order_of_weight_list(w0,to_order=3);
    
    P = np.zeros((Nr*N,Nc*(N-1)));
    
    ## Pesos de P
    wp=[weight_pr]*(Nr*N);
    
    for l in range(N-1):
        P[Nr*l:(Nr*l+Nr),Nc*l:(Nc*l+Nc)]=Q01;
    P[(Nr*(N-1)):(Nr*N),Nc*(N-2):(Nc*(N-1))]=Q00;
    
    
    Q = np.zeros((3*Nr*(N-2),Nc*(N-1)));
    
    #Pesos de Q
    wq=( [weight_pp]*Nr + [weight_dpdp]*Nr + [weight_ddpddp]*Nr )*(N-2);
    
    for l in range(N-2):
        Q[ Nr*(3*l+0):Nr*(3*l+1), Nc*(l+0):Nc*(l+1) ] = Q00;
        Q[ Nr*(3*l+0):Nr*(3*l+1), Nc*(l+1):Nc*(l+2) ] =-Q01;
        
        Q[ Nr*(3*l+1):Nr*(3*l+2), Nc*(l+0):Nc*(l+1) ] = Q10;
        Q[ Nr*(3*l+1):Nr*(3*l+2), Nc*(l+1):Nc*(l+2) ] =-Q11;
        
        Q[ Nr*(3*l+2):Nr*(3*l+3), Nc*(l+0):Nc*(l+1) ] = Q20;
        Q[ Nr*(3*l+2):Nr*(3*l+3), Nc*(l+1):Nc*(l+2) ] =-Q21;
    
    B=np.concatenate((P,Q), axis=0);
    
    c=np.zeros(Nr*N + 3*Nr*(N-2));
    
    for n in range(N):
        c[3*n  ]=Points[n][0];
        c[3*n+1]=Points[n][1];
        c[3*n+2]=Points[n][2];
    
    # Calculo de w
    C = c.reshape((-1,1)); 
    W = w0.reshape((-1,1)); 
    D = np.diag(wp+wq);
    
    j=0;
    if cnumeric:
        psccn.ModDeltaW.Wold=np.zeros(W.shape);
    
    if show:
        pbar = tqdm(total=max_iter);
    
    E=[np.square(B@W-C).mean()];
    while j<max_iter and E[-1]>=min_mse:
        dW=2*B.T@(D@(B@W-C));
        
        if beta>0:
            if cnumeric:
                delta=psccn.ModDeltaW(W);###
                dK=psccn.d_square_curvature(W,Nc,h=delta*0.5);###
            else:
                dK=pscca.d_square_curvature(W,Nc);
            
            dW=dW+beta*dK;
        
        W=W-alpha*dW;
        
        e=(B@W-C);
        E.append( (e.T@(D@e)).mean() );
        
        j=j+1;
        
        if show:
            pbar.set_description("err:%10.3E" % E[-1]);
            pbar.update(1);
    
    if show:
        pbar.close();
    
    
    return W.reshape((-1,)),E;

    
