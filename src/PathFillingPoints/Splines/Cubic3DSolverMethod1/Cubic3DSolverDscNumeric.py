import numpy as np

from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverSc import square_curvature 

def ModDeltaW(W):
    delta=np.linalg.norm(W-ModDeltaW.Wold);
    ModDeltaW.Wold=W.copy();
    
    return delta;

def d_square_curvature(W1,Nc,h=0.01):
    dK=np.zeros(W1.shape);
    
    # Calculo E1
    E1=square_curvature(W1,Nc);
    
    # Calculo E0
    for l in range(W1.size):
        W0=W1.copy();
        W0[l,0]=W0[l,0]-h;
        
        E0=square_curvature(W0,Nc);
        
        dK[l,0]=(E1-E0)/h;
        
    return dK;
