import numpy as np
from numpy.linalg import norm as Norm

from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverPoly import DPoly
from PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverPoly import DDPoly

def square_curvature0(wn):
    DDP=DDPoly(wn,0);
    DP=DPoly(wn,0);
    return (Norm(np.cross(DP,DDP))**2)/(Norm(DP)**6);
    
def square_curvature1(wn):
    DDP=DDPoly(wn,1);
    DP=DPoly(wn,1);
    return (Norm(np.cross(DP,DDP))**2)/(Norm(DP)**6);
    
def square_curvature(W,Nc=12):
    N=1+int(W.size/Nc);
    
    S=0;
    for i in range(N-1):
        S=S +square_curvature0(W[(i*Nc):(i*Nc+Nc),0]) +square_curvature1(W[(i*Nc):(i*Nc+Nc),0]);
        
    return S/(2.0*(N-1));
