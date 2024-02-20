import numpy as np
from numpy.linalg import norm as Norm

from PathFillingPoints.Splines.Cubic3DSolverMethod1.Cubic3DSolverPoly import DPoly
from PathFillingPoints.Splines.Cubic3DSolverMethod1.Cubic3DSolverPoly import DDPoly

def square_curvature0(wn):
    DDP=DDPoly(wn,0);
    DP=DPoly(wn,0);
    return (Norm(np.cross(DP,DDP))**2)/(Norm(DP)**6);
    
def square_curvature1(wn):
    DDP=DDPoly(wn,1);
    DP=DPoly(wn,1);
    return (Norm(np.cross(DP,DDP))**2)/(Norm(DP)**6);
