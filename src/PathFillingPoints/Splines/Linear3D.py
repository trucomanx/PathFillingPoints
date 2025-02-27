#  

import numpy as np
import math
import PathFillingPoints.Splines.Linear3DSolver.Linear3DSolver as solver


class LinearSpline3D():
    def __init__(self,P:list):
        self.N = len(P);
        self.w = solver.to_get_linear3d_weight_list(P);
        
    def eval(self,t):
        n=int(math.floor(t));
        return self.poly(n,t);
        
    def poly(self,n,t):
        
        nn=n;
        if(n<0):
            nn=0;
        elif(n>=(self.N-1)):
            nn=self.N-2;
        
        px=self.w[6*nn+0]+self.w[6*nn+1]*(t-nn);
        py=self.w[6*nn+2]+self.w[6*nn+3]*(t-nn);
        pz=self.w[6*nn+4]+self.w[6*nn+5]*(t-nn);
        
        return np.array([px,py,pz]);
        
    def get_w(self):
        return self.w;
        
    def get_w_cubic_format(self):
        w3=[]
        for i in range(0, len(self.w), 2):
            w3.append(self.w[i])     # coeficiente do termo x^0
            w3.append(self.w[i+1])   # coeficiente do termo x^1
            w3.append(0)         # coeficiente do termo x^2
            w3.append(0)         # coeficiente do termo x^3
        return np.array(w3);

    
