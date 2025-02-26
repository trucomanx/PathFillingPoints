#  

import numpy as np
import math
import PathFillingPoints.Splines.Cubic3DSolverMethod1.Cubic3DSolver as solver

class CubicSpline3D():
    """
    A class to compute a 3D cubic spline interpolation.
    
    Attributes:
        N (int): Number of input points.
        w (numpy): Matrix column vector of computed cubic spline weights.
        MSE (float): Mean squared error of the spline fitting.
    """
    
    def __init__(self,  Points:list,
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
                        show=False):
        """
        Initializes the cubic spline with given parameters and computes the weights.
        
        Args:
            Points (list): List of 3D points to interpolate.
            w0 (optional): Initial weight values.
            alpha (float): Learning rate for optimization.
            max_iter (int): Maximum number of iterations for optimization.
            min_mse (float): Minimum mean squared error for convergence.
            beta (float): Regularization parameter.
            weight_pr (float): Weight for primary term.
            weight_pp (float): Weight for secondary term.
            weight_dpdp (float): Weight for first derivative term.
            weight_ddpddp (float): Weight for second derivative term.
            func_offset (float): Function offset.
            cnumeric (bool): Flag to use numerical computation.
            show (bool): Flag to display debugging information.
        """
        self.N = len(Points)
        
        weight_tot = weight_pr + weight_pp + weight_dpdp + weight_ddpddp
        
        self.w, self.MSE = solver.to_get_cubic3d_weight_list(
            Points,
            w0=w0,
            alpha=alpha,
            max_iter=max_iter,
            min_mse=min_mse,
            beta=beta,
            weight_pr=(weight_pr/weight_tot),
            weight_pp=(weight_pp/weight_tot),
            weight_dpdp=(weight_dpdp/weight_tot),
            weight_ddpddp=(weight_ddpddp/weight_tot),
            func_offset=func_offset,
            cnumeric=cnumeric,
            show=show
        )
    
    def eval(self, t):
        """Evaluates the cubic spline at a given parameter t."""
        n = int(math.floor(t))
        return self.poly(n, t)
    
    def poly(self, n, t):
        """Computes the cubic polynomial at a given segment index n and parameter t."""
        nn=n;
        if(n<0):
            nn=0;
        elif(n>=(self.N-1)):
            nn=self.N-2;
        
        Nv = 12
        
        px = self.w[Nv*nn+0] + self.w[Nv*nn+1]*(t-nn) + self.w[Nv*nn+2]*(t-nn)**2 + self.w[Nv*nn+3]*(t-nn)**3
        py = self.w[Nv*nn+4] + self.w[Nv*nn+5]*(t-nn) + self.w[Nv*nn+6]*(t-nn)**2 + self.w[Nv*nn+7]*(t-nn)**3
        pz = self.w[Nv*nn+8] + self.w[Nv*nn+9]*(t-nn) + self.w[Nv*nn+10]*(t-nn)**2 + self.w[Nv*nn+11]*(t-nn)**3
        
        return np.array([px, py, pz])
    
    def dpoly(self, n, t):
        """Computes the first derivative of the cubic polynomial at a given segment index n and parameter t."""
        nn=n;
        if(n<0):
            nn=0;
        elif(n>=(self.N-1)):
            nn=self.N-2;
        
        Nv = 12
        
        dpx = self.w[Nv*nn+1] + self.w[Nv*nn+2]*2*(t-nn) + self.w[Nv*nn+3]*3*(t-nn)**2
        dpy = self.w[Nv*nn+5] + self.w[Nv*nn+6]*2*(t-nn) + self.w[Nv*nn+7]*3*(t-nn)**2
        dpz = self.w[Nv*nn+9] + self.w[Nv*nn+10]*2*(t-nn) + self.w[Nv*nn+11]*3*(t-nn)**2
        
        return np.array([dpx, dpy, dpz])
    
    def ddpoly(self, n, t):
        """Computes the second derivative of the cubic polynomial at a given segment index n and parameter t."""
        nn=n;
        if(n<0):
            nn=0;
        elif(n>=(self.N-1)):
            nn=self.N-2
        
        Nv = 12
        
        ddpx = self.w[Nv*nn+2]*2 + self.w[Nv*nn+3]*6*(t-nn)
        ddpy = self.w[Nv*nn+6]*2 + self.w[Nv*nn+7]*6*(t-nn)
        ddpz = self.w[Nv*nn+10]*2 + self.w[Nv*nn+11]*6*(t-nn)
        
        return np.array([ddpx, ddpy, ddpz])
    
    def curvature(self, n, t):
        """Computes the curvature of the spline at a given segment index n and parameter t."""
        dp = self.dpoly(n, t)
        ddp = self.ddpoly(n, t)
        
        k=np.linalg.norm(np.cross(dp, ddp)) / np.linalg.norm(dp)**3
        
        return k;
    def get_curvatures(self):
        """Computes and returns a list of the curvatures at (begin,end) along the piece spline."""
        dat = []
        for n in range(self.N - 1):
            dat.append((self.curvature(n, n),self.curvature(n, n + 1)))
        
        return dat
    
    def get_w(self):
        """Returns the computed weight list."""
        return self.w
    
    def get_mse(self):
        """Returns the mean squared error of the spline fitting."""
        return self.MSE

