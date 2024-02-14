#  
import sys
import numpy as np

import PathFillingPoints.Splines.Linear3DSolver as linsolver
from numpy.linalg import norm as Norm

Q00=np.array([  [1.0,1.0,1.0,1.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 1.0,1.0,1.0,1.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 1.0,1.0,1.0,1.0]]);

Q01=np.array([  [1.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 1.0,0.0,0.0,0.0]]);

Q10=np.array([  [0.0,1.0,2.0,3.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,1.0,2.0,3.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,1.0,2.0,3.0]]);

Q11=np.array([  [0.0,1.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0]]);

Q20=np.array([  [0.0,0.0,2.0,6.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,2.0,6.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,2.0,6.0]]);

Q21=np.array([  [0.0,0.0,2.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,2.0,0.0, 0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,2.0,0.0]]);
                
    
################################################################################

def DPoly(wn,tn):
    
    dpx = wn[1] + wn[2 ]*2*tn + wn[3 ]*3*tn**(2);
    dpy = wn[5] + wn[6 ]*2*tn + wn[7 ]*3*tn**(2);
    dpz = wn[9] + wn[10]*2*tn + wn[11]*3*tn**(2);
    
    return np.array([dpx,dpy,dpz]);

def DDPoly(wn,tn):
    
    ddpx = wn[2 ]*2 + wn[3 ]*6*tn;
    ddpy = wn[6 ]*2 + wn[7 ]*6*tn;
    ddpz = wn[10]*2 + wn[11]*6*tn;
    
    return np.array([ddpx,ddpy,ddpz]);
################################################################################
def square_curvature0(wn):
    DDP=DDPoly(wn,0);
    DP=DPoly(wn,0);
    return (Norm(np.cross(DP,DDP))**2)/(Norm(DP)**6);
    
def square_curvature1(wn):
    DDP=DDPoly(wn,1);
    DP=DPoly(wn,1);
    return (Norm(np.cross(DP,DDP))**2)/(Norm(DP)**6);
################################################################################  
def d_square_curvature0(wn):
    DDP=DDPoly(wn,0);
    DP=DPoly(wn,0);
    S1= 2*( (Norm(DDP)**2)*(Q11.T@DP)+
            (Norm( DP)**2)*(Q21.T@DDP) 
        )/(Norm(DP)**6);
    
    S2=-2*(  np.dot(DP,DDP)*(Q11.T@DDP+Q21.T@DP)
          )/(Norm(DP)**6);
    
    S3=-6*(  square_curvature0(wn)*(Q11.T@DP)
          )/(Norm(DP)**2);
    
    return S1+S2+S3;

def d_square_curvature1(wn):
    DDP=DDPoly(wn,1);
    DP=DPoly(wn,1);
    S1= 2*( (Norm(DDP)**2)*(Q10.T@DP)+
            (Norm( DP)**2)*(Q20.T@DDP) 
        )/(Norm(DP)**6);
    
    S2=-2*(  np.dot(DP,DDP)*(Q10.T@DDP+Q20.T@DP)
          )/(Norm(DP)**6);
    
    S3=-6*(  square_curvature1(wn)*(Q10.T@DP)
          )/(Norm(DP)**2);
    
    return S1+S2+S3;
    
################################################################################
    
def to_get_cubic3d_weight_list(Points:list,w0=None,alpha=0.01,max_iter=10000,min_mse=1e-5,beta=0.001):
    N = len(Points);
    
    Nr, Nc = Q00.shape;
    
    if w0==None:
        w0=linsolver.to_get_linear3d_weight_list(Points);
        w0=linsolver.change_order_of_weight_list(w0,to_order=3);
    
    P = np.zeros((Nr*N,Nc*(N-1)));
    
    for l in range(N-1):
        P[Nr*l:(Nr*l+Nr),Nc*l:(Nc*l+Nc)]=Q01;
    P[(Nr*(N-1)):(Nr*N),Nc*(N-2):(Nc*(N-1))]=Q00;
    
    
    Q = np.zeros((3*Nr*(N-2),Nc*(N-1)));
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
    C= c.reshape((-1,1)); 
    W=w0.reshape((-1,1)); 
    
    j=0;
    E=[np.square(B@W-C).mean()];
    while j<max_iter and E[-1]>=min_mse:
        dW=2*B.T@(B@W-C);
        
        if beta>0:
            dK=np.zeros(W.shape);
            for i in range(N-1):
                S=d_square_curvature0(W[(i*Nc):(i*Nc+Nc),0])
                +d_square_curvature1(W[(i*Nc):(i*Nc+Nc),0]);
                dK[(i*Nc):(i*Nc+Nc),0]=S/2.0;
                
            dW=dW+beta*dK;
        
        W=W-alpha*dW;
        
        E.append(np.square(B@W-C).mean());
        
        j=j+1;
    
    return W.reshape((-1,)),E;

    
