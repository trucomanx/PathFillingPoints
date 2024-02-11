#  

import numpy as np
import math

def to_get_weight_list(Points:list):
    N = len(Points);
    
    Q00=np.array([  [1.0,1.0,0.0,0.0,0.0,0.0],
                    [0.0,0.0,1.0,1.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0,1.0,1.0]]);
    
    Q01=np.array([  [1.0,0.0,0.0,0.0,0.0,0.0],
                    [0.0,0.0,1.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0,1.0,0.0]]);
    
    #####################
    # Formando la matrix C
    Nr, Nc = Q00.shape;
    
    P = np.zeros((Nr*N,Nc*(N-1)));
    
    for l in range(N-1):
        P[Nr*l:(Nr*l+Nr),Nc*l:(Nc*l+Nc)]=Q01;
    P[(Nr*(N-1)):(Nr*N),Nc*(N-2):(Nc*(N-1))]=Q00;
    
    
    Q = np.zeros((Nr*(N-2),Nc*(N-1)));
    for l in range(N-2):
        Q[Nr*l:(Nr*(l+1)),Nc*l:(Nc*(l+1))    ]= Q00;
        Q[Nr*l:(Nr*(l+1)),Nc*(l+1):(Nc*(l+2))]=-Q01;
    
    B=np.concatenate((P,Q), axis=0);
    
    c=np.zeros(Nr*N + Nr*(N-2));
    
    for n in range(N):
        c[3*n  ]=Points[n][0];
        c[3*n+1]=Points[n][1];
        c[3*n+2]=Points[n][2];
    
    w=np.matmul(np.linalg.inv(B),
                c.reshape((-1,1))
                );
    
    return w.reshape((-1,));

class LinearSpline3D():
    def __init__(self,P:list):
        self.N = len(P);
        self.w = to_get_weight_list(P);
        
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

    
