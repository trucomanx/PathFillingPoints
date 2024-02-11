#  

import numpy as np
import math

def to_get_weight_list(Points:list,w0=None,alpha=0.01,max_iter=10000,min_mse=1e-5):
    N = len(Points);
    
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
    
    Nr, Nc = Q00.shape;
    
    if w0==None:
        w0=0.01*np.ones(Nc*(N-1));
    
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
        W=W-alpha*dW;
        
        E.append(np.square(B@W-C).mean());
        
        j=j+1;
    
    return W.reshape((-1,)),E;

class CubicSpline3D():
    def __init__(self,P:list):
        self.N = len(P);
        self.w, self.MSE = to_get_weight_list(P);
        '''
        print("N:")
        print(self.N)
        print("w:")
        print(self.w)
        print("MSE:")
        print(self.MSE);
        '''
        
    def eval(self,t):
        n=int(math.floor(t));
        return self.poly(n,t);
        
    def poly(self,n,t):
        
        nn=n;
        if(n<0):
            nn=0;
        elif(n>=(self.N-1)):
            nn=self.N-2;
        
        Nv=12;
        
        px = self.w[Nv*nn+0] + self.w[Nv*nn+1]*(t-nn) + self.w[Nv*nn+2 ]*(t-nn)**(2) + self.w[Nv*nn+3 ]*(t-nn)**(3);
        py = self.w[Nv*nn+4] + self.w[Nv*nn+5]*(t-nn) + self.w[Nv*nn+6 ]*(t-nn)**(2) + self.w[Nv*nn+7 ]*(t-nn)**(3);
        pz = self.w[Nv*nn+8] + self.w[Nv*nn+9]*(t-nn) + self.w[Nv*nn+10]*(t-nn)**(2) + self.w[Nv*nn+11]*(t-nn)**(3);
        
        return np.array([px,py,pz]);
        
    def get_w(self):
        return self.w;

    
