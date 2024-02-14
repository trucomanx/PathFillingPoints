#  
import sys
import numpy as np

def to_get_linear3d_weight_list(Points:list):
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


def change_order_of_weight_list(w,to_order=3):
    to_order=int(to_order);
    
    if to_order<1:
        print("to_order<1");
        sys.exit();
        
    L=int(np.size(w)/2);
    
    W=np.zeros(L*(to_order+1));
    
    for l in range(L):
        W[(to_order+1)*l]   = w[2*l];
        W[(to_order+1)*l+1] = w[2*l+1];
    
    return W;
