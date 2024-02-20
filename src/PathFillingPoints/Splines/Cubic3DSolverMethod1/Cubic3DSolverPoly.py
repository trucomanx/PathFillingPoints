#!/usr/bin/python

import numpy as np

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

