#!/usr/bin/python

import sys
sys.path.append('../src')

import extra
import PathFillingPoints.Splines.Linear3D as pfp


P=[[0,0,0],[1,1,1],[1,1,2],[1,2,2]];

spl=pfp.LinearSpline3D(P);

extra.plot_data(P,spl,L=64);


