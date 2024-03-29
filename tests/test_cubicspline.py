#!/usr/bin/python

import sys
sys.path.append('../src')

import extra
import PathFillingPoints.Splines.Cubic3D as pfp
import numpy as np


P=[[0,0,0],[1,1,1],[1,1,2],[1,2,2],[2,2,2],[2,2,3],[2,2,4],[2,3,4]];

################################################################################
alpha=0.01;
beta=0.0;
spl=pfp.CubicSpline3D(P,alpha=alpha,beta=beta);

print('   alpha:',alpha)
print('    beta:',beta)
print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

print('min curvature:',np.min(spl.get_curvatures()));
print('max curvature:',np.max(spl.get_curvatures()));
print('\n')

extra.plot_data(P,spl,L=64,title='Sem curvatura e sem cubo');


################################################################################
alpha=0.01;
beta=0.04;
spl=pfp.CubicSpline3D(P,alpha=alpha,beta=beta,show=True);

print('   alpha:',alpha)
print('    beta:',beta)
print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

print('min curvature:',np.min(spl.get_curvatures()));
print('max curvature:',np.max(spl.get_curvatures()));
print('\n')

extra.plot_data(P,spl,L=64,title='Com curvatura e sem cubo');

################################################################################
alpha=0.01;
beta=0.04;
spl=pfp.CubicSpline3D(P,alpha=alpha,beta=beta,func_offset=0.1,show=True);

print('   alpha:',alpha)
print('    beta:',beta)
print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

print('min curvature:',np.min(spl.get_curvatures()));
print('max curvature:',np.max(spl.get_curvatures()));
print('\n')

extra.plot_data(P,spl,L=64,title='Com curvatura e com cubo');

################################################################################
alpha=0.01;
beta=0.04;
spl=pfp.CubicSpline3D(  P,
                        alpha=alpha,
                        beta=beta,
                        weight_pr=0.5,
                        weight_pp=4.0,
                        weight_dpdp=2.0,
                        weight_ddpddp=1.0,
                        show=True);

print('   alpha:',alpha)
print('    beta:',beta)
print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

print('min curvature:',np.min(spl.get_curvatures()));
print('max curvature:',np.max(spl.get_curvatures()));
print('\n')

extra.plot_data(P,spl,L=64,title='Com curvatura, sem cubo e com pesos');


################################################################################
alpha=0.01;
beta=0.04;
spl=pfp.CubicSpline3D(  P,
                        alpha=alpha,
                        beta=beta,
                        weight_pr=2.0,
                        weight_pp=4.0,
                        weight_dpdp=2.0,
                        weight_ddpddp=1.0,
                        func_offset=0.1,
                        show=True);

print('   alpha:',alpha)
print('    beta:',beta)
print('  MSE[0]:',spl.get_mse()[0])
print(' MSE[-1]:',spl.get_mse()[-1])
print('len(MSE):',len(spl.get_mse()))

print('min curvature:',np.min(spl.get_curvatures()));
print('max curvature:',np.max(spl.get_curvatures()));
print('\n')

extra.plot_data(P,spl,L=64,title='Com curvatura, com cubo e com pesos');

################################################################################

import PathFillingPoints.Splines.Cubic3DSolverTools.Cubic3DSolverCost as psccost

cost=psccost.cost_func( P,
                        spl.get_w(),
                        beta=beta,
                        weight_pr=2.0,
                        weight_pp=4.0,
                        weight_dpdp=2.0,
                        weight_ddpddp=1.0,
                        func_offset=0.1);
print('cost:', cost);
