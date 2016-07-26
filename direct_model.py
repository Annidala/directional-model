# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from matplotlib import rc
import math as math
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':16, 'weight':'bold'})
rc('text', usetex=True)



def _toTensor(t11, t12, t22, t33=None,  t13=0, t23=0): 
	''' 
	express the strain/stress tensor at instant t
	'''
	if t33: ### 3D case
		P=np.array([[t11, t12,t13],[t12,t22,t23],[t13,t23,t33]])
	else: ### plane stress
		P=np.array([[t11,t12],[t12,t22]])
	return P


def _toBase(tensor, theta):  
	''' 
	in plane case, change of the configuration reference : from the sollicitation reference to the material reference
	theta in radian
	'''
	P=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Tinbase=np.dot(np.transpose(P),np.dot(tensor,P))
	return Tinbase


def _approxLangevin (x, dl_type='Pade'):  
	''' 
	approximation of the inverse Langevin function, you get to choose the approximate between exact Pade and Boyce method
	'''
	dl_types=['Pade', 'Boyce']
	if dl_type not in dl_types:
		raise ValueError('Wrong approximate choice, sorry you lose ! Expected one of: %s' %dl_types)
	if dl_type=='Pade':  ### exact Pade approximate
		return x*(3.-x*x)/(1.-x*x)
	if dl_type=='Boyce': ### Taylor-exapnsion approximate
		return x*(3+9*x**2/5.+297.*x**(4)/175.+(1539/875.)*x**6+126117*(x**8)/67375.+43733439*(x**10)/21896875.)


def _computeDirectionStrain(Ftensor, ui):
	'''
	evaluate the strain seen by direction i 
	'''
	produit=np.dot(ui, Ftensor)
	return np.sqrt((np.dot(np.transpose(produit),produit)))


def _computeDeriveDirectionStrain(Ftensor, ui):
	'''
	evaluate the value of the derivative of nu-i with regards to the material strain tensor
	'''
	di=(1/_computeDirectionStrain(Ftensor,ui))*np.dot(Ftensor, np.outer(ui,ui))
	return di


def model_MR2(element,C1,C2): 
	''' 
	stress modeling with mooney rivlin 2nd order in I1
	'''
	return 2.*(element-(1./element**2))*(C1+2*C2*((element**2)+(2/element)-3))


def vi_struct(strain, theta, u1,u2,u3):
	P=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
	mat=np.dot(np.transpose(P), np.dot(strain, P))
	vu=np.zeros_like(u1)
	proj1= np.dot(mat, np.array([u1[1], u2[1],u3[1]]))
	proj2= np.dot(mat, np.array([u1[2], u2[2],u3[2]]))
	
	for i in [1,2,4,5]:
		vu[i]=vi_bis(strain, theta, u1[i], u2[i],u3[i])
	vu[0]=1/2.*np.sqrt(vu[1]**2+vu[2]**2+np.dot(proj1, np.transpose(proj2))+np.dot(proj2, np.transpose(proj1)))
	vu[3]=vu[0]
	
	return vu



### material direction definition
''' should be submitted as a vector of the 3 coordinates of the direction in 3D and 2 in 2D case
'''

u_axe11=np.array([1,0])
u_axe12=np.array([-1,0])
u_axe21=np.array([0,1])
u_axe22=np.array([0,-1])
u_axe31=np.array([np.cos(np.pi/4),np.sin(np.pi/4)])
u_axe32=np.array([-np.cos(np.pi/4),-np.sin(np.pi/4)])
u_axe41=np.array([np.cos(np.pi/4),-np.sin(np.pi/4)])
u_axe42=np.array([-np.cos(np.pi/4),np.sin(np.pi/4)])

u=np.transpose(np.array([u_axe11,u_axe12,u_axe21,u_axe22,u_axe31,u_axe32,u_axe41,u_axe42]))
		   
strain_=np.arange(1.0,1.7,0.01)
stress_=model_MR2(strain_,0.014,0.0007)


#stress_soll=_toTensor(stress,0,0)
#strain_soll=_toTensor(strain,0,1/np.sqrt(strain))

def err_matrice(p,stress=stress_, strain=strain_, u=u):
	res_sens1=np.zeros_like(stress_)
	res_sens2=np.zeros_like(stress_)
	for j in range(len(strain)):
		strain=_toTensor(strain_[j], 0, 1/np.sqrt(strain_[j])) ## def of strain tens at j
		invstrain=np.linalg.inv(strain)
		stress=_toTensor(stress_[j],0,0)  ### def of stress tens at instant j
		
		### estimation of the stress seen by the material when sollicitation in direction 0°
		strain0=_toBase(strain, 0)
		invstrain0=_toBase(invstrain, 0)
		stress0=_toBase(stress, 0)
		sig0_11=sum([(1./u.shape[1])*p[0]*_approxLangevin(_computeDirectionStrain(strain0,u[:,i])/np.sqrt(p[1]),'Pade')*_computeDeriveDirectionStrain(strain0, u[:,i]) for i in range(u.shape[1])])
		press=sig0_11[1,1]/invstrain0[1,1] ## hydrostatic pression determination
		sig0=(sig0_11-press*invstrain0)[0,0]

		### estimation of the stress seen by the material when sollicitation in direction 90°
		strain90=_toBase(strain, np.pi/2.)
		invstrain90=_toBase(invstrain, np.pi/2.)
		stress90=_toBase(stress, np.pi/2.)
		sig90_11=sum([(1./u.shape[1])*p[0]*_approxLangevin(_computeDirectionStrain(strain90,u[:,i])/np.sqrt(p[1]),'Pade')*_computeDeriveDirectionStrain(strain90, u[:,i]) for i in range(u.shape[1])])
		press=sig90_11[0,0]/invstrain90[0,0] ## hydrostatic pression determination
		sig90=(sig90_11-press*invstrain90)[1,1]
		
		res_sens1[j]=stress0[0,0]-sig0
		res_sens2[j]=stress90[1,1]-sig90

	return sum(res_sens1*res_sens1+res_sens2*res_sens2)



bounds_matrix=((0,None),(0,None))
p0=[10,10]

P_dragon=opt.minimize(err_matrice,p0,args=(stress_, strain_,u),method='SLSQP')
coeff_dragon=P_dragon.x
p=coeff_dragon

contrainte_mod=np.zeros_like(strain_)
contrainte_mod2=np.zeros_like(strain_)

for j in range(len(strain_)):
	strain=_toTensor(strain_[j], 0, 1/np.sqrt(strain_[j])) ## def of strain tens at j
	invstrain=np.linalg.inv(strain)
	stress=_toTensor(stress_[j],0,0)  ### def of stress tens at instant j
	
	### estimation of the stress seen by the material when sollicitation in direction 0°
	strain0=_toBase(strain, 0)
	invstrain0=_toBase(invstrain, 0)
	stress0=_toBase(stress, 0)
	sig0_11=sum([(1./u.shape[1])*p[0]*_approxLangevin(_computeDirectionStrain(strain0,u[:,i])/np.sqrt(p[1]),'Pade')*_computeDeriveDirectionStrain(strain0, u[:,i]) for i in range(u.shape[1])])
	press=sig0_11[1,1]/invstrain0[1,1] ## hydrostatic pression determination
	sig0=(sig0_11-press*invstrain0)[0,0]
	contrainte_mod[j]=sig0
	
	invstrain90=_toBase(invstrain, np.pi/2.)
	stress90=_toBase(stress, np.pi/2.)
	sig90_11=sum([(1./u.shape[1])*p[0]*_approxLangevin(_computeDirectionStrain(strain90,u[:,i])/np.sqrt(p[1]),'Pade')*_computeDeriveDirectionStrain(strain90, u[:,i]) for i in range(u.shape[1])])
	press=sig90_11[0,0]/invstrain90[0,0] ## hydrostatic pression determination
	sig90=(sig90_11-press*invstrain90)[1,1]
	contrainte_mod2[j]=sig90

	
plt.plot(strain_, stress_,'b');plt.plot(strain_, contrainte_mod,'r--', linewidth=5);plt.plot(strain_, contrainte_mod2, 'g')