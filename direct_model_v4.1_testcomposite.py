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


def _toTensor(t11, t12, t22, t33=None,  t13=0, t23=0): 
	''' 
	express the strain/stress tensor at instant t
	'''
	if t33: ### 3D case
		P=np.array([[t11, t12,t13],[t12,t22,t23],[t13,t23,t33]])
	else: ### plane stress
		P=np.array([[t11,t12],[t12,t22]])
	return P


def _toBase(tensor, theta, case=False):  
	''' 
	in plane case, change of the configuration reference : from the sollicitation reference to the material reference
	theta in radian
	'''
	if case==False:
		P=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
		Tinbase=np.dot(np.transpose(P),np.dot(tensor,P))
	else:
		P=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
		Tinbase=np.dot(np.transpose(P),np.dot(tensor,P))
	return Tinbase


def _approxLangevin(x, dl_type='Pade'):  
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


def _computeModeledDerivedEnergy(strainO,theta,u,p):
	'''
	compute the modeled stress part coming from the derivation of the energy expression. We do not add here the hydrostatic pressure influence
	(caution : the HP is mandatory to ensure a free stress state in non-deformed configuration)
	needs revision ..... in order to include the HP. 
	This function returns the transformation tensor, as well as the inverse transpose transfo tensor, and the partial modeled stress computed at time t
	'''
	#strain=_toBase(strainO, theta)
	invstrain=np.transpose(np.linalg.inv(strainO))
	sig=sum([p[0]*p[1]*_approxLangevin(_computeDirectionStrain(strainO,u[:,i])/np.sqrt(p[1]),'Pade')*_computeDeriveDirectionStrain(strainO, u[:,i]) for i in range(u.shape[1])])
	return invstrain, sig


def _createDirectionVector(Nbfiber,theta0, dthet):
	'''
	return an array of 2 rows and Nbfiber column containing the direction in 2D. 
	It might be easy to tune it into a 3d direction vector.
	'''
	um=[0,0]
	um=np.array([um])
	for i in Nbfiber:
		ui=[np.cos(i*dthet+theta0),np.sin(i*dthet+theta0)]
		um=np.concatenate((um,np.array([ui])))
	um=np.transpose(um[1::,::])
	return um


def test_diag(a):
	'''
	return the diagonal index
	'''
	for i in range(len(a[0])):
		if a[0][i]==a[1][i]:
			return i



def _evaluateModel(p, strain_,direction_, u):
	'''
	compute the stress with the model parameters
	'''
	contrainte_mod=np.zeros_like(strain_)
	contrainte_mod2=np.zeros_like(strain_)

	for j in range(1,len(strain_)):
		straing=_toTensor(strain_[j], 0, 1/np.sqrt(strain_[j])) ## def of strain tens at j
		invstrain=np.linalg.inv(straing)	
		### estimation of the stress seen by the material when sollicitation in direction 0°
		strain=_toBase(straing, direction_)
		
		invstrain, sig_11=_computeModeledDerivedEnergy(strain, direction_, u, p)
		a=np.where(strain<1);index=test_diag(a) ## testing the direction of the unsollicited direction

		press=sig_11[a[0][index],a[0][index]]/invstrain[a[0][index], a[0][index]] ## hydrostatic pression determination
		sig0=(sig_11-press*invstrain)
		#print sig0
		
		b=np.where(sig0!=0.)
		ind=test_diag(b) ## direction of sollicitation (eg where the stress answer is not null)
		#print ind
		contrainte_mod[j]=sig0[0,0]
		contrainte_mod2[j]=sig0[1,1]

	return contrainte_mod,contrainte_mod2

strain_=np.arange(1.0,1.7,0.01)
stress_=model_MR2(strain_,0.014,0.0007)

## Orientation distribution of the matrix
Nbdirm=np.arange(0,12,1)
theta0=0
dthet=np.pi/6
umat=_createDirectionVector(Nbdirm,theta0,dthet) 

### Orientation distribution in the knitted textile
Nbdir=np.arange(0,11, 2)
Nbdirint=np.arange(1,12, 2)
theta0=0
dthet=np.pi/6
utext=_createDirectionVector(Nbdir,theta0,dthet) 
umint=_createDirectionVector(Nbdirint, theta0, dthet)


#def err_matrice(p,stress_, strain=strain_, u=u):
	#res_sens1=np.zeros_like(stress_)
	#res_sens2=np.zeros_like(stress_)
	#for j in range(len(strain)):
		#strain=_toTensor(strain_[j], 0, 1/np.sqrt(strain_[j])) ## def of strain tens at j
		#invstrain=np.linalg.inv(strain)
		#stress=_toTensor(stress_[j],0,0)  ### def of stress tens at instant j
		
		### estimation of the stress seen by the material when sollicitation in direction 0°
		#stress0=_toBase(stress, 0)
		#strain0, invstrain0, sig0_11=_computeModeledDerivedEnergy(strain, np.pi/2, u, [p[0],p[1]])
		#press=sig0_11[1,1]/invstrain0[1,1] ## hydrostatic pression determination
		#sig0=(sig0_11-press*invstrain0)[0,0]

		#### estimation of the stress seen by the material when sollicitation in direction 90°
		#stress90=_toBase(stress, np.pi/2.)
		#strain90, invstrain90, sig90_11=_computeModeledDerivedEnergy(strain,np.pi/2., u, [p[0],p[1]])
		#press=sig90_11[0,0]/invstrain90[0,0] ## hydrostatic pression determination
		#sig90=(sig90_11-press*invstrain90)[1,1]
		
		#res_sens1[j]=stress0[0,0]-sig0
		#res_sens2[j]=stress90[1,1]-sig90
	#return sum(res_sens1*res_sens1+res_sens2*res_sens2)


#stress_dragon=model_MR2(strain_,0.014,0.0007)
#bounds_matrix=((0,None),(0,None))
#p0=[0.01,5.]

#P_dragon=opt.minimize(err_matrice,p0,args=(stress_dragon, strain_,u),method='SLSQP',bounds=bounds_matrix)
#coeff_dragon=P_dragon.x
#pdrag=coeff_dragon
#print 'coeff dragon : ', pdrag
#contrainte_mod=np.zeros_like(strain_)

####matrice dragon
#for j in range(len(strain_)):
	#strain=_toTensor(strain_[j], 0, 1/np.sqrt(strain_[j])) ## def of strain tens at j
	#invstrain=np.linalg.inv(strain)
	#stress=_toTensor(stress_[j],0,0)  ### def of stress tens at instant j
	#### estimation of the stress seen by the material when sollicitation in direction 
	#stress0=_toBase(stress, 0)
	#strain0, invstrain0, sig0_11=_computeModeledDerivedEnergy(strain, 0, u, pdrag)
	#press=sig0_11[1,1]/invstrain0[1,1] ## hydrostatic pression determination
	#sig0=(sig0_11-press*invstrain0)[0,0]
	#contrainte_mod[j]=sig0

#plt.figure()
#plt.plot(strain_, stress_dragon,'b', linewidth=2);plt.plot(strain_, contrainte_mod,'r--', linewidth=2);
#plt.plot(strain_, (stress_dragon-contrainte_mod)/stress_dragon, 'g', linewidth=3)
#plt.grid();plt.title('Dragon')

#stress_SC=model_MR2(strain_,0.086,0.003)
#bounds_matrix=((0.,None),(0.,None))
#cons = (
	##{'type' : 'ineq','fun' : lambda x : np.array(x[1]-2*x[0]/3.)},
	#{'type' : 'eq','fun' : lambda x : np.array(x[0]-pdrag[0])}
	##{'type' : 'ineq','fun' : lambda x : np.array(x[1]-0.05)}
	#)
#p0=[0.1,10.]
#P_sc37=opt.minimize(err_matrice,p0,args=(stress_SC, strain_,u),method='SLSQP',bounds=bounds_matrix)
#coeff_sc37=P_sc37.x
#p=coeff_sc37
#print 'coeff SC37 : ', p
#contrainte_mod=np.zeros_like(strain_)
#for j in range(len(strain_)):
	#strain=_toTensor(strain_[j], 0, 1/np.sqrt(strain_[j])) ## def of strain tens at j
	#invstrain=np.linalg.inv(strain)	
	#### estimation of the stress seen by the material when sollicitation in direction 0°
	#strain0, invstrain0, sig0_11=_computeModeledDerivedEnergy(strain, 0, u, p)
	#press=sig0_11[1,1]/invstrain0[1,1] ## hydrostatic pression determination
	#sig0=(sig0_11-press*invstrain0)[0,0]
	#contrainte_mod[j]=sig0

#plt.figure()
#plt.plot(strain_, stress_SC,'b', linewidth=3);plt.plot(strain_, contrainte_mod,'r--', linewidth=3);
#plt.plot(strain_, (stress_SC-contrainte_mod)/stress_SC, 'k--', linewidth=3)
#plt.grid();plt.title('SC37')



#def err_matrice(p,strain_, stress_1, stress_2, umat, utext, uint):
	#res_sens1=np.zeros_like(stress_)
	#res_sens2=np.zeros_like(stress_)
	#for j in range(len(strain)):
		#strain=_toTensor(strain_[j], 0, 1/np.sqrt(strain_[j])) ## def of strain tens at j
		#invstrain=np.linalg.inv(strain)
		#stress_dir1=_toTensor(stress_1[j],0,0)  ### def of stress tens at instant j
		#stress_dir2=_toTensor(stress_2[j],0,0)
		
		### estimation of the stress seen by the material when sollicitation in direction 1 (usually 0)
		#stress1=_toBase(stress_dir1, 0)
		
		#strain1, invstrain1, sig1_11=_computeModeledDerivedEnergy(strain, 0, u, [p[0],p[1]])
		#press=sig0_11[1,1]/invstrain0[1,1] ## hydrostatic pression determination
		#sig0=(sig0_11-press*invstrain0)[0,0]

		#### estimation of the stress seen by the material when sollicitation in direction 2 (usually 90)
		#stress2=_toBase(stress_dir2, np.pi/2.)
		#strain2, invstrain2, sig2_11=_computeModeledDerivedEnergy(strain,np.pi/2., u, [p[0],p[1]])
		#press=sig2_11[0,0]/invstrain2[0,0] ## hydrostatic pression determination
		#sig3=(sig2_11-press*invstrain2)[1,1]
		
		#res_sens1[j]=stress1[0,0]-sig1
		#res_sens2[j]=stress2[1,1]-sig2
	#return sum(res_sens1*res_sens1+res_sens2*res_sens2)