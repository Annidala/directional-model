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

#### cptmt des matrices seules
SC37=[0.086,0.003]
dragon=[0.014,0.0007]

#### cptmt des composites
C0_SC37=[0.204,0.204]
C1_SC37=[0.04259,0.04259]
C0_dragon=[0.09,0.09]
C1_dragon=[0.039,0.039]

def model_MR2(element,C1,C2): 
	return 2.*(element-(1./element**2))*(C1+2*C2*((element**2)+(2/element)-3))

def vi(strain,u1,u2,u3,N):
	return np.sqrt((u1**2.*strain**2.+(u2**2.+u3**2.)/(strain))/N) #elongation projetée sur direction ui

def vi_bis(strain,theta, u1,u2,u3):
	### fonction de passage de la déformation dans la base matérielle
	P=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
	mat=np.dot(np.transpose(P), np.dot(strain, P))
	u=np.array([[u1,u2,u3]])
	produit=np.dot(u, mat)
	return np.sqrt(np.dot(produit,np.transpose(produit)))

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


def approx_pade(C,N,vi):
	return C/(N**(3./2.))*vi*(3.-vi**2.)/(1.-vi**2.)

def transfo_ellipse(x,y,a,b,theta):
	return a*np.cos(theta)*x-b*np.sin(theta)*y, a*np.sin(theta)*x+b*np.cos(theta)*y

def approx_boyce(N,vi):
	vivi=vi/np.sqrt(N)
	return vi*(3+9*vivi/5.+297.*vivi**(2)/175.+1539/875.*vivi**3+126117*(vivi**4)/67375.+43733439*vivi**6/21896875.)

def grospate(C,N,vi):
	return C/(N**(3./2.))*(3.-vi**2.)/(1.-vi**2.)


##orientation et poids matrice isotrope
u1m=np.array([math.cos(i*np.pi/6+np.pi/2) for i in range(12)])
u2m=np.array([math.sin(i*np.pi/6+np.pi/2) for i in range(12)])
u3m=np.array([0 for i in range(12)])

### orientations matérielles
n=np.array([1.,np.cos(np.pi/3.),np.cos(np.pi/3),-1.,-np.cos(np.pi/3),-np.cos(np.pi/3)]) #exemple avec seulement 6 directions
t=np.array([0,np.sin(np.pi/3.),-np.sin(np.pi/3),0.,-np.sin(np.pi/3),np.sin(np.pi/3)])
z=np.array([0.,0.,0.,0.,0.,0.])

### orientations star
nstar=np.array([0.,np.cos(np.pi/6.),np.cos(np.pi/6.),0.,-np.cos(np.pi/6.),-np.cos(np.pi/6.)]) #exemple avec seulement 6 directions
tstar=np.array([1.,np.sin(np.pi/6.),-np.sin(np.pi/6.),-1.,-np.sin(np.pi/6.),np.sin(np.pi/6.)])
zstar=np.array([0.,0.,0.,0.,0.,0.])


def err_matrice(p,stress_1, strain, u1m=u1m, u2m=u2m, u3m=u3m):
	func_sens1=np.zeros_like(stress_1)
	func_sens2=np.zeros_like(stress_1)
	for j in range(len(strain)):
		mat_strain=np.array([[strain[j],0,0],[0,1/np.sqrt(strain[j]),0],[0,0,1/np.sqrt(strain[j])]])
		###matrice : elastomere isotrope
		Wm1=[1/12.*p[0]*np.sqrt(p[1])*(1/vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))*(strain[j]*u1m[i]**2.-(strain[j])**(-2.)*u2m[i]**2.)*(approx_boyce(p[1],vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i])/np.sqrt(p[1]))) for i in range(len(u1m))]
		Wm2=[1/12.*p[0]*np.sqrt(p[1])*(1/vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i]))*(strain[j]*u2m[i]**2.-(strain[j])**(-2.)*u1m[i]**2.)*(approx_boyce(p[1],vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i])/np.sqrt(p[1]))) for i in range(len(u1m))]
		#Wm1=[1/12.*(strain[j]*u1m[i]**2.-(strain[j])**(-2.)*u2m[i]**2.)*(
			#1/vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))*(approx_boyce(p[0],vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))) for i in range(len(u1m))]
		#Wm2=[1/12.*np.sqrt(p[0])*(strain[j]*u2m[i]**2.-(strain[j])**(-2.)*u1m[i]**2.)*(
			#1/vi_bis(strain[j],np.pi/2,u1m[i],u2m[i],u3m[i]))*(approx_boyce(p[0],vi_bis(strain[j],np.pi/2,u1m[i],u2m[i],u3m[i]))) for i in range(len(u1m))]
		func_sens1[j]=sum(Wm1)
		func_sens2[j]=sum(Wm2)
	res_sens1=stress_1-func_sens1
	res_sens2=stress_1-func_sens2

	return sum(res_sens1*res_sens1+res_sens2*res_sens2)

#C0_SC37=[0.2,0.2]
#C1_SC37=[0.043,0.043]
#C0_dragon=[0.09,0.09]
#C1_dragon=[0.039,0.039]

plt.ion()
### deformation
strain_matrice = np.arange(1.0,2.,0.01)
strain=np.arange(1.0,1.7,0.01)
### matrice
stress_matrice_dragon=model_MR2(strain_matrice,0.014,0.0007)
stress_matrice_SC37=model_MR2(strain_matrice,0.086,0.003)
### composite
stress_sens1 = model_MR2(strain, C0_dragon[0],C1_dragon[0])
stress_sens2 = model_MR2(strain, C0_dragon[1],C1_dragon[1])

###identif paramètre matrices A et B
bounds_matrix=((0,None),(0,None))
p0=[5,5]

P_dragon=opt.minimize(err_matrice,p0,args=(stress_matrice_dragon, strain_matrice,u1m,u2m,u3m),method='SLSQP')
coeff_dragon=P_dragon.x
P_SC37=opt.minimize(err_matrice,p0,args=(stress_matrice_SC37, strain_matrice,u1m,u2m,u3m),method='SLSQP')
coeff_SC37=P_SC37.x

dragon=np.zeros_like(strain_matrice)
SC37=np.zeros_like(strain_matrice)
for j in range(len(strain_matrice)):
	mat_strain=np.array([[strain_matrice[j],0,0],[0,1/np.sqrt(strain_matrice[j]),0],[0,0,1/np.sqrt(strain_matrice[j])]])
	Wm1=[1/12.*coeff_dragon[0]*np.sqrt(coeff_dragon[1])*(1/vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))*(strain_matrice[j]*u1m[i]**2.-(strain_matrice[j])**(-2.)*u2m[i]**2.)*(
		approx_boyce(coeff_dragon[1],vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i])/np.sqrt(coeff_dragon[1]))) for i in range(len(u1m))]
	Wm2=[1/12.*coeff_SC37[0]*np.sqrt(coeff_SC37[1])*(1/vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))*(strain_matrice[j]*u1m[i]**2.-(strain_matrice[j])**(-2.)*u2m[i]**2.)*(
		approx_boyce(coeff_SC37[1],vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i])/np.sqrt(coeff_SC37[1]))) for i in range(len(u1m))]
	dragon[j]=sum(Wm1)
	SC37[j]=sum(Wm2)

plt.figure()
plt.plot(strain_matrice, stress_matrice_SC37,'b',label='matrice A expe');plt.plot(strain_matrice, SC37, 'b--',linewidth=3, label='matrice A model')
plt.plot(strain_matrice, stress_matrice_dragon,'r',label='matrice B expe');plt.plot(strain_matrice, dragon, 'r--', linewidth=3,label='matrice B model')
plt.ylim(ymin=0);plt.xlim(xmin=1)
plt.xlabel('Strain (mm/mm)');plt.ylabel('Stress (MPa)')
plt.legend(loc=2);plt.title('Identification matrice');plt.grid()

def err_wifix(p,stress_1, stress_2, strain, C, N, u1m=u1m, u2m=u2m, u3m=u3m, n=n, t=t, u3=z, nstar=nstar, tstar=tstar, zstar=zstar):
	phi=p[4]
	func_sens1=np.zeros_like(stress_1)
	func_sens2=np.zeros_like(stress_1)

	wi=np.array([p[0],p[1],p[1],p[0],p[1],p[1]])
	N_mat=np.array([p[2],p[3],p[3],p[2],p[3],p[3]])
	
	for j in range(len(strain)):
		mat_strain=np.array([[strain[j],0,0],[0,1/np.sqrt(strain[j]),0],[0,0,1/np.sqrt(strain[j])]])
		###matrice : elastomere isotrope
		Wm1=[(1/12.)*C*np.sqrt(N)*(1/vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))*(strain[j]*u1m[i]**2.-(strain[j])**(-2.)*u2m[i]**2.)*(
			approx_boyce(N,vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i])/np.sqrt(N))) for i in range(len(u1m))]
		Wm2=[(1/12.)*C*np.sqrt(N)*(1/vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i]))*(strain[j]*u2m[i]**2.-(strain[j])**(-2.)*u1m[i]**2.)*(
			approx_boyce(N,vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i])/np.sqrt(N))) for i in range(len(u1m))]
		### tricot imprégné sens 1 
		Wt1=[(1/12.)*wi[i]*np.sqrt(N_mat[i])*(1/vi_bis(mat_strain,0,n[i],t[i],z[i]))*(strain[j]*n[i]**2.-(strain[j])**(-2.)*t[i]**2.)*(
			approx_boyce(N_mat[i],vi_bis(mat_strain,0,n[i],t[i],z[i])/np.sqrt(N_mat[i]))) for i in range(len(n))]
		Wl1=[(1/12.)*C*np.sqrt(N)*(1/vi_bis(mat_strain,0,nstar[i],tstar[i],zstar[i]))*(strain[j]*nstar[i]**2.-(strain[j])**(-2.)*tstar[i]**2.)*(
			approx_boyce(N,vi_bis(mat_strain,0,nstar[i],tstar[i],zstar[i])/np.sqrt(N))) for i in range(len(nstar))]
		### tricot imprégné sens 2
		Wt2=[(1/12.)*wi[i]*np.sqrt(N_mat[i])*(1/vi_bis(mat_strain,np.pi/2,n[i],t[i],z[i]))*(strain[j]*t[i]**2.-(strain[j])**(-2.)*n[i]**2.)*(
			approx_boyce(N_mat[i],vi_bis(mat_strain,np.pi/2,n[i],t[i],z[i])/np.sqrt(N_mat[i]))) for i in range(len(n))]
		Wl2=[(1/12.)*C*np.sqrt(N)*(1/vi_bis(mat_strain,np.pi/2,nstar[i],tstar[i],zstar[i]))*(strain[j]*tstar[i]**2.-(strain[j])**(-2.)*nstar[i]**2.)*(
			approx_boyce(N,vi_bis(mat_strain,np.pi/2,nstar[i],tstar[i],zstar[i])/np.sqrt(N))) for i in range(len(nstar))]
		func_sens1[j]=phi*(sum(Wt1)+sum(Wl1))+(1-phi)*sum(Wm1)
		func_sens2[j]=phi*(sum(Wt2)+sum(Wl2))+(1-phi)*sum(Wm2)
	
	res_sens1=stress_1-func_sens1+func_sens1[0]
	res_sens2=stress_2-func_sens2+func_sens2[0]
	return sum(res_sens1*res_sens1+res_sens2*res_sens2)

#cons = (
	##{'type' : 'ineq','fun' : lambda x : np.array(x[1]-2*x[0]/3.)},
	##{'type' : 'eq','fun' : lambda x : np.array(2*x[0]+4*x[1]-0.5)},
	##{'type' : 'ineq','fun' : lambda x : np.array(x[1]-0.05)}
	#)

bnds = ((0,None), (0.01,None), (0,None),(0,None),(0,1))
p0=[5,5,5,5,0.1] # Cc, Nc, Ce, Ne, Fc, Fe

P=opt.minimize(err_wifix,p0,args=(stress_sens1, stress_sens2,strain,coeff_dragon[0],coeff_dragon[1],u1m,u2m,u3m,n,t,z,nstar,tstar,zstar),method='SLSQP',bounds=bnds)
coeff=P.x

wi=np.array([P.x[0],P.x[1],P.x[1],P.x[0],P.x[1],P.x[1]])
N_mat=np.array([P.x[2],P.x[3],P.x[3],P.x[2],P.x[3],P.x[3]])

estim=np.zeros_like(strain)
estim_trame=np.zeros_like(strain)
tricot_t=np.zeros_like(strain)
tricot_c=np.zeros_like(strain)

C,N=coeff_dragon[0],coeff_dragon[1]

###modelisation composite dragon
for j in range(len(strain)):
	phi=P.x[4]
	## tenseur de def
	mat_strain=np.array([[strain[j],0,0],[0,1/np.sqrt(strain[j]),0],[0,0,1/np.sqrt(strain[j])]])
	###matrice : elastomere isotrope
	Wm1=[(1/12.)*C*np.sqrt(N)*(1/vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))*(strain[j]*u1m[i]**2.-(strain[j])**(-2.)*u2m[i]**2.)*(
		approx_boyce(N,vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i])/np.sqrt(N))) for i in range(len(u1m))]
	Wm2=[(1/12.)*C*np.sqrt(N)*(1/vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i]))*(strain[j]*u2m[i]**2.-(strain[j])**(-2.)*u1m[i]**2.)*(
			approx_boyce(N,vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i])/np.sqrt(N))) for i in range(len(u1m))]
	### sens 1 
	Wt1=[(1/12.)*wi[i]*np.sqrt(N_mat[i])*(1/vi_bis(mat_strain,0,n[i],t[i],z[i]))*(strain[j]*n[i]**2.-(strain[j])**(-2.)*t[i]**2.)*(
		approx_boyce(N_mat[i],vi_bis(mat_strain,0,n[i],t[i],z[i])/np.sqrt(N_mat[i]))) for i in range(len(n))]
	Wl1=[(1./12)*C*np.sqrt(N)*(1/vi_bis(mat_strain,0,nstar[i],tstar[i],zstar[i]))*(strain[j]*nstar[i]**2.-(strain[j])**(-2.)*tstar[i]**2.)*(
		approx_boyce(N,vi_bis(mat_strain,0,nstar[i],tstar[i],zstar[i])/np.sqrt(N))) for i in range(len(nstar))]
	### sens 2
	Wt2=[(1/12.)*wi[i]*np.sqrt(N_mat[i])*(1/vi_bis(mat_strain,np.pi/2,n[i],t[i],z[i]))*(strain[j]*t[i]**2.-(strain[j])**(-2.)*n[i]**2.)*(
		approx_boyce(N_mat[i],vi_bis(mat_strain,np.pi/2,n[i],t[i],z[i])/np.sqrt(N_mat[i]))) for i in range(len(n))]
	Wl2=[(1./12)*C*np.sqrt(N)*(1/vi_bis(mat_strain,np.pi/2,nstar[i],tstar[i],zstar[i]))*(strain[j]*tstar[i]**2.-(strain[j])**(-2.)*nstar[i]**2.)*(
		approx_boyce(N,vi_bis(mat_strain,np.pi/2,nstar[i],tstar[i],zstar[i])/np.sqrt(N))) for i in range(len(nstar))]
	estim[j]=phi*(sum(Wt1)+sum(Wl1))+(1-phi)*sum(Wm1)
	estim_trame[j]=phi*(sum(Wt2)+sum(Wl2))+(1-phi)*sum(Wm2)
	tricot_t[j]=sum(Wt1)
	tricot_c[j]=sum(Wl1)

plt.figure()
#plt.plot(strain,stress_sens1,'r',label='Exp: weft direction')
#plt.plot(strain,estim-estim[0],'r--',linewidth=2,label='Model: weft')
#plt.plot(strain, tricot,'k')
#plt.plot(strain, polym,'g')
plt.plot(strain, stress_sens2,'b', label='Exp: warp direction')
plt.plot(strain,estim_trame-estim_trame[0],'b--',linewidth=2,label='Model: warp')
#plt.plot(strain2,estim2,'b',label='Modelisation')
plt.grid();plt.legend(loc=2,prop={'family': 'serif', 'size':15});plt.ylabel('Stress (MPa)');plt.xlabel('Strain (mm/mm)');
plt.title('Optimisation du modele directionnel\nsur le composite tricot matrice B')
model_compBc=np.zeros_like(strain)
model_compBt=np.zeros_like(strain)


C2,N2=coeff_SC37[0],coeff_SC37[1]
###modelisation composite SC37
for j in range(len(strain)):
	phi=P.x[4]
	## tenseur de def
	mat_strain=np.array([[strain[j],0,0],[0,1/np.sqrt(strain[j]),0],[0,0,1/np.sqrt(strain[j])]])
	###matrice : elastomere isotrope
	Wm1=[(1/12.)*C2*np.sqrt(N2)*(1/vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i]))*(strain[j]*u1m[i]**2.-(strain[j])**(-2.)*u2m[i]**2.)*(
		approx_boyce(N2,vi_bis(mat_strain,0,u1m[i],u2m[i],u3m[i])/np.sqrt(N2))) for i in range(len(u1m))]
	Wm2=[(1/12.)*C2*np.sqrt(N2)*(1/vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i]))*(strain[j]*u2m[i]**2.-(strain[j])**(-2.)*u1m[i]**2.)*(
			approx_boyce(N2,vi_bis(mat_strain,np.pi/2,u1m[i],u2m[i],u3m[i])/np.sqrt(N2))) for i in range(len(u1m))]	
	### sens 1 
	Wt1=[(1/12.)*wi[i]*np.sqrt(N_mat[i])*(1/vi_bis(mat_strain,0,n[i],t[i],z[i]))*(strain[j]*n[i]**2.-(strain[j])**(-2.)*t[i]**2.)*(
		approx_boyce(N_mat[i],vi_bis(mat_strain,0,n[i],t[i],z[i])/np.sqrt(N_mat[i]))) for i in range(len(n))]
	Wl1=[(1/12.)*C2*np.sqrt(N2)*(strain[j]*nstar[i]**2.-(strain[j])**(-2.)*tstar[i]**2.)*(
		approx_boyce(N2,vi_bis(mat_strain,0,nstar[i],tstar[i],zstar[i])/np.sqrt(N2))) for i in range(len(nstar))]
	### sens 2
	Wt2=[(1/12.)*wi[i]*np.sqrt(N_mat[i])*(1/vi_bis(mat_strain,np.pi/2,n[i],t[i],z[i]))*(strain[j]*t[i]**2.-(strain[j])**(-2.)*n[i]**2.)*(
		approx_boyce(N_mat[i],vi_bis(mat_strain,np.pi/2,n[i],t[i],z[i])/np.sqrt(N_mat[i]))) for i in range(len(n))]
	Wl2=[(1/12.)*C2*np.sqrt(N2)*(strain[j]*tstar[i]**2.-(strain[j])**(-2.)*nstar[i]**2.)*(
		approx_boyce(N2,vi_bis(mat_strain,np.pi/2,nstar[i],tstar[i],zstar[i])/np.sqrt(N2))) for i in range(len(nstar))]
	model_compBc[j]=phi*(sum(Wt1)+sum(Wl1))+(1-phi)*sum(Wm1)
	model_compBt[j]=phi*(sum(Wt2)+sum(Wl2))+(1-phi)*sum(Wm1)


plt.figure()
#plt.plot(strain, model_MR2(strain, C0_SC37[0],C1_SC37[0]), 'r', label='Exp: weft direction')
#plt.plot(strain,model_compBc-model_compBc[0] ,'r--',label='model: weft')

#plt.figure()
plt.plot(strain, model_MR2(strain, C0_SC37[1],C1_SC37[1]), 'b', label = 'Exp: Warp direction + std')
plt.plot(strain,model_compBt-model_compBt[0] ,'b--', label='model: warp')




#plt.figure()
plt.plot(strain, abs((model_MR2(strain, C0_SC37[0],C1_SC37[0])-(model_compBc-model_compBc[0]))/model_MR2(strain, C0_SC37[0],C1_SC37[0])), 'r',label=r'rel error ($\%$)')
#plt.show()
##np.savetxt('coeff_mod_direct_E1582_dragon.txt', coeff)       
plt.xlabel('Strain');plt.ylabel('Stress (Mpa)')
plt.legend(loc=2); plt.grid()
plt.title('Injection du cptmt tricot impregB dans la matrice A\nComparaison aux valeurs expe');