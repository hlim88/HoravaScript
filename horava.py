####################################################################
# May.2020
# Quad grav rhs generator new version
#####################################################################

import dendro
from sympy import *

import numpy as np
###################################################################
# initialize
###################################################################

l1, l2, l3, l4, eta = symbols('lambda[0] lambda[1] lambda[2] lambda[3] eta')
lf0, lf1 = symbols('lambda_f[0] lambda_f[1]')

#QG related constants
a_const = symbols('a_const')
b_const = symbols('b_const')
qg_mass0_sq = symbols('qg_mass0_sq')
qg_mass2_sq = symbols('qg_mass0_sq')

PI = 3.14159265358979323846
kappa = 1/(16*PI)

# Additional parameters for damping term
R0 = symbols('QUADGRAV_ETA_R0')
ep1, ep2 = symbols('QUADGRAV_ETA_POWER[0] QUADGRAV_ETA_POWER[1]')

# declare variables (BSSN vars)
a   = dendro.scalar("alpha", "[pp]")
chi = dendro.scalar("chi", "[pp]")
K   = dendro.scalar("K", "[pp]")

Gt  = dendro.vec3("Gt", "[pp]")
b   = dendro.vec3("beta", "[pp]")
B   = dendro.vec3("B", "[pp]")

gt  = dendro.sym_3x3("gt", "[pp]")
At  = dendro.sym_3x3("At", "[pp]")

Gt_rhs  = dendro.vec3("Gt_rhs", "[pp]")

# Ricci scalar, R 
Rsc = dendro.scalar("Rsc", "[pp]")
# Aux Ricci scalar, R^
Rsch = dendro.scalar("Rsch", "[pp]")


#TODO : Clear up for documentation
# Ricci tensor, R_ab
#Rab = dendro.sym_3x3("Rab", "[pp]")
# Aux Ricci tensor, V_ab
#Vab = dendro.sym_3x3("Vab", "[pp]")

# Spatial projection of Ricci tensor related quantities
# From R_ab 
Atr = dendro.scalar("Atr", "[pp]")
Aij  = dendro.sym_3x3("Aij", "[pp]")
# From V_ab 
Btr = dendro.scalar("Btr", "[pp]")
Bij  = dendro.sym_3x3("Bij", "[pp]")

# Additional constraint as evolutions vars
# NOTE : We are not evolving constraints. 
Ci = dendro.vec3("Ci","[pp]")
#Ei = dendro.vec3("Ei","[pp]")

# Lie derivative weight
weight = -Rational(2,3)
weight_Gt = Rational(2,3)

# specify the functions for computing first and second derivatives
d = dendro.set_first_derivative('grad')    # first argument is direction
d2s = dendro.set_second_derivative('grad2')  # first 2 arguments are directions
ad = dendro.set_advective_derivative('agrad')  # first argument is direction
kod = dendro.set_kreiss_oliger_dissipation('kograd')

'''
Symbolic differentiation rules
dendro.d   = lambda i,x : symbols("grad_%d_%s"%(i,x))
dendro.ad   = lambda i,x : symbols("agrad_%d_%s"%(i,x))
dendro.kod   = lambda i,x : symbols("kograd_%d_%s"%(i,x))
dendro.d2  = lambda i,j,x : symbols("grad2_%d_%d_%s"%(min(i,j),max(i,j),x))
'''

d2 = dendro.d2

#f = Function('f')

# generate metric related quantities
dendro.set_metric(gt)
igt = dendro.get_inverse_metric()

C1 = dendro.get_first_christoffel()
C2 = dendro.get_second_christoffel()
#what's this...tried to comment it out and python compilation fails
C2_spatial = dendro.get_complete_christoffel(chi)
R, Rt, Rphi, CalGt = dendro.compute_ricci(Gt, chi)
Rie = dendro.compute_riemann()
###################################################################
# evolution equations
###################################################################

# Gauge part
# For lapse, 1+log. For shift, modified gamma driver
a_rhs = l1*dendro.lie(b, a) - 2*a*K + 0*dendro.kodiss(a)

b_rhs = [ S(3)/4 * (lf0 + lf1*a) * B[i] +
        l2 * dendro.vec_j_ad_j(b, b[i])
         for i in dendro.e_i ] + 0*dendro.kodiss(b)

eta_func = R0*sqrt(sum([igt[i,j]*d(i,chi)*d(j,chi) for i,j in dendro.e_ij]))/((1-chi**ep1)**ep2)

B_rhs = [Gt_rhs[i] - eta_func * B[i] +
         l3 * dendro.vec_j_ad_j(b, B[i]) -
         l4 * dendro.vec_j_ad_j(b, Gt[i]) + 0*kod(i,B[i])
         for i in dendro.e_i]

# Metric and extrinsic curvature

gt_rhs = dendro.lie(b, gt, weight) - 2*a*At + 0*dendro.kodiss(gt)

chi_rhs = dendro.lie(b, chi, weight) + Rational(2,3) * (chi*a*K) + 0*dendro.kodiss(chi)

AikAkj = Matrix([sum([At[i, k] * sum([dendro.inv_metric[k, l]*At[l, j] for l in dendro.e_i]) for k in dendro.e_i]) for i, j in dendro.e_ij])

#NOTE : "CAUTION" THIS IS DIFFERENT THEN Atr for Ricci. 
#NOTE : The rho, S, and Sij in BSSN eqns are not "physical" matter.
#Introduce not conformally transformed metric again
gs = gt/chi
igs = igt/chi

# Define additional constraints
# Some precomputation/definition
Aij_UU = dendro.up_up(Aij)*(chi*chi)
AiUjD = dendro.up_down(Aij)*chi

Kij = At + 1/3*gt*K
Kki = dendro.up_down(Kij)*chi

# Ci is a momentume constraint
'''
Ci = Matrix([sum([igt[j,k]*(  d(k,At[i,j]) - \
              sum(dendro.C2[m,k,i]*At[j,m] for m in dendro.e_i)) \
                  for j,k in dendro.e_ij]) for i in dendro.e_i]) - \
      Matrix([sum([Gt[j]*At[i,j] for j in dendro.e_i]) for i in dendro.e_i]) -\
      Rational(3,2)*Matrix([ \
            sum([igt[j,k]*At[k,i]*d(j,chi)/chi for j,k in dendro.e_ij])  \
            for i in dendro.e_i]) -\
      Rational(2,3)*Matrix([d(i,K) for i in dendro.e_i])
Ci = [item for sublist in Ci.tolist() for item in sublist]
'''

Ci_U = Matrix([sum([Ci[j]*igs[i,j] for j in dendro.e_i]) for i in dendro.e_i])

# Ei is determined by the spatial projection of RHS of Eqn.47
diAiUjD = Matrix([sum([d(i, gt[i,k])*Aij[k,j] + igt[i,k]*d(i,Aij[k,j])  for i, k in dendro.e_ij]) for j in dendro.e_i])
DiAiUjD = diAiUjD + Matrix([sum([sum([dendro.C3[k,k,l]*AiUjD[l,i] - dendro.C3[l,k,i]*AiUjD[k,l] for l in dendro.e_i]) for k in dendro.e_i]) for i in dendro.e_i]) 
Ei = Matrix([sum([-Kki[k,i]*Ci[k] for k in dendro.e_i]) - K*Ci[i] - d(i,Atr)/3 + d(i,Rsc)/4 for i in dendro.e_i]) - DiAiUjD

rho_qg = Rsc/4
Si_qg = Matrix([[-Ci[0], -Ci[1], -Ci[2]]])
Sij_qg = Matrix([Aij[i,j] + gs[i,j]*Atr/3 + gs[i,j]*Rsc/4 for i,j in dendro.e_ij]).reshape(3,3)
S_qg = sum([sum([Sij_qg[i,j]*igs[i,j] for i in dendro.e_i]) for j in dendro.e_i])

At_rhs = dendro.lie(b, At, weight) + chi*dendro.trace_free( a*R - dendro.DiDj(a)-0*8*pi*Sij_qg) + a*(K*At - 2*AikAkj.reshape(3, 3)) + 0*dendro.kodiss(At)

K_rhs = dendro.lie(b, K) - dendro.laplacian(a,chi) + a*(K*K/3 + dendro.sqr(At)) + 0*4*pi*a*(rho_qg + S_qg) + 0*dendro.kodiss(K)

At_UU = dendro.up_up(At)

Gt_rhs = Matrix([sum(b[j]*ad(j,Gt[i]) for j in dendro.e_i) for i in dendro.e_i]) - \
         Matrix([sum(CalGt[j]*d(j,b[i]) for j in dendro.e_i) for i in dendro.e_i]) + \
         Rational(2,3)*Matrix([ CalGt[i] * sum(d(j,b[j]) for j in dendro.e_i)  for i in dendro.e_i ]) + \
         Matrix([sum([igt[j, k] * d2(j, k, b[i]) + igt[i, j] * d2(j, k, b[k])/3 for j, k in dendro.e_ij]) for i in dendro.e_i]) - \
         Matrix([sum([2*At_UU[i, j]*d(j, a) for j in dendro.e_i]) for i in dendro.e_i]) + \
         Matrix([sum([2*a*dendro.C2[i, j, k]*At_UU[j, k] for j,k in dendro.e_ij]) for i in dendro.e_i]) - \
         Matrix([sum([a*(3/chi*At_UU[i,j]*d(j, chi) + Rational(4,3)*dendro.inv_metric[i, j]*d(j, K)) for j in dendro.e_i]) for i in dendro.e_i])
         # + kod(i,Gt[i])
n_vec = Matrix([[1/a, -b[0]/a, -b[1]/a, -b[2]/a]])
Gt_rhs = [item for sublist in Gt_rhs.tolist() for item in sublist]

# Ricci tensor and scalar

# normal vector n^a
n_vec = Matrix([[1/a, -b[0]/a, -b[1]/a, -b[2]/a]])

# Define acceleration (n^c \del_c n_a) HL : this is equal to 1/a*D_i a
a_acc = Matrix([d(i,a) for i in dendro.e_i])/a

#==========================================================================================
#==========================================================================================
# 20-Feb-23: Eq numbers are based on this version of the draft
#==========================================================================================
#==========================================================================================


#==========================================================================================
# R-equation (from R equation)
# Eqn.24 in draft
#==========================================================================================
Rsc_rhs = dendro.lie(b, Rsc) - a*Rsch

#==========================================================================================
# aux R-equation (from R equation)
# Eqn.25 in draft
#==========================================================================================
Rsch_rhs = (
	dendro.lie(b, Rsch) 
	- a*(
		dendro.laplacian(Rsc,chi) 
		+ sum([a_acc[i]*d(i,Rsc) 
		for i in dendro.e_i
		]) 
		- K*Rsch - qg_mass0_sq*Rsc
	)
	#- 2*0*a*(rho_qg - S)
)

#==========================================================================================
# A-trace-equation (from trace-projection of R_ab equation)
# Eqn.34 in draft
#==========================================================================================
Atr_rhs = (
	dendro.lie(b, Atr) 
	+ a*(
		2*sum([a_acc[i]*Ci[i] for i in dendro.e_i]) 
		- Btr
	)
) 

#==========================================================================================
# Aij-equation (from traceless-projection of R_ab equation)
# Eqn.35 in draft
#==========================================================================================

# Precomputataion of covariant derviative
Djni= (
	Matrix([
		d(j,b[i])/a 
		- b[i]*d(i,a)/(a*a) 
		+ sum([
			dendro.C3[k,j,i]*n_vec[k]
			for k in dendro.e_i
		]) 
		for i,j in dendro.e_ij
	]).reshape(3,3)
)
Dinj= (
	Matrix([
		d(i,b[j])/a 
		- b[j]*d(j,a)/(a*a) 
		+ sum([
			dendro.C3[k,i,j]*n_vec[k]
			for k in dendro.e_i
		]) 
		for i,j in dendro.e_ij
	]).reshape(3,3)
)

# from LHS
Aij_rhs1 = Matrix([
	sum([
		b[l]*d(l,Aij[i,j]) 
		for l in dendro.e_i
	]) 
	for i,j in dendro.e_ij
])
# RHS: line 1: term 1
Aij_rhs2 = 2*a/3*Atr*((Djni+Dinj)/2 + Kij)
# RHS: line 1: term 2
Aij_rhs3 = a*Matrix([
	sum([
		a_acc[k]*(
			(
				Aij[k,i]*n_vec[j]
				+Aij[k,j]*n_vec[i]
			)
			+Atr*(
				gs[k,i]*n_vec[j]/3
				+gs[k,j]*n_vec[i]/3
			)
			+(
				gs[k,i]*Ci[j]
				+gs[k,j]*Ci[i]
			)
		)
		for k in dendro.e_i
	]) 
	for i,j in dendro.e_ij
])
# combine RHS (includes RHS: line 1: term 3)
Aij_rhs = (
	- a*(
		2/3*gt*sum([
			a_acc[k]*Ci[k] 
			for k in dendro.e_i
		]) 
		+ Bij
	) 
	+ Aij_rhs1.reshape(3,3) 
	+ Aij_rhs2 
	+ Aij_rhs3.reshape(3,3)
)

#==========================================================================================
# B-trace-equation (from trace-projection of V_ab equation)
# Eqn.36 in draft
#==========================================================================================

#pre-computation
dRsc = Matrix([
	d(i,Rsc) 
	for i in dendro.e_i
])

DjdRsci= Matrix([
	d2(i,j,Rsc) 
	+ sum([
		dendro.C3[k,j,i]*dRsc[k] 
		for k in dendro.e_i
	]) 
	for i,j in dendro.e_ij
]).reshape(3,3)

DiKij = Matrix([
	sum([
		d(i,At[i,j]) 
		+ (
			d(i,K)*gs[i,j] 
			+ K*(d(i,gt[i,j])/chi 
			- d(i,chi)*gt[i,j]/(chi*chi))
		)/3 
		+ sum([
			dendro.C3[i,i,l]*Kij[l,j] 
			+ dendro.C3[j,i,l]*Kij[i,l] 
			for l in dendro.e_i
		]) 
		for i in dendro.e_i
	]) 
	for j in dendro.e_i
])

a_acc_UP = Matrix([sum([a_acc[j]*igs[i,j] for j in dendro.e_i]) for i in dendro.e_i])

DkCj = Matrix([d(k,Ci[j]) + sum([dendro.C3[k,j,l]*Ci[l] for l in dendro.e_i]) for j,k in dendro.e_ij]).reshape(3,3)
DkCi = Matrix([d(k,Ci[i]) + sum([dendro.C3[k,i,l]*Ci[l] for l in dendro.e_i]) for i,k in dendro.e_ij]).reshape(3,3)

DkKjk = Matrix([sum([ d(k,At[j,k]) + d(k,K)*gs[j,k] + K*(d(k,gt[j,k])/chi -d(k,chi)*gt[j,k]/(chi*chi))/3 - sum([dendro.C3[j,k,l]*Kij[l,j] + dendro.C3[k,j,l]*Kij[k,l] for l in dendro.e_i]) for k in dendro.e_i]) for j in dendro.e_i])
DkKik = Matrix([sum([ d(k,At[i,k]) + d(k,K)*gs[i,k] + K*(d(k,gt[i,k])/chi -d(k,chi)*gt[i,k]/(chi*chi))/3 - sum([dendro.C3[i,k,l]*Kij[l,i] + dendro.C3[k,i,l]*Kij[k,l] for l in dendro.e_i]) for k in dendro.e_i]) for i in dendro.e_i])

# from LHS and RHS: line 1: term 1
Btr_rhs1 = dendro.lie(b, Btr) +2*a*sum([a_acc[k]*Ei[k] for k in dendro.e_i]) 
# RHS: line 1: term 2 & 3
# also RHS: line 2: term 1
Btr_rhs2 = - a*(
	dendro.laplacian(Atr,chi) 
	+ sum([a_acc[i]*d(i,Atr) for i in dendro.e_i]) 
	- qg_mass2_sq*Atr 
	- K*Btr 
	- Rsch*Atr/3
)
# RHS: line 2: term 2
Btr_rhs3 = 3*a/2*(
	sum([Aij[i,j]*Aij_UU[i,j] for i,j in dendro.e_ij]) 
	+ Atr*Atr 
	- 2*sum([Ci[i]*Ci_U[i] for i in dendro.e_i])
)
# RHS: line 4: term 1
Btr_rhs4 = a/3*(qg_mass2_sq/qg_mass0_sq + 1)*Rsc*Atr
# RHS: line 4: term 2
Btr_rhs5 = -a/3*(qg_mass2_sq/qg_mass0_sq - 1)*(
	- 2*K*Rsch 
	+ dendro.laplacian(Rsc,chi) 
	- 3/4*qg_mass0_sq*Rsc
)
# RHS: line 3: term 2
Btr_rhs6 = 4*a*(
	sum([
		Ci_U[j]*(
			d(j,K) 
			- DiKij[j]
		) 
		for j in dendro.e_i
	])
)
# RHS: line 3: term 1
Btr_rhs7 = -2*a*(
	sum([
		(
			Aij_UU[i,j] 
			+ Atr*igs[i,j]/3
		)*(
			Rt[i,j]
			+K*Kij[i,j]
			-sum([
				Kij[k,i]*Kij[k,j]
				for k in dendro.e_i
			])
		) 
		for i,j in dendro.e_ij
	])
)
# RHS: line 2: term 3
Btr_rhs8 = -2*a*(
	sum([
		(
			Ci[i]*DkKjk[i]
			+ sum([
				Ci[i]*a_acc_UP[j]*Kij[i,j]
				for j in dendro.e_i
			])
		) 
		for i in dendro.e_i
	])
)
# RHS: line 2: term 4
Btr_rhs9 = -4*a*(
	sum([
     sum([
		  ( 
		 	  Kij[k,j]*DkCj[k,j]
		  )  
		   for k in dendro.e_i
	     ])
         for j in dendro.e_i
        ])
)

# combine RHS
Btr_rhs = Btr_rhs1 + Btr_rhs2 + Btr_rhs3 + Btr_rhs4 + Btr_rhs5 + Btr_rhs6 + Btr_rhs7 + Btr_rhs8 + Btr_rhs9


#==========================================================================================
# Bij-equation (from traceless-projection of R_ab equation)
# Eqn.37 in draft
#==========================================================================================


# Some precomputation
#a_acc_UP = Matrix([sum([a_acc[j]*igs[i,j] for j in dendro.e_i]) for i in dendro.e_i])
# Precomputations of covariant derivative
DkAij = np.array([ d(k,Aij[i,j]) + sum([dendro.C3[i,k,l]*Aij[l,j] + dendro.C3[j,k,l]*Aij[i,l] for l in dendro.e_i]) for i,j in dendro.e_ij for k in dendro.e_i]).reshape((3,3,3))
DiKkj = np.array([ d(i,At[k,j]) + d(i,K)*gs[k,j] + K*(d(i,gt[k,j])/chi -d(i,chi)*gt[k,j]/(chi*chi))/3 - sum([dendro.C3[k,i,l]*Kij[l,j] + dendro.C3[j,i,l]*Kij[k,l] for l in dendro.e_i]) for i,j in dendro.e_ij for k in dendro.e_i]).reshape((3,3,3))
DkKij = np.array([  d(k,At[i,j]) + d(k,K)*gs[i,j] + K*(d(k,gt[i,j])/chi -d(k,chi)*gt[i,j]/(chi*chi))/3 - sum([dendro.C3[i,k,l]*Kij[l,j] + dendro.C3[j,k,l]*Kij[i,l] for l in dendro.e_i]) for i,j in dendro.e_ij for k in dendro.e_i]).reshape((3,3,3))
DjKki = np.array([  d(j,At[k,i]) + d(j,K)*gs[k,i] + K*(d(j,gt[k,i])/chi -d(j,chi)*gt[k,i]/(chi*chi))/3 - sum([dendro.C3[k,j,l]*Kij[l,i] + dendro.C3[i,j,l]*Kij[k,l] for l in dendro.e_i]) for i,j in dendro.e_ij for k in dendro.e_i]).reshape((3,3,3))
DkKji = np.array([  d(k,At[j,i]) + d(k,K)*gs[j,i] + K*(d(k,gt[j,i])/chi -d(k,chi)*gt[j,i]/(chi*chi))/3 - sum([dendro.C3[j,k,l]*Kij[l,i] + dendro.C3[i,j,l]*Kij[k,l] for l in dendro.e_i]) for i,j in dendro.e_ij for k in dendro.e_i]).reshape((3,3,3))

#DkKjk = Matrix([sum([ d(k,At[j,k]) + d(k,K)*gs[j,k] + K*(d(k,gt[j,k])/chi -d(k,chi)*gt[j,k]/(chi*chi))/3 - sum([dendro.C3[j,k,l]*Kij[l,j] + dendro.C3[k,j,l]*Kij[k,l] for l in dendro.e_i]) for k in dendro.e_i]) for j in dendro.e_i])
#DkKik = Matrix([sum([ d(k,At[i,k]) + d(k,K)*gs[i,k] + K*(d(k,gt[i,k])/chi -d(k,chi)*gt[i,k]/(chi*chi))/3 - sum([dendro.C3[i,k,l]*Kij[l,i] + dendro.C3[k,i,l]*Kij[k,l] for l in dendro.e_i]) for k in dendro.e_i]) for i in dendro.e_i])
#DkCj = Matrix([d(k,Ci[j]) + sum([dendro.C3[k,j,l]*Ci[l] for l in dendro.e_i]) for j,k in dendro.e_ij])
#DkCi = Matrix([d(k,Ci[i]) + sum([dendro.C3[k,i,l]*Ci[l] for l in dendro.e_i]) for i,k in dendro.e_ij])

# from LHS
Bij_rhs1 = Matrix([
	sum(
		b[l]*d(l,Bij[i,j])
		for l in dendro.e_i
	) 
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 1: term 1
Bij_rhs2 = 2/3*a*Btr*(
	(Dinj + Djni)/2 
	+ Kij
)
# RHS: line 1: term 2
Bij_rhs3 = Matrix([
	2*a*sum([
		a_acc[k]*(
			(
				Bij[k,i]*n_vec[j] 
				+ Bij[k,j]*n_vec[i]
			)/2 
			+ Btr*(
				gs[k,i]*n_vec[j] 
				+ gs[k,j]*n_vec[i]
			)/6 
			+ (
				gs[k,i]*Ei[j] 
				+ gs[k,j]*Ei[i]
			)/2
		)
		for k in dendro.e_i
	])
	for i,j in dendro.e_ij
]).reshape(3,3) 
# RHS: line 1: term 4
Bij_rhs4 = -a*gs*Btr_rhs/3
# RHS: line 2: term 1: 1st part
# also RHS:line 3: term 1: 1st part
Bij_rhs5 = (
	-Matrix([
		a*(
			gs[i,j]*dendro.laplacian(Atr,chi)/3 
			- qg_mass2_sq*Aij[i,j] 
			- qg_mass2_sq*gs[i,j]*Atr/3
		) 
		for i,j in dendro.e_ij
	]).reshape(3,3) 
	- a*dendro.laplacianR2(Aij,chi)
)
# RHS: line 2: term 1 (2nd part)
Bij_rhs6 = -Matrix([
	a*(
		sum([
			a_acc_UP[k]*DkAij[i,j,k] 
			+ a_acc_UP[k]*d(k,Atr)*gs[i,j]/3 
			for k in dendro.e_i
		])
	)
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 2: term 2
# also RHS: line 1: term 3 (matter term)
Bij_rhs7 = Matrix([
	a*K*(
		Bij[i,j] 
		+ gs[i,j]*Btr/3
	) 
	+ 2*a*Sij_qg[i,j]
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 2: term 3 
Bij_rhs8 = a*Matrix([
	-(
		Ci[i]*DkKjk[j] 
		+ Ci[j]*DkKjk[i] 
		+ sum([
			Ci[i]*a_acc_UP[k]*Kij[k,j]
			+ Ci[j]*a_acc_UP[k]*Kij[k,i] 
			for k in dendro.e_i
		])
	) 
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 2: term 4
Bij_rhs9 = -2*a*Matrix([
	sum([
		Kij[k,i]*DkCj[k,j] 
		+ Kij[k,j]*DkCj[k,i] 
		for k in dendro.e_i
	])
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 3: term 2
Bij_rhs10 = Matrix([
	a/2*gs[i,j]*(
		Atr*Atr/3
		+ sum([
			sum([
				Aij_UU[k,l]*Aij[k,l] 
				for l in dendro.e_i
			]) 
			-2*Ci_U[k]*Ci[k] 
			for k in dendro.e_i
		])
	) 
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 5: term 1
Bij_rhs11 = Matrix([
	a/3*(
		qg_mass2_sq/qg_mass0_sq + 1
	)*(
		Rsc*(
			Aij[i,j] 
			+ gs[i,j]*Atr/3
		)
	)
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 5: term 2
Bij_rhs12 = -Matrix([
	a/3*(
		qg_mass2_sq/qg_mass0_sq - 1
	)*(
		DjdRsci[i,j] 
		- qg_mass0_sq*gs[i,j]*Rsc/4 
		- 2*Kij[i,j]*Rsch
	) 
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 4: term 1
Bij_rhs13 = -Matrix([
	2*a*sum([
		(
			Aij_UU[k,l] 
			+ igs[k,l]*Atr/3
		)*(
			Rie[i,k,j,l] 
			+ Kij[i,j]*Kij[l,k]/2 
			- Kij[i,l]*Kij[j,k]/2
		) 
		for k,l in dendro.e_ij
	]) 
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 4: term 2
Bij_rhs14 = Matrix([
	4*a*sum([
		Ci_U[k]*(
			DkKij[k,i,j] 
			- DkKij[i,j,k]/2
			+ DkKij[j,i,k]/2
		) 
		for k in dendro.e_i
	]) 
	for i,j in dendro.e_ij
]).reshape(3,3)
# RHS: line 3: term 1 (part 2)
Bij_rhs15 = -Matrix([
	Rsc*(
		Aij[i,j] 
		+ Atr*gs[i,j]/3
	)/6 
	for i,j in dendro.e_ij
]).reshape(3,3)

# collect RHS terms
Bij_rhs = (
	Bij_rhs1 + Bij_rhs2 + Bij_rhs3 + Bij_rhs4 
	+ Bij_rhs5 + Bij_rhs6 + Bij_rhs7 + Bij_rhs8 
	+ Bij_rhs9 + Bij_rhs10 + Bij_rhs11 + Bij_rhs12 
	+ Bij_rhs13 + Bij_rhs14 + Bij_rhs15
) 

#==========================================================================================
# Ci-equation (from mixed projection of R_ab equation)
# Eqn.39 in draft
# not actually needed (could be solved by constraint)
#==========================================================================================
#Ci_rhs1 = Matrix([sum([b[j]*ad(j,Ci[i]) - Ci[j]*d(j,b[i]) + weight*Ci[i]*d(j,b[j])  for j in dendro.e_i]) for i in dendro.e_i])
#Ci_rhs2 = Matrix([sum([a_acc[k]*(Aij[k,i] + gs[i,k]*Atr/3) + n_vec[i]*a_acc[k]*Ci[k] for k in dendro.e_i]) for i in dendro.e_i])
#Ci_rhs = Ci_rhs1 + Ci_rhs2 - Ei
#Ci_rhs = [item for sublist in Ci_rhs.tolist() for item in sublist]

#RHS of Eqn.43, same argument from Aij_rhs is applicable for this 
ok
#TODO : Additional constraints, C_k, E_k go here if we want to evolve and monitor

#_I = gt*igt
#print(simplify(_I))

#_I = gt*dendro.inv_metric
#print(simplify(_I))


###
# Substitute ...
#for expr in [a_rhs, b_rhs[0], b_rhs[1], b_rhs[2], B_rhs[0], B_rhs[1], B_rhs[2], K_rhs, chi_rhs, Gt_rhs[0], Gt_rhs[1], Gt_rhs[2], gt_rhs[0], gt_rhs[0,0], gt_rhs[1,1], gt_rhs[2,2], gt_rhs[0,1], gt_rhs[0,2], gt_rhs[1,2], At_rhs[0,0], At_rhs[0,1], At_rhs[0,2], At_rhs[1,1], At_rhs[1,2], At_rhs[2,2]]:
#    for var in [a, b[0], b[1], b[2], B[0], B[1], B[2], chi, K, gt[0,0], gt[0,1], gt[0,2], gt[1,1], gt[1,2], gt[2,2], Gt[0], Gt[1], Gt[2], At[0,0], At[0,1], At[0,2], At[1,1], At[1,2], At[2,2]]:
#        expr.subs(d2(1,0,var), d2(0,1,var))
#        expr.subs(d2(2,1,var), d2(1,2,var))
#        expr.subs(d2(2,0,var), d2(0,2,var))
#
#print (a_rhs)
#print (G_rhs)


###################################################################
# generate code
###################################################################

outs = [a_rhs, b_rhs, gt_rhs, chi_rhs, At_rhs, K_rhs, Gt_rhs, B_rhs, Rsc_rhs, Rsch_rhs, Atr_rhs, Aij_rhs, Btr_rhs, Bij_rhs]
vnames = ['a_rhs', 'b_rhs', 'gt_rhs', 'chi_rhs', 'At_rhs', 'K_rhs', 'Gt_rhs', 'B_rhs', 'Rsc_rhs', 'Rsch_rhs', 'Atr_rhs', 'Aij_rhs', 'Btr_rhs', 'Bij_rhs']
#outs = [Ci_rhs]
#vnames = ['Ci_rhs']
#dendro.generate_debug(outs, vnames)
dendro.generate(outs, vnames, '[pp]')
#numVars=len(outs)
#for i in range(0,numVars):
#    dendro.generate_separate([outs[i]],[vnames[i]],'[pp]')
