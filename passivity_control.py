#!/usr/bin/env python
import numpy as np 
import scipy as sci
import pandas as pd
import sys
import csv
import rospy
from numpy import linalg
from scipy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=6)

class Adaptive_Control:
	def __init__(self):
		self.m1 = 4.0
		self.m2 = 4.0
		self.L1 = 2.5
		self.L2 = 1.5
		self.J1 = (1.0/12.0)*self.m1*self.L1**2
		self.J2 = (1.0/12.0)*self.m2*self.L2**2
		self.Lc1 = self.L1/2
		self.Lc2 = self.L2/2

		self.k = 1    #signal amplitude
		self.ke = 1	  #passivity gain
		self.k_gamma = 1 #adaptive gain
		self.kD = 1  #Controller gain
		self.k_S = 1
		self.w = 1
		self.gamma = self.k_gamma*np.identity(5)
		self.K = self.kD*np.identity(2)
		self.L = self.ke*np.identity(2)
		self.g_const = 9.81
		self.Ks = self.k_S*np.identity(2)
		self.theta_star = np.array([[self.m1*self.Lc1**2 + self.m2*(self.L1**2 + self.Lc2**2) + self.J1 + self.J2,
									self.m2*self.Lc2**2 + self.J2,
									self.m2*self.L1*self.Lc2,
									self.m1*self.Lc1 + self.m2*self.L1,
									self.m2*self.L2]]).T
		print(self.theta_star)


	def computed_torque_ode(self, w, t, i):
		'''
		Defines the differential equations for the coupled spring-mass system.
		w :  vector of the state variables:

						w[0] = q_dot_1 (joint velocity)
						w[1] = q_dot_2
						w[2] = q_1 (joint position)
						w[3] = q_2 

		'''
		z = 0.5
		a = self.k*np.sin(self.w*t)
		b = self.k*np.sin(0.5*self.w*t)
		c = self.k*np.sin(0.3*self.w*t)
		d = self.k*np.sin(0.4*self.w*t)

		dz = 0
		da = self.k*self.w*np.cos(self.w*t)
		db = 0.5*self.k*self.w*np.cos(0.5*self.w*t)
		dc = 0.3*self.k*self.w*np.cos(0.3*self.w*t)
		dd = 0.4*self.k*self.w*np.cos(0.4*self.w*t)


		ddz = 0
		dda = -self.k*(self.w**2)*np.sin(self.w*t)
		ddb = -0.25*self.k*(self.w**2)*np.sin(0.5*self.w*t)
		ddc = -0.09*self.k*(self.w**2)*np.sin(0.3*self.w*t)
		ddd = -0.16*self.k*(self.w**2)*np.sin(0.4*self.w*t)

		traj = np.array([z, a, a+b, a+b+c, a+b+c+d])
		dtraj = np.array([dz, da, da+db, da+db+dc, da+db+dc+dd])
		ddtraj = np.array([ddz, dda, dda+ddb, dda+ddb+ddc, dda+ddb+ddc+ddd])


		q_d = np.array([[traj[i], traj[i]]]).T
		q_dot_d = np.array([[dtraj[i], dtraj[i]]]).T
		q_ddot_d = np.array([[ddtraj[i], ddtraj[i]]]).T

		q = np.array([[w[2],w[3]]]).T
		q_dot = np.array([[w[0],w[1]]]).T


		e = q - q_d
		e_dot = q_dot - q_dot_d
		s = e_dot + self.Ks.dot(e)
		v = q_dot_d - self.L.dot(e)
		a = q_ddot_d - self.L.dot(e_dot)

		#True Dynamic Model
		d11 = self.Lc1**2 + self.m2*(self.L1**2+self.Lc2**2+2*self.L1*self.Lc2*np.cos(q[1,0])) + self.J1 + self.J2
		d12 = self.m2*(self.Lc2**2+self.L1*self.Lc2*np.cos(q[1,0])) + self.J2
		d21 = d12
		d22 = self.m2*self.Lc2**2 + self.J2

		h = -self.m2*self.L1*self.Lc2*np.sin(q[1,0])

		c11 = h*q_dot[1,0]
		c12 = h*(q_dot[1,0]+q_dot[0,0])
		c21 = -h*q_dot[0,0]
		c22 = 0

		g1 = (self.m1*self.Lc1 + self.m2*self.L1)*self.g_const*np.cos(q[0,0]) + self.m2*self.Lc2*self.g_const*np.cos(q[0,0]+q[1,0])
		g2 = self.m2*self.Lc2*self.g_const*np.cos(q[0,0]+q[1,0])

		M = np.array([[d11,d12],[d21,d22]])
		C = np.array([[c11,c12],[c21,c22]])
		G = np.array([[g1,g2]]).T

		#Compute Torque
		tau = M.dot(a) + C.dot(v) + G - self.K.dot(s)

		#Input to Robot
		q_ddot = np.linalg.inv(M).dot(tau - C.dot(q_dot) - G)


		dwdt = [q_ddot[0,0], q_ddot[1,0], q_dot[0,0], q_dot[1,0]]

		return dwdt

	def adaptive_ode(self, w, t, i, th):
		'''
		Defines the differential equations for the coupled spring-mass system.
		w :  vector of the state variables:

						w[0] = theta_1, 
						w[1] = theta_2, 
						w[2] = theta_3, 
						w[3] = theta_4, 
						w[4] = theta_5,
						w[5] = q_dot_1 (joint velocity)
						w[6] = q_dot_2
						w[7] = q_1 (joint position)
						w[8] = q_2 

		'''
		z = 0.5
		a = self.k*np.sin(self.w*t)
		b = self.k*np.sin(0.5*self.w*t)
		c = self.k*np.sin(0.3*self.w*t)
		d = self.k*np.sin(0.4*self.w*t)

		dz = 0
		da = self.k*self.w*np.cos(self.w*t)
		db = 0.5*self.k*self.w*np.cos(0.5*self.w*t)
		dc = 0.3*self.k*self.w*np.cos(0.3*self.w*t)
		dd = 0.4*self.k*self.w*np.cos(0.4*self.w*t)


		ddz = 0
		dda = -self.k*(self.w**2)*np.sin(self.w*t)
		ddb = -0.25*self.k*(self.w**2)*np.sin(0.5*self.w*t)
		ddc = -0.09*self.k*(self.w**2)*np.sin(0.3*self.w*t)
		ddd = -0.16*self.k*(self.w**2)*np.sin(0.4*self.w*t)

		traj = np.array([z, a, a+b, a+b+c, a+b+c+d])
		dtraj = np.array([dz, da, da+db, da+db+dc, da+db+dc+dd])
		ddtraj = np.array([ddz, dda, dda+ddb, dda+ddb+ddc, dda+ddb+ddc+ddd])


		q_d = np.array([[traj[i], traj[i]]]).T
		q_dot_d = np.array([[dtraj[i], dtraj[i]]]).T
		q_ddot_d = np.array([[ddtraj[i], ddtraj[i]]]).T

		q = np.array([[w[7],w[8]]]).T
		q_dot = np.array([[w[5],w[6]]]).T


		e = q - q_d
		e_dot = q_dot - q_dot_d
		s = e_dot + self.Ks.dot(e)
		v = q_dot_d - self.L.dot(e)
		a = q_ddot_d - self.L.dot(e_dot)

		#Parameter Adaptation
		theta_hat = np.array([[w[0], w[1], w[2], w[3], w[4]]]).T

		Y = np.array([[a[0,0],   a[1,0],    np.cos(q[1,0])*(2*a[0,0]+a[1,0]) - q_dot[1,0]*np.sin(q[1,0])*v[0,0] - (q_dot[0,0]+q_dot[1,0])*np.sin(q[1,0])*v[0,0]       ,    self.g_const*np.cos(q[0,0]), self.g_const*np.cos(q[0,0]+q[1,0])],
					  [0,    a[0,0]+a[1,0] ,       np.cos(q[1,0])*a[0,0]+np.sin(q[1,0])*q_dot[0,0]*v[0,0]                    ,  0, self.g_const*np.cos(q[0,0]+q[1,0]) ]])

		theta_hat_dot = -(np.linalg.inv(self.gamma).dot(Y.T)).dot(s)
		tau = Y.dot(theta_hat) - self.K.dot(s)

		#True Dynamic Model
		d11 = self.Lc1**2 + self.m2*(self.L1**2+self.Lc2**2+2*self.L1*self.Lc2*np.cos(q[1,0])) + self.J1 + self.J2
		d12 = self.m2*(self.Lc2**2+self.L1*self.Lc2*np.cos(q[1,0])) + self.J2
		d21 = d12
		d22 = self.m2*self.Lc2**2 + self.J2

		h = -self.m2*self.L1*self.Lc2*np.sin(q[1,0])

		c11 = h*q_dot[1,0]
		c12 = h*(q_dot[1,0]+q_dot[0,0])
		c21 = -h*q_dot[0,0]
		c22 = 0

		g1 = (self.m1*self.Lc1 + self.m2*self.L1)*self.g_const*np.cos(q[0,0]) + self.m2*self.Lc2*self.g_const*np.cos(q[0,0]+q[1,0])
		g2 = self.m2*self.Lc2*self.g_const*np.cos(q[0,0]+q[1,0])

		M = np.array([[d11,d12],[d21,d22]])
		C = np.array([[c11,c12],[c21,c22]])
		G = np.array([[g1,g2]]).T

		#Input to Robot
		q_ddot = np.linalg.inv(M).dot(tau - C.dot(q_dot) - G)


		dwdt = [theta_hat_dot[0,0], theta_hat_dot[1,0], theta_hat_dot[2,0], theta_hat_dot[3,0], theta_hat_dot[4,0],
				q_ddot[0,0], q_ddot[1,0], q_dot[0,0], q_dot[1,0]]

		return dwdt


	def adaptive_simulation(self):
		theta_1 = 0.0
		theta_2 = 0.0
		theta_3 = 0.0
		theta_4 = 0.0
		theta_5 = 0.0
		q_dot_1 = 0.0
		q_dot_2 = 0.0
		q_1 = 0.0
		q_2 = 0.0


		# ODE solver parameters
		abserr = 1.0e-8
		relerr = 1.0e-6
		stoptime = 100
		numpoints = 250
		theta_0 = 1.2*self.theta_star
		t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
		w0 = [theta_0[0], theta_0[1], theta_0[2], theta_0[3], theta_0[4], q_dot_1, q_dot_2, q_1, q_2]

		j = 0
		for i in range(3):
			wsol = odeint(self.adaptive_ode, w0, t, args =(i,self.theta_star), atol=abserr, rtol=relerr)
			time = np.asarray(t)
			data = np.asarray(wsol)
			z = 0.5
			a = self.k*np.sin(self.w*time)
			b = self.k*np.sin(0.5*self.w*time)
			c = self.k*np.sin(0.3*self.w*time)
			d = self.k*np.sin(0.4*self.w*time)
			dz = 0
			da = self.k*self.w*np.cos(self.w*time)
			db = 0.5*self.k*self.w*np.cos(0.5*self.w*time)
			dc = 0.3*self.k*self.w*np.cos(0.3*self.w*time)
			dd = 0.4*self.k*self.w*np.cos(0.4*self.w*time)
			traj_des = np.array([z, a, a+b, a+b+c, a+b+c+d])
			dtraj_des = np.array([dz, da, da+db, da+db+dc, da+db+dc+dd])
			e = data[:,7] - traj_des[i]
			e_dot = data[:,5] - dtraj_des[i]
			s = (e_dot + e)[:,np.newaxis]
			data = np.column_stack((data,s))
			self.plot_adaptive_results(time,data,j)
			j += 1
		plt.show()

	def ct_simulation(self):
		q_dot_1 = 0.0
		q_dot_2 = 0.0
		q_1 = 0.0
		q_2 = 0.0


		# ODE solver parameters
		abserr = 1.0e-8
		relerr = 1.0e-6
		stoptime = 100
		numpoints = 200
		t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
		w0 = [q_dot_1, q_dot_2, q_1, q_2]

		j = 0
		for i in range(1,3):
			wsol = odeint(self.computed_torque_ode, w0, t, args =(i,), atol=abserr, rtol=relerr)
			time = np.asarray(t)
			data = np.asarray(wsol)
			z = 0.5
			a = self.k*np.sin(self.w*time)
			b = self.k*np.sin(0.5*self.w*time)
			c = self.k*np.sin(0.3*self.w*time)
			d = self.k*np.sin(0.4*self.w*time)
			dz = 0
			da = self.k*self.w*np.cos(self.w*time)
			db = 0.5*self.k*self.w*np.cos(0.5*self.w*time)
			dc = 0.3*self.k*self.w*np.cos(0.3*self.w*time)
			dd = 0.4*self.k*self.w*np.cos(0.4*self.w*time)
			traj_des = np.array([z, a, a+b, a+b+c, a+b+c+d])
			dtraj_des = np.array([dz, da, da+db, da+db+dc, da+db+dc+dd])
			e = data[:,2] - traj_des[i]
			e_dot = data[:,0] - dtraj_des[i]
			s = (e_dot + e)[:,np.newaxis]
			data = np.column_stack((data,s))
			self.plot_ct_results(time,data,j)
			j += 1
		plt.show()


	def plot_adaptive_results(self, time, data, i):
		plt.subplot(3,2,2*i+1)
		plt.plot(time,data[:,9])
		plt.xlabel('t')
		plt.ylabel('s')
		plt.subplot(3,2,2*i+2)
		plt.plot(time,data[:,0])
		plt.plot(time,data[:,1])
		plt.plot(time,data[:,2])
		plt.plot(time,data[:,3])
		plt.plot(time,data[:,4])
		plt.xlabel('t')
		plt.legend(['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5'])

	def plot_ct_results(self, time, data, i):
		plt.subplot(2,1,i+1)
		plt.plot(time,data[:,4])
		plt.xlabel('t')
		plt.ylabel('s')
		

if __name__ == '__main__':
	rospy.init_node("test_node", anonymous=True,disable_signals=True)
	adapt = Adaptive_Control()
	adapt.ct_simulation()
	adapt.adaptive_simulation()
