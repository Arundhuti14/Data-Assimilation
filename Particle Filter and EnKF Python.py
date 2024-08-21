import pylab as plt
import numpy as np
import scipy.integrate as integrate
from numpy.linalg import inv
import math

# Simulation parameters
tc = 100  # Total simulation time
mu = 3  # Initial guess for damping coefficient
true_mu = 3.5  # True value of damping coefficient
X0 = [0.1, 0.2, true_mu]  # Initial state for the true system
x0 = [0.1, 0.2, mu]  # Initial state for the model
tspan = np.arange(0, tc + 0.001, 0.01)  # Time span for ODE solver
mem = 50  # Number of ensemble members
sigma = 0.1  # Variance for perturbations

# Import functions for the Van der Pol oscillator
from van import Vanderpolt
from van import Vanderpol2

# Simulate the true system
xr = integrate.odeint(Vanderpolt, X0, tspan)

# Initialize the ensemble with perturbations
s1 = np.random.normal(0, sigma, mem)
s2 = np.random.normal(0, sigma, mem)
s3 = np.random.normal(0, 1, mem)
x0_en = np.zeros((mem, 3))
x0_en[:, 0] = x0[0] + s1
x0_en[:, 1] = x0[1] + s2
x0_en[:, 2] = x0[2] + s3
np.savetxt('initial.dat', x0_en)  # Save initial ensemble to file
fixed = x0_en  # Fixed ensemble for later use

# Observation setup
dt = 0.5  # Time step for the assimilation
timestep = np.arange(dt, tc + 0.001, dt)  # Time steps for assimilation
length = len(timestep)
u = np.zeros((length, 2))  # Array to store observations
R = 0.01  # Observation noise variance
matrixR = [[R, 0], [0, R]]  # Observation noise covariance matrix
H = np.array([[1, 0, 0], [0, 1, 0]])  # Observation operator

# Generate synthetic observations
up = []
t1 = 50
for k in range(length):  
    up1 = np.zeros((mem, 2))
    u[k, 0] = xr[t1, 0] + np.random.normal(0, math.sqrt(R), 1)
    u[k, 1] = xr[t1, 1] + np.random.normal(0, math.sqrt(R), 1)
    t1 += 50
    up1[:, 0] = u[k, 0] + np.random.normal(0, math.sqrt(R), mem)
    up1[:, 1] = u[k, 1] + np.random.normal(0, math.sqrt(R), mem)
    up.append(up1)
np.savetxt('obs.dat', u)  # Save observations to file

# Particle filter setup
sss = []
ts = []
pxf = []
pmean = []
posmean = []
tim = 0
totalt = []

# Particle filter assimilation loop
for t in timestep: 
    x_f = np.zeros((mem, 3))  
    s = 0
    ensem = []
    for i in range(mem):
        tspan2 = np.arange(t-dt, t + 0.001, 0.01)
        totalt.append(tspan2)
        XX = x0_en[i]
        z = integrate.odeint(Vanderpol2, XX, tspan2)
        s += z
        x_f[i, :] = z[-1, :] 
        ensem.append(z)     
    sss.append(ensem)
    ts.append(tspan2)
    pxf.append(x_f)
    mean = s / mem    
    pmean.append(mean)   
    covariance = np.cov(np.transpose(x_f))
    vhat = u[tim, :] - x_f[:, 0:2]
    tim += 1
    weight = np.zeros((mem, 1))
    for i in range(mem):
        weight[i] = 1 / (1 + vhat[i, :].dot(inv(matrixR)).dot(np.transpose(vhat[i, :])))   
    weight = weight / sum(weight)
    
    # Calculate effective sample size
    neff = 1 / np.sum(weight**2)
    cum = np.cumsum(weight)
    
    # Resampling if necessary
    if neff < mem / 2:
        from van import particle
        for i in range(mem):
            x0_en[i, :] = x_f[particle(cum, mem), :]
    else:
        x0_en = x_f
    
    # Stop updates if past a certain time
    if t >= 80:
        x0_en = x_f
  
    # Compute weighted mean
    pos = np.sum(weight * x_f, axis=0)
    posmean.append(pos)

# EnKF (Ensemble Kalman Filter) setup
ssse = []
exf = []
emean = []
enPmean = []
tim = 0
x0_en = fixed

# EnKF assimilation loop
for t in timestep: 
    enx_f = np.zeros((mem, 3))  
    s = 0
    enseEn = []
    for i in range(mem):
        tspan2 = np.arange(t-dt, t + 0.001, 0.01)
        XX = x0_en[i]
        z = integrate.odeint(Vanderpol2, XX, tspan2)
        s += z
        enx_f[i, :] = z[-1, :]
        enseEn.append(z)            
    ssse.append(enseEn)
    exf.append(enx_f)
    mean = s / mem    
    emean.append(mean)   
    covariance = np.cov(np.transpose(enx_f))
    vhat = up[tim]
    tim += 1
    hdash = np.transpose(H)
    
    # Calculate Kalman gain
    kg = covariance.dot(hdash).dot(inv(H.dot(covariance).dot(hdash) + matrixR))
    ty = H.dot(np.transpose(enx_f))
    innovation = np.transpose(vhat) - ty
    xa = np.transpose(enx_f) + kg.dot(innovation)
    xa = np.transpose(xa)
    x0_en = xa
    
    # Stop updates if past a certain time
    if t >= 80:
        x0_en = enx_f
    
    # Compute analysis mean
    xamean = np.mean(xa, axis=0)
    enPmean.append(xamean)

# Plot results
fig = plt.figure()
ax = plt.axes()
plt.plot(tspan, xr[:, 0])  
for i in range(len(timestep)):
    plt.plot(timestep[i], posmean[i][0], '-o', color='blue') 
    plt.plot(timestep[i], enPmean[i][0], '-o', color='black') 
    plt.plot(timestep[i], u[i, 0], '-o', color='yellow') 

plt.legend()
plt.show()

# Plot ensemble trajectories and mean
fig = plt.figure()
ax = plt.axes()
for i in range(len(timestep)):
    for k in range(mem):
        plt.plot(ts[i][:], ssse[i][k][:, 0], color='grey')
    plt.plot(timestep[i], enPmean[i][0], '-o', color='black') 
    plt.plot(timestep[i], u[i, 0], '-o', color='yellow') 

fig = plt.figure()
ax = plt.axes()
for i in range(len(timestep)):
    for k in range(mem):
        plt.plot(ts[i][:], ssse[i][k][:, 0], color='grey')
    plt.plot(ts[i][:], emean[i][:, 0], '-o', color='black') 
    plt.plot(timestep[i], u[i, 0], '-o', color='yellow') 
