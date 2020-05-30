# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib import animation
from scipy.integrate import solve_ivp
from timeit import default_timer as timed

# positions on x-axis of particles
positions = np.hstack((np.arange(-0.001875*320, 0, 0.001875), np.arange(0.0075, 0.0075*81, 0.0075)))
# density, velocity, energy, and pressure of all particles
properties = np.vstack((np.repeat([[1,0,2.5,1]], 320, axis=0), np.repeat([[0.25,0,1.795,0.1795]], 80, axis=0)))
# create data from positions and properties
data = np.append(positions.reshape([400,1]), properties, axis=1)
# mass of 1 particle
mass = 0.001875

# create combinations of i and j to avoid using loops
combinations = np.array(list(product(np.arange(408), np.arange(408))))
i, j = combinations[:,0], combinations[:,1]

def FORCE(t, data):
    data = data.reshape(400, -1)
    
    # create and append virtual particles
    positions1 = np.arange(-0.001875*324, -0.001875*320, 0.001875)
    positions2 = np.arange(0.0075*81, 0.0075*84, 0.0075)
    properties1 = np.vstack((np.repeat([[10,0,5,10]], 1, axis=0), np.repeat([[3,0,2.5,3]], 1, axis=0), np.repeat([[1,0,2.5,1]], 2, axis=0)))
    properties2 = np.vstack((np.repeat([[0.25,0,1.795,0.1795]], 2, axis=0), np.repeat([[3,0,2.5,3]], 1, axis=0), np.repeat([[10,0,2.5,10]], 1, axis=0)))    
    virtual1 = np.append(positions1.reshape([len(positions1),1]), properties1, axis=1)    
    virtual2 = np.append(positions2.reshape([len(positions2),1]), properties2, axis=1)  
    data2 = np.vstack((virtual1, data, virtual2))
    
    position = data2[:,0]
    density = data2[:,1]
    velocity = data2[:,2]
    energy = data2[:,3]
    pressure = data2[:,4]
    
    h = 0.75 * (mass / density[i] + mass / density[j])
    dx = position[i] - position[j]
    R = np.abs(dx) / h
    
    # masks for calculating w and dw/dx
    r1 = R < 1
    r2 = np.logical_and(R >= 1, R < 2)
    
    # values for w
    w = np.zeros(len(data2)**2)
    w[r1] = (2/3 - R[r1]**2 + 0.5 * R[r1]**3) / h[r1]
    w[r2] = 1/6 * (2 - R[r2])**3 / h[r2]
    
    # values for dw/dx
    dwdx = np.zeros(len(data2)**2)
    dwdx[r1] = (- 2 + 3/2 * R[r1]) * dx[r1] / h[r1]**2 / h[r1]
    dwdx[r2] = - 0.5 * (2 - R[r2])**2 * dx[r2] / h[r2] / np.abs(dx[r2]) / h[r2]
    
    # ingredients for viscosity
    dv = velocity[i] - velocity[j]
    c = 0.5 * (np.sqrt(0.4 * energy[i]) + np.sqrt(0.4 * energy[j]))
    rho = 0.5 * (density[i] + density[j])
    phi = h * dv * dx / (np.abs(dx)**2 + (0.1 * h)**2)
    
    # mask for calculating viscosity
    filtPI = dx * dv < 0    
    # values for viscosity
    PI = np.zeros(len(data2)**2)
    PI[filtPI] = (- c[filtPI] * phi[filtPI] + (phi[filtPI])**2 ) / rho[filtPI]
    
    # calculate change in velocity
    dvdt = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * dwdx
    # calculate change in energy
    dedt = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * (velocity[i] - velocity[j]) * dwdx
    
    # calculate density and assign to old data
    data[:,1] = (np.sum((mass * w).reshape(len(data2), -1), axis=1))[4:-4]
    # calculate pressure and assign to old data
    data[:,4] = 0.4 * data[:,1] * data[:,3]
    
    new = np.zeros((400, 5))
    # change in position is velocity
    new[:,0] = data[:,2]
    # change in density is zero
    new[:,1] = 0
    # reshape and sum change in velocity
    new[:,2] = (- np.sum(dvdt.reshape(len(data2), -1), axis=1))[4:-4]
    # reshape and sum change in energy
    new[:,3] = (np.sum(dedt.reshape(len(data2), -1), axis=1) * 0.5)[4:-4]
    # change in pressure is zero
    new[:,4] = 0

    return new.flatten()

runtime = 1.2

time = timed()
sol = solve_ivp(FORCE, [0, runtime], data.flatten())
print(timed()-time)

a, b = np.shape(sol.y)
results = sol.y.reshape((400, 5, b))

# %%

titles = ['Density', 'Velocity', 'Energy', 'Pressure']
ylabels = [r'Density (Kg/m$^2$)', 'Velocity (m/s)', 'Internal energy (J/kg)', r'Pressure (N/m$^2$)']
ylims = [[0, 1.4], [-1, 1.5], [1.4, 3.5], [0, 1.2]]
fig, axs = plt.subplots(2, 2, figsize=(9,9))
fig.subplots_adjust(hspace=.3, wspace=.35)
axs = axs.ravel()
plot_time = 30

for i in range(4):
    axs[i].plot(data[:,0], data[:,i+1], c='black', linestyle='--', linewidth=1, label='initial')
    axs[i].plot(results[:,0,-1], results[:,i+1,-1], c='black', linewidth=1, label=(str(runtime) + ' s'))
    axs[i].set_xlabel('x (m)')
    axs[i].set_ylabel(ylabels[i])
    axs[i].set_xlim([-0.6, 0.6])
    axs[i].set_ylim(ylims[i])
    axs[i].grid(which='both')
    # axs[i].set_title(titles[i])
    axs[i].legend()
    
plt.savefig(f'boundary{str(runtime)}.png', bbox_inches='tight')

# %%
fig = plt.figure(figsize=(10,7))
fig.suptitle('Density')
ax = plt.axes(xlim=(-0.7, 0.7), ylim=(-0.25, 1.5))
line, = ax.plot([], [], 'c')
def animate(i):
    line.set_data(results[:,0,i], results[:,1,i])
    return line,
anim = animation.FuncAnimation(fig, animate, frames=b, interval=1)
# anim.save('density.mp4')
# %%
fig = plt.figure(figsize=(10,7))
fig.suptitle('Velocity')
ax = plt.axes(xlim=(-0.7, 0.7), ylim=(-1, 1.5))
line, = ax.plot([], [], 'c')
def animate(i):
    line.set_data(results[:,0,i], results[:,2,i])
    return line,
anim = animation.FuncAnimation(fig, animate, frames=b, interval=10)
# anim.save('velocity.mp4')
# %%
fig = plt.figure(figsize=(10,7))
fig.suptitle('Energy')
ax = plt.axes(xlim=(-0.7, 0.7), ylim=(1, 8))
line, = ax.plot([], [], 'c')
def animate(i):
    line.set_data(results[:,0,i], results[:,3,i])
    return line,
anim = animation.FuncAnimation(fig, animate, frames=b, interval=10)
# anim.save('energy.mp4')
# %%
fig = plt.figure(figsize=(10,7))
fig.suptitle('Pressure')
ax = plt.axes(xlim=(-0.7, 0.7), ylim=(-0.5, 1.5))
line, = ax.plot([], [], 'c', markersize=5)
def animate(i):
    line.set_data(results[:,0,i], results[:,4,i])
    return line,
anim = animation.FuncAnimation(fig, animate, frames=b, interval=10)
# anim.save('pressure.mp4')