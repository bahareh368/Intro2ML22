# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from autograd import grad 
# import autograd-wrapped numpy
import autograd.numpy as np  # install: pip install autograd
#%matplotlib inline
import matplotlib.pyplot as plt
#import numpy as np


#inputs: weight_history (list of all w in the optimization), cost_history (list of the corresponding g(w)s), 
#      trajectory (boolean that determines whether we are plotting the optimization as a consecutive run or as sampled points)
def plot_descent_2d(g, weight_history, cost_history, trajectory = True):
    fig, ax = plt.subplots(1,2) #create a plot
    
    ws = np.array(weight_history) #cast the list to a numpy array
    
    #define values for the contour plot
    m1 = 0.1*np.mean(ws[:,0])
    m2 = 0.1*np.mean(ws[:,1])
    x, y = np.meshgrid(np.linspace(np.min(ws[:,0])-m1, np.max(ws[:,0])+m1, 100), 
                       np.linspace(np.min(ws[:,1])-m2, np.max(ws[:,1])+m2, 100))
    z = g(np.array([x,y]))
    
    #plot the contour and steps
    col = ax[1].contourf(x,y,z, levels = 10)
    ax[1].contour(x,y,z, levels = 10, colors = 'w', alpha = 0.2)
    ax[1].plot(ws[:,0], ws[:,1], 'rx', markersize = 10, markeredgewidth = 2)
    if trajectory == True: #also plot trajectory if applicable
        ax[1].plot(ws[:,0], ws[:,1], 'k--') 
        
    #formatting
    ax[1].set_xlabel(r'$w_1$')
    ax[1].set_ylabel(r'$w_2$')
    ax[1].set_title(r'Trajectory of descent')
    ax[1].set_xlim([np.min(ws[:,0])-m1, np.max(ws[:,0])+m1])
    ax[1].set_ylim([np.min(ws[:,1])-m2, np.max(ws[:,1])+m2])
    plt.colorbar(col)
        
    #plot the cost function    
    ax[0].plot(cost_history, 'rx', markersize = 10, markeredgewidth = 2)
    
    #also plot the trajectory if applicable and format the plot
    if trajectory == True: 
        ax[0].plot(cost_history, 'k--')
        ax[0].set_xlabel('k')
    else:
        ax[0].set_xlabel('point number (arbitrary)')
    ax[0].set_ylabel(r'$g(\mathbf{w})$')
    ax[0].set_title('Cost')
            
    plt.tight_layout() #this line ensures the axis labels don't overlap








