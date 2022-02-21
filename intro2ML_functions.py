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
    
    
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis] # enter your code here
    x_stds = np.std(x,axis = 1)[:,np.newaxis]  # enter your code here  

    # check to make sure thta x_stds > small threshold because we can not devide by zero, 
    # for those not, we divide by 1 instead of original standard deviation
    ind = np.argwhere(x_stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((x_stds.shape))
        adjust[ind] = 1.0
        x_stds += adjust

    # create standard normalizer function
    normalizer = lambda data: (data - x_means)/x_stds # your code here

    # create inverse standard normalizer
    inverse_normalizer = lambda data: data*x_stds + x_means # your code here

    # return normalizer 
    return normalizer,inverse_normalizer

from autograd import grad 

# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations),w
def gradient_descent(g,alpha,max_its,w):
    
    # compute gradient module using autograd
    gradient = grad(g)

    # gradient descent loop
    weight_history = [w]              # weight history container
    cost_history = [g(w)]             # cost function history container
    for k in range(max_its):
        
        # evaluate the gradient
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval
        
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
        
    return weight_history,cost_history








