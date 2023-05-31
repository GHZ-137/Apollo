"""
Plot functions
"""

############################################
# Imports
############################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from functions_optimization import *

def plot(values, x_values, labels, err = [], y_inf = 0., y_sup = 1., d_x = 2, grid = True, grey = False, x_label ='', y_label ='', title = '', f_name = 'image.png', points = False):
    """
    Make a plot of lines of values and saves it to an image.
    Arguments:
        values - [y values]
        x_values - array of independent variable values
        labels - list of labels of each array
        y_inf, y_sup - min. and max. of y axis
        d_x - x ticks spacing
        grid - whether overimpose a dashed grid or not (default yes)
        f_name - image file name
    """
    colors = ['blue', 'green', 'orange']
    x_max = len(values[0])

    plt.clf()
    ax = plt.figure().gca()

    x_ticks = list( range(0, len(x_values)+1, d_x) )
    #print(x_ticks)
    x_labels = [0] + ["%.d" % (x_values[idx-1]+1) for idx in x_ticks[1:]]
    ax.set_xticks( x_ticks, minor = False)
    ax.set_xticklabels(x_labels, rotation = 90)
    

    if grid:
        ax.xaxis.grid(linestyle= 'dashed')
        ax.yaxis.grid(linestyle= 'dashed')
    plt.ylim( [y_inf, y_sup] )
    plt.xlim( left=0 )

    if x_label != '':
        plt.xlabel(x_label)
    if y_label != '':
        plt.ylabel(y_label)        

    for cont in range(len(values)):
        color = colors[ cont % len(colors)]
        if grey and cont ==0:
            color = 'blue' 
        if grey and cont !=0:
            color = (0.,0.,1., .2)

        if cont < len(labels):
            label = labels[cont]
        else:
            label = None
  
        if points:
            plt.plot(range(0, x_max), values[cont], label = label, linestyle='--', marker='o', color = color )
        else:
            plt.plot(range(0, x_max), values[cont], label = label, color = color )
        if err != []:
            #plt.errorbar(range(0, x_max), values[cont], err[cont], ecolor = 'black', barsabove = True, errorevery = d_x, label = label,  color = color )
            plt.fill_between(range(0, x_max), values[cont]- err[cont], values[cont] + err[cont], color = color, alpha = 0.25)
            
    plt.legend(loc="upper right")
    if title !='':
        plt.title(title)
    plt.tight_layout()
    
    print (f_name)
    plt.savefig(f_name)
    return

def scatter(values, x_values, labels, y_inf = 0., y_sup = 1., d_x = 2, grid = True, grey = False, x_label ='', y_label ='', title = '', f_name = 'image.png'):
    """
    Make a plot of lines of values and saves it to an image.
    Arguments:
        values - list of arrays of values
        x_values - array of independent variable values
        labels - list of labels of each array
        y_inf, y_sup - min. and max. of y axis
        d_x - spacing between x axis ticks
        grid - whether overimpose a dashed grid or not (default yes)
        f_name - image file name
    """
    colors = ['blue', 'green', 'orange']
    x_max = len(values[0])

    plt.clf()
    ax = plt.figure().gca()

    x_ticks = range(0, len(x_values), d_x)
    x_labels = ["%.2f" % x_values[idx] for idx in x_ticks]
    ax.set_xticks( x_ticks, minor = False)
    ax.set_xticklabels(x_labels, rotation = 90)
    

    if grid:
        ax.xaxis.grid(linestyle= 'dashed')
        ax.yaxis.grid(linestyle= 'dashed')
    plt.ylim( [y_inf, y_sup] )
    plt.xlim( left=0 )

    if x_label != '':
        plt.xlabel(x_label)
    if y_label != '':
        plt.ylabel(y_label)        

    for cont in range(len(values)):
        color = colors[ cont % len(colors)]
        if grey and cont ==0:
            color = 'blue' 
        if grey and cont !=0:
            color = (0.,0.,1., .2)

        if cont < len(labels):
            label = labels[cont]
        else:
            label = None
            
        plt.plot(range(0, x_max),values[cont], label = label,  linestyle='--', marker='o', color = color )
    plt.legend(loc="lower right")
    if title !='':
        plt.title(title)
    plt.tight_layout()
    
    print (f_name)
    plt.savefig(f_name)
    return



def plt_grid(f_name, grid, title = ''):
    """ Save grid plot of a 2D array """
    plt.clf()
    h, w = grid.shape
    m = np.min([w,h])

    ##plt.bar3d(x, y, bottom, width, depth, top, shade=True)
    
    #plt.figure(figsize = ((h/m) * 5., (w/m) * 5.))
    plt.pcolor(grid, cmap = 'Spectral_r', edgecolors = 'k')
    plt.colorbar()
    ###plt.clim(-0.01, None)
    ##plt.clim(None, 0.15)
    #plt.colorbar(fraction=0.04, pad=0.04)
    plt.title(title)
    plt.savefig(f_name)
    
    return

def plt_bar3d(f_name, grid, title, z_lim):
    """ Save 3d bar plot of a 2D array """
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.set_zlim(0., z_lim)
    h, w = grid.shape

    y = range(h)
    x = range(w)
    z = grid.flatten()
    xx, yy = np.meshgrid(x, y)
    X, Y = xx.ravel(), yy.ravel()
    height = np.zeros_like (z)
    

    dx = .5
    dy = .5
    dz = z
    ax1.bar3d(X, Y, height, dx, dy, dz, color='#00ceaa', shade=True)
    #ax1.set_xlabel('X')

    plt.title(title)
    plt.savefig(f_name)
    
    return


# Draw piecewise vars given a controller an path
def plot_gains(vars_, n_reg, controller, path):
    new_y1, new_y2, new_y3 = vars_
    F1, F2, F3 = controller.ref_data.get_gains()
    F1 = linearize(F1, n_reg, np.min(F1), np.max(F1), new_y = new_y1 )
    F2 = linearize(F2, n_reg, np.min(F2), np.max(F2), new_y = new_y2 )
    F3 = linearize(F3, n_reg, np.min(F3), np.max(F3), new_y = new_y3 )

    plt.clf()
    plt.scatter(controller.ref_data.X_and_lam[:,2], F1)
    plt.xlabel('v')
    plt.ylabel('F1')
    plt.grid(True)
    plt.title('F1')
    plt.savefig(path + '_F1.png')

    plt.clf()
    plt.scatter(controller.ref_data.X_and_lam[:,2], F2)
    plt.xlabel('v')
    plt.ylabel('F2')
    plt.grid(True)
    plt.title('F2')
    plt.savefig(path + '_F2.png')

    plt.clf()
    plt.scatter(controller.ref_data.X_and_lam[:,2], F3)
    plt.xlabel('v')
    plt.ylabel('F3')
    plt.grid(True)
    plt.title('F3')
    plt.savefig(path + '_F3.png')

    return 
