import os
import numpy as np

from pymooCFD.setupOpt import dataDir, mapDir, obj_labels, var_labels
from pymooCFD.util.handleData import loadCP

def mapGen1(heatmap=False):

    ########################################################################################################################
    checkpointFile = os.path.join(dataDir, 'checkpoint-gen1.npy')
    alg = loadCP(checkpointFile)    
    ########################################################################################################################
    X = alg.pop.get('X')
    F = alg.pop.get('F')
    ########################################################################################################################
    ##### SCATTER PLOTS #######
    ###########################
    from pymoo.visualization.scatter import Scatter
    # https://pymoo.org/visualization/scatter.html

    ##### Variable vs. Objective Plots ######
    # extract objectives and variables columns and plot them against each other
    for x_i, x in enumerate(X.transpose()):
        for f_i, f in enumerate(F.transpose()):
            plot = Scatter(title=f'{var_labels[x_i]} vs. {obj_labels[f_i]}',
                            labels=[var_labels[x_i], obj_labels[f_i]]
                            )
            xy = np.column_stack((x,f))
            plot.add(xy)
            # best fit line
            # for d in range(3):
            #     m, b = np.polyfit(x, f, d)
            #     xy = np.column_stack((x, m*x+b))
            #     plot.add(xy)
            m, b = np.polyfit(x, f, 1)
            xy = np.column_stack((x, m*x+b))
            plot.add(xy)
            
            plot.save(os.path.join(mapDir, f'{var_labels[x_i].replace(" ", "_")}-vs-{obj_labels[f_i].replace(" ", "_")}.png'))
            plot.show()

    # if there are more than 2 objectives create array of scatter plots comparing
    # the trade-off between 2 objectives at a time
    if len(F.transpose()) > 2:
        ####### Pair Wise Objective Plots #######
        # Pairwise Scatter Plots of Function Space
        plot = Scatter(tight_layout=True)
        plot.add(F, s=10)
        plot.add(F[-1], s=30, color="red")
        plot.save(os.path.join(mapDir, 'pairwise-scatter.png'))
        plot.show()


