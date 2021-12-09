
import matplotlib.pyplot as plt
import numpy as np

def renderField(field, magnet):
    
    xs = np.linspace(-10, 10, 20)
    zs = np.linspace(-10, 10, 20)
        
    fig2, ax = plt.subplots()
    X,Z = np.meshgrid(xs,zs)
    U,V = field[:,:,0], field[:,:,1]
    
    xNorth, zNorth, xSouth, zSouth = getMarkers(magnet)
    
    ax.streamplot(X, Z, U, V, color=np.log(U**2+V**2), density=2)
    
    plt.plot(xNorth, zNorth, marker = "o", color = "blue", markersize = 12, linewidth = 0)
    
    plt.show()
    
def getMarkers(magnet):
    height = magnet.shape[0]
    width = magnet.shape[1]
    
    xNorth = []
    zNorth = []
    
    xSouth = []
    zSouth = []

    for i in range(height):
        for j in range(width):
            if magnet[i, j, 0] == 0 and magnet[i, j, 1] == 0:
                xNorth.append(i - 10)
                zNorth.append(j - 10)
                
    return xNorth, zNorth, xSouth, zSouth