

#import optnet.sudoku.models

#model = optnet.sudoku.models.OptNetEq(2, 0.1, "qpth")

import magpylib as mag3
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

def gen(n):
    
    granularity = 20
    
    magnets = []    
    fields = []
    
    minBound = -10
    maxBound = 10
    
    avgTime = 0
    for i in range(n):
        
        start = time.time()
        
        xPos = random.randint(minBound + 2, maxBound - 2)
        zPos = random.randint(minBound + 2, maxBound - 2)
        
        s = mag3.magnet.Cylinder(magnetization=(0,0,350), dimension=(4,4), position = (xPos, 0, zPos))
    
        xs = np.linspace(minBound, maxBound, granularity)
        zs = np.linspace(minBound, maxBound, granularity)
        
        B_field = np.array([[s.getB([x,0,z]) for x in xs] for z in zs])
        B_field = B_field[:, :, [0, 2]]
        
        xPosMapped = xPos + int(abs(minBound - maxBound) / 2)
        zPosMapped = zPos + int(abs(minBound - maxBound) / 2)
        
        mag_field = np.zeros(shape = (20, 20, 2))
        for j in range(20):
            for k in range(20):
                mag_field[j, k] = np.array([xPosMapped, zPosMapped]) - np.array([j, k])

        magnets.append(mag_field)
        fields.append(B_field)
        
        avgTime = (avgTime + (time.time() - start)) / 2
        
        if i % math.ceil(n / 10) == 0:
            
            print("[", i, "] Completed. Average generation time:", avgTime)
        
    magnets = np.array(magnets)
    fields = np.array(fields)
    
    return magnets, fields