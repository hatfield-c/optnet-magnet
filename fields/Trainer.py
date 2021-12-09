import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import Parameters
import fields.ModelGenerator

def train():
    if Parameters.MODEL_TYPE == "optnet":
        model = fields.ModelGenerator.OptNetEq(        
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM, 
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM,
            Parameters.FC_H_SIZE
        )
    elif Parameters.MODEL_TYPE == "fc":
        model = fields.ModelGenerator.FC(
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM, 
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM, 
            Parameters.FC_H_SIZE, 
            Parameters.N_HIDDEN
        )
        
    model = model.cuda()

    trainFields = np.load(Parameters.PATHS["fields-train"])
    trainMagnets = np.load(Parameters.PATHS["magnets-train"])
    
    trainFields = torch.from_numpy(trainFields)
    trainMagnets = torch.from_numpy(trainMagnets)
    
    dataCount = trainMagnets.size(0)
    batchCount = int(dataCount / Parameters.BATCH_SIZE)
    
    optimizer = optim.Adam(model.parameters(), lr = Parameters.LEARNING_RATE)
    
    avgTime = 0
    start_e = time.time()
    start_total = time.time()
    
    loss_data = {}

    for e in range(Parameters.EPOCHS):
        
        print("Epoch", e)
        print("  Time per epoch:", time.time() - start_e)
        
        start_e = time.time()
        
        batch_data_t = torch.DoubleTensor(Parameters.BATCH_SIZE, Parameters.FIELD_HEIGHT, Parameters.FIELD_WIDTH, Parameters.FIELD_DIM)
        batch_targets_t = torch.DoubleTensor(Parameters.BATCH_SIZE, Parameters.FIELD_HEIGHT, Parameters.FIELD_WIDTH, Parameters.FIELD_DIM)
        
        batch_data_t = batch_data_t
        batch_targets_t = batch_targets_t
        
        batch_data = Variable(batch_data_t, requires_grad=False)
        batch_targets = Variable(batch_targets_t, requires_grad=False)        
        
        for b in range(0, dataCount, Parameters.BATCH_SIZE):
            start_b = time.time()
            
            indices = []
            for i in range(Parameters.BATCH_SIZE):
                index = random.randrange(0, dataCount)
                indices.append(index)

            batch_data.data[:] = trainMagnets[indices]
            batch_targets.data[:] = trainFields[indices]
            
            batch_data = batch_data.data.cuda()
            batch_targets = batch_targets.cuda()
            
            optimizer.zero_grad()
            preds = model(batch_data)
            loss = nn.MSELoss()(preds, batch_targets)
            loss.backward()
            optimizer.step()

            batchIndex = int(b / Parameters.BATCH_SIZE)

            avgTime = (avgTime + (time.time() - start_b)) / 2
            eta = (avgTime * (batchCount - batchIndex))  + (avgTime * batchCount * (Parameters.EPOCHS - (e + 1)))

            if batchIndex % 10 == 0 or batchIndex == batchCount - 1:
                print("   [", e, ":", b,  "]")
                print("     Type     :", Parameters.MODEL_TYPE)
                print("     Loss     :", loss.item())
                print("     Avg. Time:", avgTime)
                print("     ETA      :", int(eta / 60), "mins")
                print("\n")
                
        loss_data[e] = loss.cpu().item()
            
    print("\nCompleted in", int((time.time() - start_total) / 60), "minutes.")
        
    return model, loss_data
