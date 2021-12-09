import numpy as np
import torch
import torch.nn as nn

import Parameters

import fields.Trainer
import fields.Renderer
import fields.ModelGenerator
import fields.DataGenerator

if Parameters.ACTION == "generate":
    magnets, fields = fields.DataGenerator.gen(Parameters.TEST_SIZE)

    print(magnets.shape, fields.shape)

    np.save(Parameters.PATHS["magnets-test"], magnets)
    np.save(Parameters.PATHS["fields-test"], fields)
    
if Parameters.ACTION == "train":
    model, loss = fields.Trainer.train()
    
    torch.save(model.state_dict(), Parameters.PATHS["model"])
    
    indices = list(loss.keys())
    indices.sort()
    
    loss_np = np.zeros(len(loss))
    for i in indices:
        index = int(i)
        loss_np[index] = loss[index]
        
    np.savetxt(Parameters.PATHS["save-results"], loss_np)

if Parameters.ACTION == "render":
    if Parameters.MODEL_TYPE == "optnet":
        model = fields.ModelGenerator.OptNetEq(        
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM, 
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM,
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM
        )
    elif Parameters.MODEL_TYPE == "fc":
        model = fields.ModelGenerator.FC(            
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM, 
            Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM, 
            Parameters.FC_H_SIZE, 
            Parameters.N_HIDDEN
        )
        
    model.load_state_dict(torch.load(Parameters.PATHS["model"]))
    model = model.cuda()
    model.eval()
    
    trainFields = np.load(Parameters.PATHS["fields-test"])
    trainMagnets = np.load(Parameters.PATHS["magnets-test"])
    
    field = trainFields[Parameters.RENDER_INDEX]
    mag = trainMagnets[Parameters.RENDER_INDEX]
    
    fields.Renderer.renderField(field, mag)
    
    mags = trainMagnets[Parameters.RENDER_INDEX : Parameters.RENDER_INDEX + Parameters.BATCH_SIZE]
    mags = torch.Tensor(mags)
    mags = mags.double().to('cuda:0')
    
    pred = model(mags)
    
    loss = nn.MSELoss()(pred, mags)
    
    print("Model Type:", Parameters.MODEL_TYPE)
    print("Index:", Parameters.RENDER_INDEX)
    print("Loss:", loss.item())
    
    pred = pred.view(Parameters.BATCH_SIZE, Parameters.FIELD_HEIGHT, Parameters.FIELD_WIDTH, Parameters.FIELD_DIM)
    pred = pred[0]
    
    pred = pred.cpu().detach().numpy()
    
    fields.Renderer.renderField(pred, mag)