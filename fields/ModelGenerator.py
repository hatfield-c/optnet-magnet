
#import scipy.sparse as spa

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from qpth.qp import QPFunction

import Parameters

class OptNetEq(nn.Module):
    def __init__(self, inSize, outSize, hSize, Qpenalty = 0.1):
        super().__init__()
        
        self.Q = Variable(Qpenalty*torch.eye(outSize).double().cuda())
        
        networks = {}
        
        for c in [ "p", "G", "h", "A", "b" ]:
        
            layers = [
                torch.nn.Linear(inSize, hSize).double().cuda(),
                torch.nn.ELU().double().cuda(),
                torch.nn.Linear(hSize, hSize).double().cuda(),
                torch.nn.Linear(hSize, hSize).double().cuda(),
                torch.nn.ELU().double().cuda(),
                torch.nn.Linear(hSize, hSize).double().cuda()
            ]
            
            networks[c] = layers
            
        self.eq_depth = 50
            
        networks["p"].append(torch.nn.Linear(hSize, outSize).double().cuda())
        networks["G"].append(torch.nn.Linear(hSize, outSize * self.eq_depth).double().cuda())
        networks["h"].append(torch.nn.Linear(hSize, self.eq_depth).double().cuda())
        networks["A"].append(torch.nn.Linear(hSize, outSize * self.eq_depth).double().cuda())
        networks["b"].append(torch.nn.Linear(hSize, self.eq_depth).double().cuda())
        
        self.p_net = torch.nn.ModuleList(networks["p"])
        self.G_net = torch.nn.ModuleList(networks["G"])
        self.h_net = torch.nn.ModuleList(networks["h"])
        self.A_net = torch.nn.ModuleList(networks["A"])
        self.b_net = torch.nn.ModuleList(networks["b"])
        
        self.networks = {
            "p": self.p_net,
            "G": self.G_net,
            "h": self.h_net,
            "A": self.A_net,
            "b": self.b_net,
        }

    def forward(self, magnets):

        data = magnets.view(Parameters.BATCH_SIZE, -1).double().cuda()

        params = { "p": data, "G": data, "h": data, "A": data, "b": data }

        for param in params:
            network = self.networks[param]
            
            for layer in network:
                params[param] = layer(params[param])

        y = QPFunction(verbose=-1)(
            self.Q, 
            params["p"], 
            params["G"].view(Parameters.BATCH_SIZE, self.eq_depth, Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM), 
            params["h"],
            params["A"].view(Parameters.BATCH_SIZE, self.eq_depth, Parameters.FIELD_HEIGHT * Parameters.FIELD_WIDTH * Parameters.FIELD_DIM),
            params["b"]
        ).view(Parameters.BATCH_SIZE, Parameters.FIELD_HEIGHT, Parameters.FIELD_WIDTH, Parameters.FIELD_DIM)

        return y

class FC(nn.Module):
    def __init__(self, inSize, outSize, hSize, nHidden):
        super().__init__()
        
        layers = []
        
        for i in range(nHidden):
            if i == 0:
                fc = torch.nn.Linear(inSize, hSize).double().cuda()
            elif i == nHidden - 1:
                fc = torch.nn.Linear(hSize, outSize).double().cuda()
            else:
                fc = torch.nn.Linear(hSize, hSize).double().cuda()
            
            layers.append(fc)
            
            if i % 2 == 0:
                elu = torch.nn.ELU().double().cuda()
                layers.append(elu)
            
        self.nn = torch.nn.ModuleList(layers).double().cuda()

    def forward(self, magnets):        
        
        x = magnets.view(Parameters.BATCH_SIZE, -1)
        for layer in self.nn:
            x = layer(x)

        return x.view(Parameters.BATCH_SIZE, Parameters.FIELD_HEIGHT, Parameters.FIELD_WIDTH, Parameters.FIELD_DIM)
