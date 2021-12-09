# optnet-magnet
The code used in my presentation on using OptNet to generate magnetic fields.

# Requirements
- numpy
- Pytorch
- matplotlib
- qtph
- magpylib

# Installation
Either git clone this repo, or download it directly.

# Instructions
To run the code, execute the command:
- python Application.py

on the commandline.

To select different actions or rendering options, modify the file:
- Parameters.py

This Parameters.py file has the following variables which can be set:
- N_HIDDEN : The number of hidden layers in the fully connected neural network.
- FC_H_SIZE : The number of nodes in each hidden layer of the fully connected neural network.
- EPOCHS : The number of epochs to train for.
- BATCH_SIZE : The batch size of the training process.
- LEARNING_RATE : The learning rate of the training process.
- INEQ_DEPTH : The height of the inequality constraints G and h for the OptNet layer.
- EQ_DEPTH : The height of the equality constraints A and b for the OptNet layer.
- TEST_SIZE : The number of datapoints to generate with magpylib.
- RENDER_INDEX : The index of the test data to render.
- MODEL_TYPE : The type of model to be used when the application is run (only options are "optnet" or "fc").
- ACTION : The action the program will take when the application is run (only options are "generate", "train", or "render").
- PATH :  A dictionary containing the various paths used in the application.

Note: the Pytorch models were not included in this github upload, as they are too large. You will need to retrain the models yourself if you wish to use them.

This is very easy to do - simply set the MODEL_TYPE parameter to the model type you wish to train, and then set the ACTION parameter to the "train" action. Then, run the Application.py file, and the program will automatically train and then save the model into the 'models' folder.