# Tracker_EIC_Recon
Code for the reconstruction of a far forward scattered electron tracker at the EIC.

# Parsing Data

ProcessTaggerG4Graph.C is a ROOT based script that allows to parse the output of a Geant4 simulation of the ePIC detector into ROOT tree containing data relevant to the object condensation method.

parseData.py allows to parse the output of ProcessTaggerG4Graph.C into tensorflow ragged arrays containing the hit information (X,Y position, layer, time, energy deposition, module) and the truth information (object ID, noise label, Px, Py, Pz, is quasi-real, is brehmsstralung).

Both steps are required to convert the Geant4 output into training data for the object condensation method.

# data_functions

Set of functions that allow to read in the training data, add inneficiencies or noise, combine several files into one to increase the number of events per file.

The scripts combine_events, add_efficiency_to_data, add_noise then use these functions.

# dataset_functions

Set of functions to make training and testing sets from the training data. Some normalisation is applied, variables such as energy deposition or time are removed, shuffle events so that hits from a same track aren't following each other, etc. 

Note: The training and testing set are currently fixed sized arrays with some padding (all 0) added to the truth and hit ragged arrays created above. This is a shortcut due to not being able to find code for GNet layers that supports ragged arrays.

# calc_metrics

Code to calculate the tracking, momentum prediction and PID prediction metrics (eg tracking efficiency and purity).

# plot_functions, plot_functions_trainTest

Set of functions to make plots, plot_functions is mainly concerned with plotting the data (ie P, Theta, Phi, number of tracks per event etc). This is read in by the plot_data script.

plot_functions_trainTest has code to plot the efficiency and purity in various ways, or plotting the tracker, latent space etc...

# track_building

Code to make track from hits using the object condensation model, or using the truth info to make true tracks. Used when testing the model.

# model_functions 

Code to define the model architecture and load a model from saved files. Note that to load a saved model you need to know the model architecture. Note also that you will have to change the input size if you change the padding in the training data.

# Layers, caloGraphNN, garnet

Code taken from https://github.com/jkiesele/SOR for the implementation of GNet layers. Please give credit where credit is due.

# betaLosses

Code adapted from https://github.com/jkiesele/SOR to calculate the object condensation loss. Note that if you want to add PID or momentum prediction you need to scroll to the bottom of this file and change the loss that is outputted.

# object_condensation_functions

Code to train the object condensation model, read in by object_condensation_main. Will also test the model as a function of epoch. Note that we train in batches of epochs (ie 15 batches of 40 epochs). The trained model will be saved at each batch to a directory called models

# object_condensation_testing

Code to apply some tests to a trained model (eg metrics as a function of the number of noise hits per event). Read in by object_condensation_mainTesting.

# best_models

Some saved models. 

_combinedEvents_wInEff_noised uses the output of the combine_events, add_efficiency_toData and add_noise_toData scripts. Recommend ep 4.

# Dependencies

The file environment_multiPlatform.yml contains the list of dependencies, with the file environment.yml specifying the version. You can try creating your own conda environment from these files using conda env create -f environment.yml. Pay special attention to the tensorflow-gpu version (2.2.0).