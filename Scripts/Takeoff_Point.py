import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib, os
import matplotlib.pyplot as plt

def PrepareData(DataInPath):
    RawData = pd.read_csv(DataInPath)
    #Change to make: Instead of dropping data, import only the columns we need
    #This will be a bit difficult as sometimes columns have the same name
    MLData = RawData[['SPEED_SOUND_START_EVENT', 'HEAD_WIND', 'MSL_ALT', 'AFE_ALT', 'TAS_START_EVENT', 'P64: True Airspeed at Liftoff (knots)', 'MACH_NUMBER_START_EVENT', 'GS_SEGMENT', 'TAS_SEGMENT', 'MACH_NUMBER_SEGMENT', 'DRAG', 'LIFT', 'LAT', 'LON', 'DISTANCE_START_EVENT', 'DISTANCE_END_EVENT', 'FUEL_QUANTITY', 'FUELFLOW_START_EVENT', 'FUELFLOW_SEGMENT', 'N1', 'THRUST_START_EVENT', 'THRUST_SEGMENT', 'LIFTOFF', 'Takeoff Runway Starting Latitude', 'Takeoff Runway Starting Longitude']]
    MLData = MLData.dropna()
    MLData = MLData.sample(frac = 1, replace = False)
    print(MLData.LIFTOFF)


    trn, val, test = np.split(MLData, [int(0.8*len(MLData)), int(0.9*len(MLData))], axis = 0)
    trn_independent = trn.drop('LIFTOFF', axis = 1)
    val_independent = val.drop('LIFTOFF', axis = 1)
    test_independent = test.drop('LIFTOFF', axis = 1)
    trn_dependent = trn.LIFTOFF
    val_dependent = val.LIFTOFF
    test_dependent = test.LIFTOFF

    FinalDataset = {'trn' :  {'i' : trn_independent,  'd' : trn_dependent},
                    'val' :  {'i' : val_independent,  'd' : val_dependent},
                    'test' : {'i' : test_independent, 'd' : test_dependent}}

    return FinalDataset

def DefineAndTrainNN(Dataset, MainPath):
    #Create the model
    normlayer = tf.keras.layers.Normalization()
    normlayer.adapt(Dataset['trn']['i'].to_numpy())

    model = tf.keras.Sequential([
                normlayer,
                tf.keras.layers.Dense(units = 8, activation = 'relu'),
                tf.keras.layers.Dense(units = 16, activation = 'relu'),
                tf.keras.layers.Dense(units = 32, activation = 'relu'),
                tf.keras.layers.Dense(units = 1, activation = 'linear')])

    model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
                loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                metrics = ['accuracy'])

    #Train the model now
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(MainPath, 'Out/Takeoff Point Predictions/Checkpoints/checkpoint'),
                save_weights_only = True,
                save_best_only = True,
                monitor = 'loss',
                mode = 'min')
    learning_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor = 'loss',
                factor = 0.1,
                patience = 5,
                verbose = 1,
                mode = 'min',
                min_delta = 0.0001,
                cooldown = 25,
                min_lr = 1E-10)

    history = model.fit(
                Dataset['trn']['i'],
                Dataset['trn']['d'],
                batch_size = 25000,
                epochs = 500,
                validation_data = (Dataset['val']['i'], Dataset['val']['d']),
                callbacks = [checkpoint_callback, learning_callback])

    model.load_weights(os.path.join(MainPath, 'Out/Takeoff Point Predictions/Checkpoints/checkpoint'))
    model.save(os.path.join(MainPath, 'Out/Takeoff Point Predictions/Models'))
    print('\n-----------Testing the Model:-----------\n')
    print(model.evaluate(Dataset['test']['i'], Dataset['test']['d']))
    Dataset['test']['p'] = model.predict(Dataset['test']['i'])

    return Dataset, history


def PlotOutputs(MainPath, history, Dataset):
    #Plot the loss over time
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MainPath, 'Out/Takeoff Point Predictions/Figures/Accuracy_Epoch.jpg'), dpi = 600)
    plt.close()

    #Plot predicted versus actual values
    plt.scatter(Dataset['test']['d'], Dataset['test']['p'], c = 'r', s = 4, label = 'Test Data', alpha = 0.009)
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.savefig(os.path.join(MainPath, 'Out/Takeoff Point Predictions/Figures/Actual_Predicted.jpg'), dpi = 600)
    plt.close()


###########################################################################################################################################
#                                                             RUN THE NEURAL NET
###########################################################################################################################################

MainPath = pathlib.Path(__file__).parent.parent.resolve()
DataPath = 'C:/Users/Zayn.Roohi/Documents/OASIS/takeoff_distance_A320_A330_A340.csv'

Dataset = PrepareData(DataPath)
Dataset, history = DefineAndTrainNN(Dataset, MainPath)
PlotOutputs(MainPath, history, Dataset)