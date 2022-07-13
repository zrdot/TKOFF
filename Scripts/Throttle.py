import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
import matplotlib.pyplot as plt

def PrepareData(DataInPath, viz):
    RawData = pd.read_csv(DataInPath)
    LiftoffData = RawData[RawData['LIFTOFF'] == 1]
    LiftoffData = LiftoffData.set_index('FLIGHT_ID')

    #Change to make: Instead of dropping data, import only the columns we need
    #This will be a bit difficult as sometimes columns have the same name
    MLData = LiftoffData.drop(['TIME_OFFSET', 'FLIGHT_ID.1', 'STATED_SEGMENT_START_OF_TAKEOFF', 'APT_AIRCRAFT_RUNWAY_STAGE',
                            'LIFTOFF', 'TIME_ON_GROUND_BEFORE_LIFTOFF_(SECONDS)', 'P64: Duration of Taxi Out (Minutes)',
                            'DURATION','SPEED_SOUND_START_EVENT', 'AFE_ALT', 'TAS_START_EVENT', 'P64: True Airspeed at Liftoff (knots)',
                            'GS_SEGMENT', 'TAS_SEGMENT', 'MACH_NUMBER_SEGMENT', 'LAT', 'LON', 'DISTANCE_START_EVENT', 'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF',
                            'DISTANCE_END_EVENT', 'FUELFLOW_SEGMENT', 'THRUST_SEGMENT', 'LIFTOFF', 'APT_CODE', 'P64  Air Temperature (total best available) at Start of Event (library) (Deg Celsius)',
                            'Takeoff Runway Starting Latitude', 'Takeoff Runway Starting Longitude', 'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF',
                            'SHARE_OF_REMAINING_RUNWAY_LENGTH_AT_DETECTED_AIRBORNE_POINT_IN_RUNWAY_VICINITY', 'FEET_OF_REMAINING_RUNWAY_LENGTH_AT_DETECTED_AIRBORNE_POINT_IN_RUNWAY_VICINITY',
                            'RUNWAY_LENGTH', 'DISPLACED_THRESHOLD', 'DEPARTURE_AIRPORT_CODE', 'RUNWAY_END', 'EFFECTIVE_RUNWAY_LENGTH',
                            'SHARE_OF_REMAINING_EFFECTIVE_RUNWAY_LENGTH_AT_DETECTED_AIRBORNE_POINT_IN_RUNWAY_VICINITY', 'FEET_OF_REMAINING_EFFECTIVE_RUNWAY_LENGTH_AT_DETECTED_AIRBORNE_POINT_IN_RUNWAY_VICINITY',
                            'DISTANCE_FROM_THRESHOLD_AT_POINT_ONE', 'DISTANCE_FROM_POINT_ONE_FEET', 'GROUNDSPEED_AT_POINT_ONE_KNOTS',
                            'WELL-BEHAVED_TRAJECTORY', 'LATITUDE_AT_POINT_ONE', 'LONGITUDE_AT_POINT_ONE', 'LATITUDE_AT_DETECTED_AIRBORNE_POINT_IN_RUNWAY_VICINITY',
                            'LONGITUDE_AT_DETECTED_AIRBORNE_POINT_IN_RUNWAY_VICINITY', 'ACTYPE', 'P64  Air Temperature (outside) at Start of Event (library) (Deg Celsius)',
                            'STAGE_LENGTH_ID', 'STAGE_LENGTH_ID.1', 'DISTANCE_FROM_RUNWAY_END.1', 'ACTYPE.1'], axis=1)

    MLData[['AIRPORT', 'AIRCRAFT_TYPE', 'RUNWAY', 'STAGE']] = MLData['APT_AIRCRAFT_RUNWAY_STAGE.1'].str.split('_', expand=True)
    MLData = MLData.drop('APT_AIRCRAFT_RUNWAY_STAGE.1', axis = 1)
    MLData = MLData.dropna()
    MLData = MLData.sample(frac = 1, replace = False)

    StringDataset = MLData.copy()

    MLData['AIRPORT'] = pd.factorize(MLData.AIRPORT)[0] + 1
    MLData['AIRCRAFT_TYPE'] = pd.factorize(MLData.AIRCRAFT_TYPE)[0] + 1
    MLData['RUNWAY'] = pd.factorize(MLData.RUNWAY)[0] + 1
    MLData['STAGE'] = pd.factorize(MLData.STAGE)[0] + 1

    if viz == False:
        trn, val, test = np.split(MLData, [int(0.8*len(MLData)), int(0.9*len(MLData))], axis = 0)
        trn_independent = trn.drop('N1', axis = 1)
        val_independent = val.drop('N1', axis = 1)
        test_independent = test.drop('N1', axis = 1)
        trn_dependent = trn.N1
        val_dependent = val.N1
        test_dependent = test.N1

        FinalDataset = {'trn' :  {'i' : trn_independent,  'd' : trn_dependent},
                        'val' :  {'i' : val_independent,  'd' : val_dependent},
                        'test' : {'i' : test_independent, 'd' : test_dependent}}

        return FinalDataset

    if viz == True:
        Dataset = {'i' : MLData.drop('N1', axis = 1), 'd' : MLData.N1}
        return Dataset, StringDataset.reindex(Dataset['i'].columns, axis=1)


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
                loss = 'mse',
                metrics = [tf.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tfa.metrics.r_square.RSquare()])


    #Train the model now
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(MainPath, 'Out/Throttle Predictions/Checkpoints/checkpoint'),
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
                batch_size = 400,
                epochs = 500,
                validation_data = (Dataset['val']['i'], Dataset['val']['d']),
                callbacks = [checkpoint_callback, learning_callback])

    model.load_weights(os.path.join(MainPath, 'Out/Throttle Predictions/Checkpoints/checkpoint'))
    model.save(os.path.join(MainPath, 'Out/Throttle Predictions/Models/'))
    print('\n-----------Testing the Model:-----------\n')
    print(model.evaluate(Dataset['test']['i'], Dataset['test']['d']))
    Dataset['test']['p'] = model.predict(Dataset['test']['i'])

    return Dataset, history, model


def PlotOutputs(MainPath, history, Dataset):
    #Plot the loss over time
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MainPath, 'Out/Throttle Predictions/Figures/Loss_Epoch.jpg'), dpi = 600)
    plt.close()

    #Plot predicted versus actual values
    plt.scatter(Dataset['test']['d'], Dataset['test']['p'], c = 'r', s = 4, label = 'Test Data')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.savefig(os.path.join(MainPath, 'Out/Throttle Predictions/Figures/Actual_Predicted.jpg'), dpi = 600)
    plt.close()