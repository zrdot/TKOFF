from gc import callbacks
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib, os

def PrepareData(DataInPath):
    RawData = pd.read_csv(DataInPath)
    LiftoffData = RawData[RawData['LIFTOFF'] == 1]
    #Change to make: Instead of dropping data, import only the columns we need
    #This will be a bit difficult as sometimes columns have the same name
    MLData = LiftoffData.drop(['TIME_OFFSET', 'FLIGHT_ID', 'FLIGHT_ID.1', 'STATED_SEGMENT_START_OF_TAKEOFF', 'APT_AIRCRAFT_RUNWAY_STAGE',
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
                            'STAGE_LENGTH_ID', 'STAGE_LENGTH_ID.1', 'DISTANCE_FROM_RUNWAY_END.1'], axis=1)

    MLData.rename(columns = {'ACTYPE.1' : 'ACTYPE'}, inplace = True)
    MLData.rename(columns = {'APT_AIRCRAFT_RUNWAY_STAGE.1' : 'APT_AIRCRAFT_RUNWAY_STAGE'}, inplace = True)
    
    #Change: Split the APT_AIRCRAFT_RUNWAY_STAGE column into 4 (and possible drop STAGE?)
    MLData = MLData.dropna()
    MLData = MLData.sample(frac = 1, replace = False)
    #Classifications don't do anything right now, will add them in next
    ClassificationData = MLData[['ACTYPE', 'APT_AIRCRAFT_RUNWAY_STAGE']]
    MLData = MLData.drop(['ACTYPE', 'APT_AIRCRAFT_RUNWAY_STAGE'], axis = 1)

    trn, val, test = np.split(MLData, [int(0.7*len(MLData)), int(0.85*len(MLData))], axis = 0)
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

def DefineAndTrainNN(Dataset, MainPath):
    #Create the model
    #Note - need to actually figure out the best settings here
    #Changes should be number of layers, numer of neurons, activation function
    #also need to look at learning rate, possibly add a kernal regularizer
    #Later on the number of epochs
    #Finally, decide if adam is really the best optimizer for this type of problem (I think it is?)
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
                filepath = os.path.join(MainPath, 'Out/Checkpoints/checkpoint'),
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
                min_lr = 0.000000000000000000000001)

    history = model.fit(
                Dataset['trn']['i'],
                Dataset['trn']['d'],
                batch_size = 400,
                epochs = 500,
                validation_data = (Dataset['val']['i'], Dataset['val']['d']),
                callbacks = [checkpoint_callback, learning_callback])

    model.load_weights(os.path.join(MainPath, 'Out/Checkpoints/checkpoint'))
    model.save(os.path.join(MainPath, 'Out/Models'))
    print('\n-----------Best Training Data:-----------\n')
    print(model.evaluate(Dataset['test']['i'], Dataset['test']['d']))


#def PlotOutputs():
    #Plot the output of the final best-fit model to see how good of a model it was
    #Don't necessarily need to do this as the R^2 does the same thing, but visualizations are always nice

#Actually run the program
MainPath = pathlib.Path(__file__).parent.parent.resolve()
DataPath = 'C:/Users/Zayn.Roohi/Documents/OASIS/TestDataLarge.csv' #takeoff_distance_A320_A330_A340.csv'

Dataset = PrepareData(DataPath)
DefineAndTrainNN(Dataset, MainPath)

#Simple  string data structures: https://www.tensorflow.org/tutorials/keras/regression
#Complex string data structures: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers