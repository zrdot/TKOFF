import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def PrepareData(DataInPath):
    print("\nImporting Data...")
    RawData = pd.read_csv(DataInPath)
    print("Filtering & Processing Data...")
    LiftoffData = RawData[RawData['LIFTOFF'] == 1]
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
    
    MLData = MLData.dropna()
    MLData = MLData.sample(frac = 1, replace = False)

    #Need to figure out a way to not drop these string columns, but for now will keep it as this
    MLData = MLData.drop(['ACTYPE', 'APT_AIRCRAFT_RUNWAY_STAGE'], axis = 1)

    print("Preparing Data for ML...")

    trn, val, test = np.split(MLData, [int(0.7*len(MLData)), int(0.85*len(MLData))], axis = 0)


    trn_independent = trn.drop('N1', axis = 1)
    val_independent = trn.drop('N1', axis = 1)
    test_independent = trn.drop('N1', axis = 1)
    trn_dependent = trn.N1
    val_dependent = val.N1
    test_dependent = test.N1

    FinalDataset = {'trn' :  {'i' : trn_independent,  'd' : trn_dependent},
                    'val' :  {'i' : val_independent,  'd' : val_dependent},
                    'test' : {'i' : test_independent, 'd' : test_dependent}}

    return FinalDataset

def DefineNNArchitecture(Dataset):
    print("Creating ML Model...\n")
    normlayer = tf.keras.layers.Normalization()
    normlayer.adapt(Dataset['trn']['i'])

    model = tf.keras.Sequential([
        normlayer,
        tf.keras.layers.Dense(8, activation = 'tanh'),
        tf.keras.layers.Dense(1, activation = 'linear')])

    model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
                loss = 'mse',
                metrics = [tf.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tfa.metrics.r_square.RSquare()])

    return model

#def TrainModel():
    #Train the model - save the best fit model

#def PlotOutputs():
    #Plot the output of the final best-fit model to see how good of a model it was

#Actually run the program
#MainPath = pathlib.Path(__file__).parent.parent.resolve()
#OutPath = os.path.join(MainPath, "Out")
DataPath = 'C:/Users/Zayn.Roohi/Documents/OASIS/TestDataLarge.csv' #takeoff_distance_A320_A330_A340.csv'

Dataset = PrepareData(DataPath)
model = DefineNNArchitecture(Dataset)

#Simple data structures: https://www.tensorflow.org/tutorials/keras/regression
#Create data structures: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers