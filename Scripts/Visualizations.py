import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib, os
import matplotlib.pyplot as plt
import Throttle as trl

MainPath = pathlib.Path(__file__).parent.parent.resolve()
ModelPath = os.path.join(MainPath, 'Best Models/Distance')
DataPath = 'C:/Users/Zayn.Roohi/Documents/OASIS/takeoff_distance_A320_A330_A340.csv'

#########################################################################################################
Dataset, Dataset_with_strings = trl.PrepareData(DataPath, True)

model = tf.keras.models.load_model(ModelPath)
Dataset['p'] = model.predict(Dataset['i'])
PredictedValues = pd.DataFrame(Dataset['p'], index = Dataset['d'].index)

FinalDataset = pd.concat([Dataset_with_strings, PredictedValues], axis=1)

FinalDataset.to_csv(os.path.join(MainPath, 'Out/Visualizations/Dataout.csv'))