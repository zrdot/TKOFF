import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib, os
import matplotlib.pyplot as plt
import Throttle as trl
import altair as alt

MainPath = pathlib.Path(__file__).parent.parent.resolve()

def CreateNewModel(DataPath):
    Dataset = trl.PrepareData(DataPath, False)
    Dataset, history, model = trl.DefineAndTrainNN(Dataset, MainPath)
    trl.PlotOutputs(MainPath, history, Dataset)
    Dataset, Dataset_with_strings = trl.PrepareData(DataPath, True)

    Dataset['p'] = model.predict(Dataset['i'])
    Predicted = pd.DataFrame(Dataset['p'], index = Dataset['i'].index, columns = ['PREDICTED_N1'])
    Actual = pd.DataFrame(Dataset['d'], index = Dataset['d'].index, columns = ['N1'])
    Predicted['ERROR'] = abs((Actual.N1 - Predicted.PREDICTED_N1) / Actual.N1 * 100)
    print("\nMaximum Error: ", Predicted.ERROR.max(), "\n")

    FinalDataset = pd.concat([Dataset_with_strings, Predicted, Actual], axis=1)
    FinalDataset.to_csv(os.path.join(MainPath, 'Out/Visualizations/Data Out.csv'))

    CreateVisualizations(FinalDataset)    

def RunExistingModel(DataPath):
    ModelPath = os.path.join(MainPath, 'Out/Throttle Predictions/Models/')
    Dataset, Dataset_with_strings = trl.PrepareData(DataPath, True)
    model = tf.keras.models.load_model(ModelPath)

    Dataset['p'] = model.predict(Dataset['i'])
    Predicted = pd.DataFrame(Dataset['p'], index = Dataset['i'].index, columns = ['PREDICTED_N1'])
    Actual = pd.DataFrame(Dataset['d'], index = Dataset['d'].index, columns = ['N1'])
    Predicted['ERROR'] = abs((Actual.N1 - Predicted.PREDICTED_N1) / Actual.N1 * 100)
    print("\nMaximum Error: ", Predicted.ERROR.max(), "\n")

    FinalDataset = pd.concat([Dataset_with_strings, Predicted, Actual], axis=1)
    FinalDataset.to_csv(os.path.join(MainPath, 'Out/Visualizations/Data Out.csv'))

    CreateVisualizations(FinalDataset)


def CreateVisualizations(FinalDataset):
    SelectionAC = alt.selection_multi(fields = ['AIRCRAFT_TYPE'], bind = 'legend')
    chartAC = alt.Chart(FinalDataset).mark_circle(size = 15).encode(
                        x = alt.X('N1', title='Actual Value', scale = alt.Scale(domain=[75, 105])),
                        y = alt.Y('PREDICTED_N1', title='Predicted Value', scale = alt.Scale(domain=[75, 105])),
                        color = alt.Color('AIRCRAFT_TYPE', title = 'Aircraft Type'),
                        tooltip = [alt.Tooltip('AIRCRAFT_TYPE', title = "Aircraft"), alt.Tooltip('AIRPORT', title = "Airport"), alt.Tooltip('N1', title = "Throttle Setting", format=",.2f"), alt.Tooltip('PREDICTED_N1', title = "Predicted Setting", format=",.2f"), alt.Tooltip('ERROR', title = "Percent Error", format=",.2f")],
                        opacity = alt.condition(SelectionAC, alt.value(1), alt.value(0.01))
                ).add_selection(SelectionAC).interactive()

    SelectionAP = alt.selection_multi(fields = ['AIRPORT'], bind = 'legend')
    chartAP = alt.Chart(FinalDataset).mark_circle(size = 15).encode(
                        x = alt.X('N1', title='Actual Value', scale = alt.Scale(domain=[75, 105])),
                        y = alt.Y('PREDICTED_N1', title='Predicted Value', scale = alt.Scale(domain=[75, 105])),
                        color = alt.Color('AIRPORT', title = 'AIRPORT'),
                        tooltip = [alt.Tooltip('AIRCRAFT_TYPE', title = "Aircraft"), alt.Tooltip('AIRPORT', title = "Airport"), alt.Tooltip('N1', title = "Throttle Setting", format=",.2f"), alt.Tooltip('PREDICTED_N1', title = "Predicted Setting", format=",.2f"), alt.Tooltip('ERROR', title = "Percent Error", format=",.2f")],
                        opacity = alt.condition(SelectionAP, alt.value(1), alt.value(0.01))
                ).add_selection(SelectionAP).interactive()

    chartErrorAP = alt.Chart(FinalDataset).mark_bar().encode(
                        x = alt.X('AIRPORT', title='Airport'),
                        y = alt.Y('ERROR', title='Error'),
                        color = alt.Color('AIRPORT', title = 'AIRPORT'),
                        tooltip = [alt.Tooltip('AIRPORT', title = "Airport"), alt.Tooltip('ERROR', title = "Percent Error", format=",.2f")])

    chartErrorAC = alt.Chart(FinalDataset).mark_bar().encode(
                        x = alt.X('AIRCRAFT_TYPE', title='Aircraft'),
                        y = alt.Y('ERROR', title='Error'),
                        color = alt.Color('AIRCRAFT_TYPE', title = 'AIRCRAFT_TYPE'),
                        tooltip = [alt.Tooltip('AIRCRAFT_TYPE', title = "Aircraft"), alt.Tooltip('ERROR', title = "Percent Error", format=",.2f")])

    chartAP.save(os.path.join(MainPath, 'Out/Visualizations/Airport Chart.html'))
    chartAC.save(os.path.join(MainPath, 'Out/Visualizations/Aircraft Chart.html'))
    chartErrorAP.save(os.path.join(MainPath, 'Out/Visualizations/Airport Error.html'))
    chartErrorAC.save(os.path.join(MainPath, 'Out/Visualizations/Aircraft Error.html'))
    chartAP.show()
    chartAC.show()
    chartErrorAP.show()
    chartErrorAC.show()

###########################################################################################################################################
#                                                 RUN THE PROGRAM
###########################################################################################################################################
DataPath = 'C:/Users/Zayn.Roohi/Documents/OASIS/takeoff_distance_A320_A330_A340.csv'

#Call the different functions
CreateNewModel(DataPath)
#RunExistingModel(DataPath)