import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib, os
import matplotlib.pyplot as plt
import Throttle as trl
import Takeoff_Distance as tkd
import altair as alt

MainPath = pathlib.Path(__file__).parent.parent.resolve()

def CreateNewModel(DataPath, ScriptToRun):
    if ScriptToRun == 'Throttle':
        Dataset = trl.PrepareData(DataPath, False)
        Dataset, history, model = trl.DefineAndTrainNN(Dataset, MainPath)

        trl.PlotOutputs(MainPath, history, Dataset)
        Dataset, Dataset_with_strings = trl.PrepareData(DataPath, True)

        Dataset['p'] = model.predict(Dataset['i'])
        Predicted = pd.DataFrame(Dataset['p'], index = Dataset['i'].index, columns = ['PREDICTED_N1'])
        Actual = pd.DataFrame(Dataset['d'], index = Dataset['d'].index, columns = ['N1'])
        Predicted['ERROR'] = abs((Actual.N1 - Predicted.PREDICTED_N1) / Actual.N1 * 100)
        Predicted['ABSOLUTE_ERROR'] = abs(Actual.N1 - Predicted.PREDICTED_N1)
        print("\nMaximum Error: ", Predicted.ERROR.max(), "Percent\n")

        FinalDataset = pd.concat([Dataset_with_strings, Predicted, Actual], axis=1)
        FinalDataset.to_csv(os.path.join(MainPath, 'Out/Visualizations/Throttle Data Out.csv'))

        CreateVisualizations(FinalDataset, ScriptToRun)

    elif ScriptToRun == 'Takeoff Distance':
        Dataset = tkd.PrepareData(DataPath, False)
        Dataset, history, model = trl.DefineAndTrainNN(Dataset, MainPath)

        tkd.PlotOutputs(MainPath, history, Dataset)
        Dataset, Dataset_with_strings = tkd.PrepareData(DataPath, True)

        Dataset['p'] = model.predict(Dataset['i'])
        Predicted = pd.DataFrame(Dataset['p'], index = Dataset['i'].index, columns = ['PREDICTED_DISTANCE_FROM_RUNWAY_END'])
        Actual = pd.DataFrame(Dataset['d'], index = Dataset['d'].index, columns = ['DISTANCE_FROM_RUNWAY_END'])
        Predicted['ERROR'] = abs((Actual.DISTANCE_FROM_RUNWAY_END - Predicted.PREDICTED_DISTANCE_FROM_RUNWAY_END) / Actual.DISTANCE_FROM_RUNWAY_END * 100)
        Predicted['ABSOLUTE_ERROR'] = abs(Actual.DISTANCE_FROM_RUNWAY_END - Predicted.PREDICTED_DISTANCE_FROM_RUNWAY_END)
        print("\nMaximum Error: ", Predicted.ERROR.max(), "Percent\n")

        FinalDataset = pd.concat([Dataset_with_strings, Predicted, Actual], axis=1)
        FinalDataset.to_csv(os.path.join(MainPath, 'Out/Visualizations/Takeoff Distance Data Out.csv'))

        CreateVisualizations(FinalDataset, ScriptToRun)
    else:
        print('Please enter either \'Throttle\' or \'Takeoff Distance\'')


def RunExistingModel(DataPath, ScriptToRun):
    if ScriptToRun == 'Throttle':
        ModelPath = os.path.join(MainPath, 'Out/Throttle Predictions/Models/')
        Dataset, Dataset_with_strings = trl.PrepareData(DataPath, True)
        model = tf.keras.models.load_model(ModelPath)

        Dataset['p'] = model.predict(Dataset['i'])
        Predicted = pd.DataFrame(Dataset['p'], index = Dataset['i'].index, columns = ['PREDICTED_N1'])
        Actual = pd.DataFrame(Dataset['d'], index = Dataset['d'].index, columns = ['N1'])
        Predicted['ERROR'] = abs((Actual.N1 - Predicted.PREDICTED_N1) / Actual.N1 * 100)
        Predicted['ABSOLUTE_ERROR'] = abs(Actual.N1 - Predicted.PREDICTED_N1)
        print("\nMaximum Error: ", Predicted.ERROR.max(), "Percent\n")

        FinalDataset = pd.concat([Dataset_with_strings, Predicted, Actual], axis=1)
        FinalDataset.to_csv(os.path.join(MainPath, 'Out/Visualizations/Throttle Data Out.csv'))

        CreateVisualizations(FinalDataset, ScriptToRun)

    elif ScriptToRun == 'Takeoff Distance':
        ModelPath = os.path.join(MainPath, 'Out/Takeoff Distance Predictions/Models/')
        Dataset, Dataset_with_strings = tkd.PrepareData(DataPath, True)
        model = tf.keras.models.load_model(ModelPath)

        Dataset['p'] = model.predict(Dataset['i'])
        Predicted = pd.DataFrame(Dataset['p'], index = Dataset['i'].index, columns = ['PREDICTED_DISTANCE_FROM_RUNWAY_END'])
        Actual = pd.DataFrame(Dataset['d'], index = Dataset['d'].index, columns = ['DISTANCE_FROM_RUNWAY_END'])
        Predicted['ERROR'] = abs((Actual.DISTANCE_FROM_RUNWAY_END - Predicted.PREDICTED_DISTANCE_FROM_RUNWAY_END) / Actual.DISTANCE_FROM_RUNWAY_END * 100)
        Predicted['ABSOLUTE_ERROR'] = abs(Actual.N1 - Predicted.PREDICTED_N1)
        print("\nMaximum Error: ", Predicted.ERROR.max(), "Percent\n")

        FinalDataset = pd.concat([Dataset_with_strings, Predicted, Actual], axis=1)
        FinalDataset.to_csv(os.path.join(MainPath, 'Out/Visualizations/Takeoff Distance Data Out.csv'))

        CreateVisualizations(FinalDataset, ScriptToRun)

    else:
        print('Please enter either \'Throttle\' or \'Takeoff Distance\'')
    

def CreateVisualizations(FinalDataset, ScriptToRun):
    if ScriptToRun == 'Throttle':
        x = 'N1'
        y = 'PREDICTED_N1'
        p = ' Throttle '
        d = [75, 105]
    elif ScriptToRun == 'Takeoff Distance':
        x = 'DISTANCE_FROM_RUNWAY_END'
        y = 'PREDICTED_DISTANCE_FROM_RUNWAY_END'
        p = ' Takeoff Distance '
        d = [0, 10000]

    SelectionAC = alt.selection_multi(fields = ['AIRCRAFT_TYPE'], bind = 'legend')
    chartAC = alt.Chart(FinalDataset).mark_circle(size = 15).encode(
                        x = alt.X(x, title='Actual Value', scale = alt.Scale(domain = d)),
                        y = alt.Y(y, title='Predicted Value', scale = alt.Scale(domain = d)),
                        color = alt.Color('AIRCRAFT_TYPE', title = 'Aircraft Type'),
                        tooltip = [alt.Tooltip('AIRCRAFT_TYPE', title = "Aircraft"), alt.Tooltip('AIRPORT', title = "Airport"), alt.Tooltip(x, title = "Throttle Setting", format=",.2f"), alt.Tooltip(y, title = "Predicted Setting", format=",.2f"), alt.Tooltip('ERROR', title = "Percent Error", format=",.2f")],
                        opacity = alt.condition(SelectionAC, alt.value(1), alt.value(0.01))
                ).add_selection(SelectionAC).interactive()

    SelectionAP = alt.selection_multi(fields = ['AIRPORT'], bind = 'legend')
    chartAP = alt.Chart(FinalDataset).mark_circle(size = 15).encode(
                        x = alt.X(x, title='Actual Value', scale = alt.Scale(domain = d)),
                        y = alt.Y(y, title='Predicted Value', scale = alt.Scale(domain = d)),
                        color = alt.Color('AIRPORT', title = 'AIRPORT'),
                        tooltip = [alt.Tooltip('AIRCRAFT_TYPE', title = "Aircraft"), alt.Tooltip('AIRPORT', title = "Airport"), alt.Tooltip(x, title = "Throttle Setting", format=",.2f"), alt.Tooltip(y, title = "Predicted Setting", format=",.2f"), alt.Tooltip('ERROR', title = "Percent Error", format=",.2f")],
                        opacity = alt.condition(SelectionAP, alt.value(1), alt.value(0.01))
                ).add_selection(SelectionAP).interactive()

    UniqueAirports = FinalDataset.AIRPORT.unique()
    UniqueAircraft = FinalDataset.AIRCRAFT_TYPE.unique()
    AirportError =  {'Airport'  : [], 'Absolute Error' : [], 'Average Error' : [], 'Maximum Error' : []}
    AircraftError = {'Aircraft' : [], 'Absolute Error' : [], 'Average Error' : [], 'Maximum Error' : []}

    for airport in UniqueAirports:
        AirportError['Airport'].append(airport)
        AirportError['Absolute Error'].append(FinalDataset.loc[FinalDataset['AIRPORT'] == airport, 'ABSOLUTE_ERROR'].sum())
        AirportError['Average Error'].append(FinalDataset.loc[FinalDataset['AIRPORT'] == airport, 'ABSOLUTE_ERROR'].sum() / FinalDataset['AIRPORT'].value_counts()[airport])
        AirportError['Maximum Error'].append(FinalDataset.loc[FinalDataset['AIRPORT'] == airport, 'ABSOLUTE_ERROR'].max())


    for aircraft in UniqueAircraft:
        AircraftError['Aircraft'].append(aircraft)
        AircraftError['Absolute Error'].append(FinalDataset.loc[FinalDataset['AIRCRAFT_TYPE'] == aircraft, 'ABSOLUTE_ERROR'].sum())
        AircraftError['Average Error'].append(FinalDataset.loc[FinalDataset['AIRCRAFT_TYPE'] == aircraft, 'ABSOLUTE_ERROR'].sum() / FinalDataset['AIRCRAFT_TYPE'].value_counts()[aircraft])
        AircraftError['Maximum Error'].append(FinalDataset.loc[FinalDataset['AIRCRAFT_TYPE'] == aircraft, 'ABSOLUTE_ERROR'].max())

    AirportError = pd.DataFrame.from_dict(AirportError)
    AircraftError = pd.DataFrame.from_dict(AircraftError)

    chartErrorAP1 = alt.Chart(AirportError).mark_bar().encode(
                        x = alt.X('Airport', title='Airport'),
                        y = alt.Y('Absolute Error', title='Sum of Absolute Error'),
                        color = alt.Color('Airport', title = 'Airport'),
                        tooltip = [alt.Tooltip('Airport', title = "Airport"), alt.Tooltip('Absolute Error', title = "Absolute Error", format=",.2f"), alt.Tooltip('Average Error', title = "Average Error", format=",.2f"), alt.Tooltip('Maximum Error', title = "Maximum Error", format=",.2f")])
    chartErrorAP2 = alt.Chart(AirportError).mark_bar().encode(
                        x = alt.X('Airport', title='Airport'),
                        y = alt.Y('Average Error', title='Average Absolute Error'),
                        color = alt.Color('Airport', title = 'Airport'),
                        tooltip = [alt.Tooltip('Airport', title = "Airport"), alt.Tooltip('Absolute Error', title = "Absolute Error", format=",.2f"), alt.Tooltip('Average Error', title = "Average Error", format=",.2f"), alt.Tooltip('Maximum Error', title = "Maximum Error", format=",.2f")])
    chartErrorAP3 = alt.Chart(AirportError).mark_bar().encode(
                        x = alt.X('Airport', title='Airport'),
                        y = alt.Y('Maximum Error', title='Maximum Absolute Error'),
                        color = alt.Color('Airport', title = 'Airport'),
                        tooltip = [alt.Tooltip('Airport', title = "Airport"), alt.Tooltip('Absolute Error', title = "Absolute Error", format=",.2f"), alt.Tooltip('Average Error', title = "Average Error", format=",.2f"), alt.Tooltip('Maximum Error', title = "Maximum Error", format=",.2f")])

    chartErrorAP = alt.hconcat(chartErrorAP1, chartErrorAP2, chartErrorAP3)

    chartErrorAC1 = alt.Chart(AircraftError).mark_bar().encode(
                        x = alt.X('Aircraft', title='Aircraft'),
                        y = alt.Y('Absolute Error', title='Sum of Absolute Error'),
                        color = alt.Color('Aircraft', title = 'Aircraft'),
                        tooltip = [alt.Tooltip('Aircraft', title = "Aircraft"), alt.Tooltip('Absolute Error', title = "Absolute Error", format=",.2f"), alt.Tooltip('Average Error', title = "Average Error", format=",.2f"), alt.Tooltip('Maximum Error', title = "Maximum Error", format=",.2f")])
    chartErrorAC2 = alt.Chart(AircraftError).mark_bar().encode(
                        x = alt.X('Aircraft', title='Aircraft'),
                        y = alt.Y('Average Error', title='Average Absolute Error'),
                        color = alt.Color('Aircraft', title = 'Aircraft'),
                        tooltip = [alt.Tooltip('Aircraft', title = "Aircraft"), alt.Tooltip('Absolute Error', title = "Absolute Error", format=",.2f"), alt.Tooltip('Average Error', title = "Average Error", format=",.2f"), alt.Tooltip('Maximum Error', title = "Maximum Error", format=",.2f")])
    chartErrorAC3 = alt.Chart(AircraftError).mark_bar().encode(
                        x = alt.X('Aircraft', title='Aircraft'),
                        y = alt.Y('Maximum Error', title='Maximum Absolute Error'),
                        color = alt.Color('Aircraft', title = 'Aircraft'),
                        tooltip = [alt.Tooltip('Aircraft', title = "Aircraft"), alt.Tooltip('Absolute Error', title = "Absolute Error", format=",.2f"), alt.Tooltip('Average Error', title = "Average Error", format=",.2f"), alt.Tooltip('Maximum Error', title = "Maximum Error", format=",.2f")])

    chartErrorAC = alt.hconcat(chartErrorAC1, chartErrorAC2, chartErrorAC3)

    chartAP.save(os.path.join(MainPath, 'Out/Visualizations/' + p + 'Airport Chart.html'))
    chartAC.save(os.path.join(MainPath, 'Out/Visualizations/' + p + 'Aircraft Chart.html'))
    chartErrorAP.save(os.path.join(MainPath, 'Out/Visualizations/' + p + 'Airport Error.html'))
    chartErrorAC.save(os.path.join(MainPath, 'Out/Visualizations/' + p + 'Aircraft Error.html'))
    chartAP.show()
    chartAC.show()
    chartErrorAP.show()
    chartErrorAC.show()

###########################################################################################################################################
#                                                 RUN THE PROGRAM
###########################################################################################################################################
DataPath = 'C:/Users/Zayn.Roohi/Documents/OASIS/takeoff_distance_A320_A330_A340.csv'

#note that running an existing model DOES NOT WORK PROPERLY due to a glitch with TensorFlow
#as of right now you must always go through the entire process of creating an entirely new script

#RunExistingModel(DataPath, 'Throttle')
CreateNewModel(DataPath, 'Takeoff Distance')