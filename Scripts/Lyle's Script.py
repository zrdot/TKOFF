import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pathlib
import os.path 
import numpy as np
import webbrowser as wb

#TODO: propagate detailed aircraft type
MainPath = pathlib.Path(__file__).parent.parent.resolve()
OutPath = os.path.join(MainPath, "Out")
InPath = 'C:/Users/Zayn.Roohi/Documents/OASIS'
filename = 'takeoff_distance_A320_A330_A340'
file_extension = '.csv'

df = pd.read_csv(os.path.join(InPath, filename + file_extension), index_col=["FLIGHT_ID", "TIME_OFFSET"])
df.rename(
    columns={'HEAD_WIND':'HEAD_WIND_KNOTS'
             ,'FUEL_QUANTITY':'FUEL_QUANTITY_KILOGRAMS'
             ,'P64  Air Temperature (outside) at Start of Event (library) (Deg Celsius)':'AIR_TEMP_OUTSIDE_CELSIUS'
            }
    , inplace=True)

def scatter_plot(df_schedule, color_field = "APT_AIRCRAFT_RUNWAY_STAGE" ):
    #TODO:make this a function
    x_max = stated_max = df_schedule.DISTANCE_FROM_RUNWAY_END_liftoff.max()
    x_min = stated_min = df_schedule.DISTANCE_FROM_RUNWAY_END_liftoff.min()
    y_max = detected_max = df_schedule.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF.max()
    y_min = detected_min = df_schedule.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF.min()

    perfect_fit_min = x_min if x_min < y_min else y_min
    perfect_fit_max = x_max if x_max > y_max else y_max

    fig = px.scatter(
        df_schedule.reset_index(level = ["FLIGHT_ID"])
        , title='Liftoff distance from runway end (feet)'
        , x='DISTANCE_FROM_RUNWAY_END_liftoff'
        ,y='DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF'
        , color=color_field
        , labels = {'DISTANCE_FROM_RUNWAY_END_liftoff':'As stated in CFDR',
                    'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF':'As detected from trajectory',
                    'APT_AIRCRAFT_RUNWAY_STAGE': 'Airport / Aicraft Type / Runway / Stage Length ID'
                   }
        , hover_data= ['FLIGHT_ID']#, 'DISTANCE_FROM_RUNWAY_END', 'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF']
        , opacity = 0.6
    )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min, perfect_fit_max]
            , mode = "lines"
            , name = 'Perfect Fit'
            , line = go.scatter.Line(color = 'black', dash = 'dash')
            , opacity=0.4
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min-275, perfect_fit_max-275]
            , mode = "lines"
            , name = 'Perfect Fit - 275 ft (a.k.a detected liftoff one second before actual)'
            , line = go.scatter.Line(color = 'black', dash = 'longdash')
            , opacity=0.4
        )

    )


    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min-550, perfect_fit_max-550]
            , mode = "lines"
            , name = 'Perfect Fit - 550 ft (a.k.a detected liftoff two seconds before actual)'
            , line = go.scatter.Line(color = 'purple', dash = 'longdashdot')
            , opacity=0.4
        )

    )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min+275, perfect_fit_max+275]
            , mode = "lines"
            , name = 'Perfect Fit + 275 ft (a.k.a detected liftoff one second after actual)'
            , line = go.scatter.Line(color = 'black', dash = 'longdash')
            , opacity=0.4
        )

    )

    fig.add_trace(
        go.Scatter(
            x = [perfect_fit_min, perfect_fit_max]
            , y = [perfect_fit_min+550, perfect_fit_max+550]
            , mode = "lines"
            , name = 'Perfect Fit + 550 ft (a.k.a detected liftoff two seconds after actual)'
            , line = go.scatter.Line(color = 'purple', dash = 'longdashdot')
            , opacity=0.4
        )

    )

    fig.update_yaxes(
    #     scaleanchor = "x",
        scaleratio = 1,
      )
    
    return fig

def plot_metrics_for_individual_flights(dataframe, flight_sample_size = 10):

    metric_maxes = {}

    plotbook_filename = 'plotbook' + '_' + str(flight_sample_size) if flight_sample_size != -1 else 'plotbook'
    df = dataframe
    if 'FLIGHT_ID' in df.columns:
        df_grouped_by_flight = df.drop(columns = ['FLIGHT_ID'], axis = 1).groupby("FLIGHT_ID")
        df["FLIGHT_ID"] = df["FLIGHT_ID"].astype(str)
    else:
        df_grouped_by_flight = df.groupby("FLIGHT_ID")
    
    fn = plotbook_filename + '.html'
    open( os.path.join(OutPath,fn), 'w')
    #     #TODO: write plots to pdf file
    #     fn = plotbook_filename + '.pdf'

    flight_count = 0
    for flight, group in df_grouped_by_flight:
        flight_sample_size = len(group) if flight_sample_size == -1 else flight_sample_size

        if flight_count == flight_sample_size:
            break

#         metrics = ['MSL_ALT']
        metrics = ['AFE_ALT']
        metrics += ['N1', 'TAS_SEGMENT']

        #create Figure with subplots for each metric for a given flight
        fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

        for m in metrics:
            metric_maxes[m]=group[m].max() if m != 'N1' else 100

            #create and add a subplot to the Figure
            sub = go.Scatter(
                    x=group.DISTANCE_FROM_RUNWAY_END
                    , y=group[m]
                    , mode="markers"
                    , name = m
            )
            fig.add_trace(sub, row=metrics.index(m)+1, col=1)

    #         #turn off auto range adjustment
    #         fig.update_layout(
    #             yaxis_autorange = False
    #             )        
    #         fig.update_yaxes({}, row=metrics.index(m)+1, col=1)

            dl_cgtd = detected_liftoff_cumul_grnd_trk_dist = df_liftoff.loc[flight,"DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF"]
            fig.add_trace(
                go.Scatter(
                    x=[dl_cgtd, dl_cgtd]
                    , y=[0,metric_maxes[m]]
                    , mode = "lines"
                    , name = "Detected Liftoff"
                    , line = go.scatter.Line(color = 'gray')
                )
                , row=metrics.index(m)+1, col=1 
            )        

            rprt_liftoff_cgtd = reported_liftoff_cumul_grnd_trk_dist = df_liftoff.loc[flight,"DISTANCE_FROM_RUNWAY_END"]
            fig.add_trace(
                go.Scatter(
                    x=[rprt_liftoff_cgtd, rprt_liftoff_cgtd]
                    , y=[0,metric_maxes[m]]
                    , mode = "lines"
                    , name = "Reported Liftoff"
                    , line = go.scatter.Line(color = 'orange', dash = 'dash')
                )
                , row=metrics.index(m)+1, col=1 
            )

            fig.layout.yaxis.update(title_text = metrics[0])
            fig.layout.yaxis2.update(title_text = metrics[1], tickvals=[0] + np.arange(70,100,5))
            fig.layout.yaxis3.update(title_text = metrics[2])
            fig.layout.update(title_text = "Flight " + str(flight) + '<br>' + 'Airport / Aicraft Type / Runway / Stage Length ID: ' + 
                              str(df_liftoff.loc[flight,"APT_AIRCRAFT_RUNWAY_STAGE.1"]))
        
        #write plots to html file         
        html = fig.to_html()
        with open( os.path.join(OutPath,fn), 'a') as f:
            f.write(html) 

        flight_count = flight_count + 1
    print(os.path.join(OutPath,fn))
        
    return (os.path.join(OutPath,fn))

modes = ['Undetected','Takeoff Ground Roll']

df["Detected_Trajectory_Mode"] = modes.index('Undetected')
df.Detected_Trajectory_Mode = df.Detected_Trajectory_Mode.astype("int")

df_liftoff = pd.DataFrame(df[df.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF.notna()])
df_liftoff.reset_index(level = 1, inplace = True)

df_rollstart = df[df.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF.notna()]
df_rollstart.reset_index(level = 1, inplace = True)

#formulate flight schedule
salient_rollstart_col_list = ['DISTANCE_FROM_THRESHOLD_AT_POINT_ONE', 'DISTANCE_FROM_POINT_ONE_FEET',
       'GROUNDSPEED_AT_POINT_ONE_KNOTS', 'WELL-BEHAVED_TRAJECTORY',
#        'LATITUDE_AT_POINT_ONE', 'LONGITUDE_AT_POINT_ONE',
        'STATED_SEGMENT_START_OF_TAKEOFF',
       'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF'
        ]
extraneous_schedule_columns_list = [col for col in df_rollstart.columns if col not in salient_rollstart_col_list + ["DISTANCE_FROM_RUNWAY_END", 'TIME_OFFSET','FUEL_QUANTITY_KILOGRAMS', 'HEAD_WIND_KNOTS', 'APT_AIRCRAFT_RUNWAY_STAGE.1']]

df_schedule = df_liftoff.drop(salient_rollstart_col_list, axis = 1).join(
    df_rollstart.drop(
        columns = extraneous_schedule_columns_list, axis = 1
    )
    , how = 'left'
    , lsuffix = '_liftoff' 
    , rsuffix = '_rollstart'
)

#coalesce the grouping column
mask = (df_schedule['APT_AIRCRAFT_RUNWAY_STAGE.1_liftoff'].notna())
df_schedule.loc[mask, "APT_AIRCRAFT_RUNWAY_STAGE"] = df_schedule['APT_AIRCRAFT_RUNWAY_STAGE.1_liftoff']
mask = (df_schedule['APT_AIRCRAFT_RUNWAY_STAGE.1_rollstart'].notna())
df_schedule.loc[mask, "APT_AIRCRAFT_RUNWAY_STAGE"] = df_schedule['APT_AIRCRAFT_RUNWAY_STAGE.1_rollstart']

df_schedule["TAKEOFF_GROUND_ROLL_DISTANCE_DETECTED_IN_FEET"] = df_schedule.DISTANCE_FROM_RUNWAY_END_liftoff - df_schedule.DISTANCE_FROM_RUNWAY_END_rollstart
df_schedule["liftoff_detection_quality_numerical"] = abs(df_liftoff.DISTANCE_FROM_RUNWAY_END_AT_DETECTED_LIFTOFF - df_liftoff.DISTANCE_FROM_RUNWAY_END) / df_liftoff.DISTANCE_FROM_RUNWAY_END 

#instantiate bins for grouping on the quality of liftoff detection
liftoff_detection_quality_bins = {0:"very high", 0.01:"high", 0.05:"good", 0.1:"mediocre", 1:"poor"}

labels = []
i=0
for k, v in sorted(liftoff_detection_quality_bins.items()):
    labels.append(v)
    i += 1
labels_as_tuple = tuple(labels)

df_schedule["liftoff_detection_quality_categorized"] = pd.cut(
        df_schedule["liftoff_detection_quality_numerical"]
        , list(liftoff_detection_quality_bins.keys())+[100000]
        , labels = labels_as_tuple
    )

flights_with_very_high_liftoff_detection_quality = df_schedule[df_schedule.liftoff_detection_quality_categorized == "very high"].index.to_series()

#add average N1 during detected ground roll to the schedule
df_traj_takeoff_ground_roll = df[df.Detected_Trajectory_Mode == modes.index('Takeoff Ground Roll')]
df_traj_tkoff_gd_rl_groupby = df_traj_takeoff_ground_roll.groupby(by=["FLIGHT_ID"])
df_schedule["N1_during_detected_takeoff_ground_roll"] = df_traj_tkoff_gd_rl_groupby.mean().N1

#def mark_ground_roll_segments(df, flight_id, liftoff_time_offset, rollstart_time_offset, ground_roll_mode_index):
    #return a dataframe of start and stop times by flight
#select rows where TIME_OFFSET is within TIME_OFFSET_rollstart and TIME_OFFSET_liftoff
df_aug = df.loc[:,['N1']]
df_aug = df_aug.join(df_schedule, rsuffix="_sched", how="left").reset_index(level="TIME_OFFSET")
df_aug["Detected_Ground_Roll"] = (df_aug.TIME_OFFSET >= df_aug.TIME_OFFSET_rollstart) & (df_aug.TIME_OFFSET <= df_aug.TIME_OFFSET_liftoff)
df_aug.set_index('TIME_OFFSET',append=True, inplace=True)
mask = (df_aug.Detected_Ground_Roll == True)
df.loc[mask, 'Detected_Trajectory_Mode'] = modes.index('Takeoff Ground Roll')

inp_vars = ["HEAD_WIND_KNOTS_liftoff", "FUEL_QUANTITY_KILOGRAMS_rollstart", "AIR_TEMP_OUTSIDE_CELSIUS"]
head_wind_bin_width = 2.5
#create the categorizations of input vars including numerical bins on the input variables in inp_vars list
df_schedule_groupby = df_schedule.groupby(by="APT_AIRCRAFT_RUNWAY_STAGE") 
df_schedule_bins = pd.DataFrame(
    index=pd.Index(df_schedule_groupby.indices.keys(), name = "APT_AIRCRAFT_RUNWAY_STAGE")
    , columns=inp_vars) #for holding the bin definitions

for i in inp_vars:
    frame = pd.DataFrame()
    num_bins = bin_width = 0
    for group, inp_var in df_schedule_groupby:

        if i == "FUEL_QUANTITY_KILOGRAMS_rollstart":
            bin_width = 0.015*inp_var[i].max()
        elif i == "HEAD_WIND_KNOTS_liftoff":
            bin_width = head_wind_bin_width 
        elif i == "AIR_TEMP_OUTSIDE_CELSIUS":
            bin_sequence = [0,(80-32)*5/9,1000]       
        else:
            num_bins = 10
        
        if bin_width > 0:
            num_bins = 1 + int((inp_var[i].max() - inp_var[i].min()) / bin_width)
        
        bins = num_bins if num_bins > 0 else bin_sequence

        print(group, i, "bins ", bins, sep='-->') #testing
        #pass the group-inp_var combo as a series extracted from the groupby object
        var_group_combo_as_series = pd.Series(inp_var[i], name=i)
        (var_group_combo_binned, bins) = pd.cut(var_group_combo_as_series
                           , bins
                           , labels=False
                           , retbins=True
                          )

        var_group_combo_binned = var_group_combo_binned.to_frame(name=i)
        #store the rows or columns
        frame = pd.concat([frame, var_group_combo_binned], axis = 0)

        #store the category definitions
        df_schedule_bins.loc[group, i] = bins
    
    df_schedule[i+"_BIN"] = frame


#some preparation for group analysis
df_schedule_grouped = pd.DataFrame()
grouping_criteria = ["APT_AIRCRAFT_RUNWAY_STAGE", "HEAD_WIND_KNOTS_liftoff_BIN", "FUEL_QUANTITY_KILOGRAMS_rollstart_BIN", "AIR_TEMP_OUTSIDE_CELSIUS_BIN"] 
df_schedule_grouped = df_schedule.reset_index().set_index(grouping_criteria+["FLIGHT_ID"]).groupby(by=grouping_criteria)
min_flight_count = 7

#plot relationship of actual N1 to actual ground roll distance for a given weight group, weather group, runway group, aircraft type
#loop through the groups
df_schedule_grouped_filtered = df_schedule_grouped.filter(lambda x: x['N1'].count() >= min_flight_count).reset_index(["FLIGHT_ID"])

x = 'DISTANCE_FROM_RUNWAY_END_liftoff'
y = 'N1'
    
fig = px.scatter(
    df_schedule_grouped_filtered
    , title='CFDR actual N1 vs actual ground roll distance for<br>a given temperature/weight/wind/runway/aircraft group<br>'+ 'min flight count per group: ' + str(min_flight_count)
        + '<br>head winds within: ' + str(head_wind_bin_width) + 'knots'
        + '<br>Temperature Category 0 means below 80 Degrees Farenheit'
    , x=x
    , y=y
    , color=df_schedule_grouped_filtered.index.to_series()
    , trendline = 'ols'
    , labels = {'DISTANCE_FROM_RUNWAY_END_liftoff':'Distance in feet from deparing runway end at liftoff as stated in CFDR',
                'N1':'Throttle setting at liftoff',
#                 'APT_AIRCRAFT_RUNWAY_STAGE': 'Airport / Aicraft Type / Runway / Stage Length ID'
               }
#     , hover_data= ['FLIGHT_ID']
)
fig.layout.yaxis.update(range=(80,100))
fig.layout.xaxis.update(range=(5000,11000))
fig.layout.legend.update(title = "Airport / Aicraft Type / Runway / Stage Length ID<br>, Headwind Category, Weight Category, Air Temp Category")
file = fig.to_html()
with open(OutPath+'\\N1_distance_correlation.html', 'w') as f:
    f.write(file)
    
df_schedule_bins["FUEL_QUANTITY_KILOGRAMS_rollstart_BIN_WIDTH"] = df_schedule_bins.FUEL_QUANTITY_KILOGRAMS_rollstart.apply(lambda x: x[1] - x[0])

resulting_groups = df_schedule_grouped_filtered.reset_index(level="APT_AIRCRAFT_RUNWAY_STAGE").APT_AIRCRAFT_RUNWAY_STAGE.unique()#.to_list()

pd.DataFrame(df_schedule_bins.FUEL_QUANTITY_KILOGRAMS_rollstart_BIN_WIDTH.loc[list(resulting_groups)])

#TODO: get the actual bin into the legend
df_schedule_grouped_filtered.index.to_frame().drop(
    columns=["APT_AIRCRAFT_RUNWAY_STAGE", "HEAD_WIND_KNOTS_liftoff_BIN", "FUEL_QUANTITY_KILOGRAMS_rollstart_BIN", "AIR_TEMP_OUTSIDE_CELSIUS_BIN"]
    , axis=1).join(df_schedule_bins).HEAD_WIND_KNOTS_liftoff.to_frame().reset_index()


detection_overview_plot = scatter_plot(df_schedule[df_schedule.liftoff_detection_quality_categorized.notna() == True], "liftoff_detection_quality_categorized")
detection_overview_plot.show()
plot_html = detection_overview_plot.to_html()
with open(OutPath+'\\plot_detection_overview.html', 'w') as f:
    f.write(plot_html)

#use group to highlight the groups that merit further investigation
detection_plot_by_group = scatter_plot(df_schedule)
detection_plot_by_group.show()
plot_html = detection_plot_by_group.to_html()
with open(OutPath+'\\plot_detection_by_group.html', 'w') as f:
    f.write(plot_html)

df_high_qual_lift = df.join(flights_with_very_high_liftoff_detection_quality,how = 'inner')
fp = plot_metrics_for_individual_flights(df_high_qual_lift, -1)
wb.open(url=fp)

#focus in on one pair of flights
#df_schedule = df_schedule.loc[[1124373, 1128101],:]

#two flights with similar stated liftoff point 
flights = [1128101, 1124373]
plot_metrics_for_individual_flights(df.loc[flights,])


fig = px.scatter(
    df_rollstart
    , title='Start of takeoff roll, distance from runway end (feet)'
    , x='DISTANCE_FROM_RUNWAY_END'
    , y='DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF'
    , color="APT_AIRCRAFT_RUNWAY_STAGE.1" 
    , labels = {'DISTANCE_FROM_RUNWAY_END':'As stated in CFDR',
                'DISTANCE_FROM_RUNWAY_END_AT_DETECTED_START_OF_TAKEOFF':'As detected from trajectory',
                'APT_AIRCRAFT_RUNWAY_STAGE.1': 'Airport / Aicraft Type / Runway / Stage Length ID'
               }
)

fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )

fig.show()

file = fig.to_html()
with open(OutPath+'\plot_rollstart.html', 'w') as f:
    f.write(file)

fig_px = px.scatter(
    df_liftoff
    , title='Liftoff distance from runway end (feet)'
    , x='DISTANCE_FROM_RUNWAY_END'
    ,y='MSL_ALT'
    , color = "FLIGHT_ID"
    , labels = {'DISTANCE_FROM_RUNWAY_END':'Distance from runway end (feet)',
                'MSL_ALT':'Altitude above mean sea level (feet)',
                'APT_AIRCRAFT_RUNWAY_STAGE.1': 'Airport / Aicraft Type / Runway / Stage Length ID'
               }
)

file = fig_px.to_html()
with open(OutPath+'\plot_high_quality_by_flight.html', 'w') as f:
    f.write(file)
    
fig_px.show()