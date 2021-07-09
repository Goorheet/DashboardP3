# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:01:25 2021

@author: Rein
"""

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

external_stylesheets = ['http://psychiatrieschoorl.nl/css/stylesheet.css']

import Model

#Make DataFrames
##Data Analysis
###Forecast validation
val_h = Model.df_val[['H_m0[m]','GFS_H_m0_c[m]',]]
val_p = Model.df_val[['T_E[s]','GFS_T_E[s]',]]
val_f = Model.df_val[['Flux[kW/m]','GFS_Flux[kW/m]',]]

###Forecast model comparison
com_h = Model.df_forecast[['EWAM_H_m0[m]','GFS_H_m0[m]',]]
com_p = Model.df_forecast[['EWAM_T_E[s]','GFS_T_E[s]',]]
com_d = Model.df_forecast[['EWAM_DIR_c','GFS_DIR_c']]
com_f = Model.df_forecast[['EWAM_Flux[kW/m]','GFS_Flux[kW/m]',]]

###WEC Power production
wec = Model.df_wec

##Regression
###Summary
regressors = ['EWAM + GFS','EWAM','GFS']
mean_absolute = [Model.MAE_XGB4,Model.MAE_XGB2,Model.MAE_XGB3]
mean_squared = [Model.MSE_XGB4,Model.MSE_XGB2,Model.MSE_XGB3]
root_mean_squared = [Model.RMSE_XGB4,Model.RMSE_XGB2,Model.RMSE_XGB3]
cv_root_mean_squared = [Model.cvRMSE_XGB4,Model.cvRMSE_XGB2,Model.cvRMSE_XGB3]

df_errors = pd.DataFrame(regressors,columns=['Model'])
df_errors['Mean absolute error'] = mean_absolute
df_errors['Mean squared error'] = mean_squared
df_errors['Root mean squared error'] = root_mean_squared
df_errors['CV Root mean squared error'] = cv_root_mean_squared 

###EWAM + GFS
df_hyb = pd.DataFrame(Model.y4_pred_XGB[1:200], columns=['Model'])
df_hyb['Real'] = Model.y4_test[1:200]

###EWAM
df_EWAM = pd.DataFrame(Model.y2_pred_XGB[1:200], columns=['Model'])
df_EWAM['Real'] = Model.y2_test[1:200]

###GFS
df_GFS = pd.DataFrame(Model.y3_pred_XGB[1:200], columns=['Model'])
df_GFS['Real'] = Model.y3_test[1:200]

df_feat = pd.DataFrame(Model.model.feature_importances_) 
df_feat.columns = ['Correlation score']
df_feat['Name of variable'] = Model.df.columns[1:12]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H2(['Project 3 - Forecasting power production of Mutriku Wave Energy Converter'],
        )
        ], style={'color': '#3D9970'})
    ,
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data Analysis', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4')  ,
    ]),
    html.Div(id='tabs-content'),
    html.Div(['by Rein Arnold (98023)'], style={'float': 'right','display': 'inline-block','position': 'absolute','top': '10px','right': '10px','font-style': 'italic'})
])

@app.callback(
    Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.P('In this section, the data used for the forecasting model will be analysed. This analysis can be split in the analysis of the output data (power production by WEC) and the input data (wave forecasts).',style={'padding-top': '15px'},),                        
            html.H4('WEC Power Production',style={'padding-top': '20px'},),
            html.P('The power production data for the Mutriku WEC has been supplied by Mutriku and has a time span of two months. The data supplied is not for the whole power plant, but for one of the sixteen turbines: turbine 8. The data is on a second time scale and includes information about the status of the turbine, classified with a number. The power production data is resampled to a hourly time scale, as this is the time scale of the wave forecasts. Any periods of abnormal operating modes, such as maintenance or outage, are then filtered out to avoid inaccuracies in the forecasting model.'),
            html.Div([
            dcc.Graph(
              figure= px.line(wec,                 
              title="Power production Mutriku turbine 8",
              ),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(wec,
                              title=" "),
              className='six columns'
              ),
            ]),
            html.P('The boxplot shows that for the power production of the WEC, no outliers are identified. As all values are sensible, there are no outliers removed.'),
            html.H4('Wave Forecast Validation',style={'padding-top': '20px'},),
            html.P('A validation of the wave forecast data is carried out. This is done to ensure the wave forecasts are accurate for the location of the Mutriku Power Plant. This is done by comparing the historical wave forecasts to the sea states as measured by a sensor placed in front of the Mutriku Power Plant. This sensor gathered sea state data in the spring of 2018. A validation is made for the significant wave height, the wave energy period and the wave energy flux.'),
            html.Div([
            dcc.Dropdown( 
        id='dropdown2',
        options=[
            {'label': 'Significant wave height', 'value': 'hm0'},
            {'label': 'Energy period', 'value': 'ep'},
            {'label': 'Energy flux ', 'value': 'fl'},
        ], 
        value='hm0'
        ),
            html.Div(id='validation_graphs')
        ]),
            
            html.H4('Wave Forecast Model Comparison',style={'padding-top': '15px'},),
            html.P('In this section the EWAM wave forecast data is compared with the GFS wave forecast data. This comparison is important for the determination of accuracy of both models, as very different results between the model show uncertainty. Comparisons are carried out for the parameters considered in the forecasting model: significant wave height, wave energy period, wave direction and energy flux. ',style={'padding-top': '15px'},),
             html.Div([
            dcc.Dropdown( 
        id='dropdown3',
        options=[
            {'label': 'Significant wave height', 'value': 'hm0'},
            {'label': 'Energy period', 'value': 'ep'},
            {'label': 'Wave direction', 'value': 'dir'},
            {'label': 'Energy flux ', 'value': 'fl'},
        ], 
        value='hm0'
        ),
            html.Div(id='comparison_graphs')
        ]),
            
                    ],className="twelve columns")        
    
    elif tab == 'tab-2':
        return html.Div([
     html.Div([
            html.Div([
                html.Div(['On the basis of the elbow curve displayed below the number of clusters is decided at 4, as the relative gain in score is low for n>4.'],
                         style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '15px', 'padding-bottom': '10px'}
                     ),
                html.Img(src='assets/elbow.png',className="twelve columns"),
                html.Div(['The three figures on the right side of this page show the four clusters for the energy flux, the wave direction and a 3d plot of the clusters for the significant wave height and the wave energy period. The four clusters identified can be characterised by the power production of the WEC: low power, medium power, high power and very high power.'],
                         style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '15px', 'padding-bottom': '10px'}
                    ),
                   html.P('The correlation between wave energy flux and power production is clearly visible. It is also visible that after a certain wave energy flux is reached, the power production is not increasing anymore. At this moment the turbine has reached its maximum capacity.',style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '5px', 'padding-bottom': '5 px'}),
         html.P('The correlation between wave direction and power production is not very pronounced, and that the clusters are uniformly spread over the wave directions.',style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '5px', 'padding-bottom': '5px'}),
         html.P('In the bottom figure, the clusters are shown for the core input parameters: the significant wave height and the wave energy period. The low power cluster is clearly located at smaller significant wave heights and smaller wave energy periods. For every higher power cluster, the significant wave heights and wave energy periods are increasing.',style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '5px', 'padding-bottom': '5px'}),
        ]       ,className="four columns"),
     html.Div([
         html.Img(src='assets/cluster1.png',className="six columns"),
         html.Img(src='assets/cluster2.png',className="six columns"),
     
         html.Img(src='assets/cluster3d.png',className="eight columns")
         ],className="eight columns")
     
     ])    
     ])
            
    elif tab == 'tab-3':
            return html.Div([
            html.Div(['The features for the forecasting model are selected according to the XGBoost feature importances scale, as shown below. The wave direction angles for both EWAM and GFS where the only variables that decreased the accuracy of the regression model. For that reason the wave direction angles in degrees are not regarded and only the cosine of the wave direction angles is considered for the regression model. '],style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '15px'}),
                dcc.Graph(id='features',    
                      figure= px.histogram(df_feat, x="Name of variable", y="Correlation score", log_y=True, title="Random Forest feature importances (log scale)")),
                html.P('Regarding the feature importance scores, the following observations can be made:',style={'display': 'inline-block', 'padding-left': '15px', 'padding-top': '15px'},),
                html.Li('Overall, the EWAM wave forecast model outperforms the GFS wave forecast model. A reason for this might be the higher resolution of the EWAM model (5KM) opposed to the GFS model (16KM). Even though this is the case, the hybrid of the models is still used for the regression, as the accuracy of the hybrid input is higher compared to using only EWAM. A remark has to be made for the wave energy period: for this parameter the GFS model outperforms the EWAM model.',style={'padding-left': '40px','padding-right': '40px','padding-top': '15px'}),
                html.Li('The wave energy flux is the most important parameter for the forecasting model. This is to be expected: the power produced by the WEC is directly dependent on the energy transmitted by the wave. The importance of feature engineering is clearly demonstrated.',style={'padding-left': '40px','padding-right': '40px'}),
                html.Li('Another highlight of the importance of feature engineering is the much higher performance of the cosine of the wave direction angles over the wave direction angles in degrees.',style={'padding-left': '40px','padding-right': '40px'}),
                html.Li('The importance of the clustering process is clearly visible. The cluster parameter is the second most important parameter of the regression model, afther the EWAM energy flux.',style={'padding-left': '40px','padding-right': '40px'}),
        ])
            
    
    elif tab == 'tab-4':
        return html.Div([
            html.P('A regression of the power production by the Wave Energy Converter is made in two ways: by considering a hybrid of both EWAM and GFS forecasting models and by considering the forecasting models individually. The hybrid model has the smallest errors that are within the specifications set by ASHRAE for energy forecasting models (cvRMSE < 15%).',style={'padding-top': '15px'},),
            dcc.Dropdown( 
        id='dropdown',
        options=[
            {'label': 'Summary', 'value': 'sum'},
            {'label': 'EWAM + GFS', 'value': 'hyb'},
            {'label': 'EWAM', 'value': 'EWAM'},
            {'label': 'GFS', 'value': 'GFS'},
        ], 
        value='sum'
        ),
            html.Div(id='regression_model')
        ])
    
    
@app.callback(Output('regression_model', 'children'), 
              Input('dropdown', 'value'))    

def render_model(model_name):
    
    if model_name == 'sum':
        return html.Div([
            html.H6('Overview of errors obtained for each forecasting method'),
            html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df_errors.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df_errors.iloc[i][col]) for col in df_errors.columns
            ]) for i in range(len(df_errors))
        ])
    ])       
            ])
            

    elif model_name == 'hyb':
        return html.Div([
            dcc.Graph(id='graph1', 
              figure= px.scatter(x=Model.y4_pred_XGB, y=Model.y4_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),
            dcc.Graph(id='graph2', 
              figure= px.line(df_hyb,
                              title="First 200 entries of tested data")    
              ),    
            ])

    elif model_name == 'EWAM':
        return html.Div([
            dcc.Graph(id='graph1', 
              figure= px.scatter(x=Model.y2_pred_XGB, y=Model.y2_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ),
            dcc.Graph(id='graph2', 
              figure= px.line(df_EWAM,
                              title="First 200 entries of tested data")    
              ),    
            ])

    elif model_name == 'GFS':
        return html.Div([
            dcc.Graph(id='graph3', 
              figure= px.scatter(x=Model.y3_pred_XGB, y=Model.y3_test,                 
              labels=dict(x="Model", y="Real"),
              title="Model validation scatterplot")
              ), 
            dcc.Graph(id='graph4', 
              figure= px.line(df_GFS,
                              title="First 200 entries of tested data")
              ),
            ])

@app.callback(Output('validation_graphs', 'children'), 
              Input('dropdown2', 'value'))  

def render_validation_graphs(item):
    if item == 'hm0':
        return html.Div([
            dcc.Graph(
              figure= px.line(val_h,                 
              title="Validation of significant wave height"),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(val_h,
                              title=" "),
              className='six columns'
              ),
            html.P('The graphs show that the GFS forecast for wave height is very accurate.')
            ])
    elif item == 'ep':
        return html.Div([
            dcc.Graph(
              figure= px.line(val_p,                 
              title="Validation of wave energy period"),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(val_p,
                              title=" "),
              className='six columns'
              ),
            html.P('Above plots show that the measured data and the forecasted data for wave energy period have very similar mean and standard deviation. The main difference are the outliers: the measured data contains more outliers with high periods and the forecasted data contains more outliers with low periods.')
            ])
    elif item == 'fl':
        return html.Div([
            dcc.Graph(
              figure= px.line(val_f,                 
              title="Validation of wave energy flux"),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(val_f,
                              title=" "),
              className='six columns'
              ),
            html.P('From the above data analysis it is visible that the Flux is very similar for the forecasts and the measurements. This is very important conclusion, as the power production of the WEC is directly dependent on the wave energy flux.')
            ])
    

@app.callback(Output('comparison_graphs', 'children'), 
              Input('dropdown3', 'value'))  

def render_comparison_graphs(item):
    if item == 'hm0':
        return html.Div([
            dcc.Graph(
              figure= px.line(com_h,                 
              title="Comparison of significant wave height"),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(com_h,
                              title=" "),
              className='six columns'
              ),
            html.P('Wave heights are very similar for both models. On average, EWAM estimates slightly higher than GFS, while GFS has higher outliers.')
            ])
    elif item == 'ep':
        return html.Div([
            dcc.Graph(
              figure= px.line(com_p,                 
              title="Comparison of wave energy period"),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(com_p,
                              title=" "),
              className='six columns'
              ),
            html.P('As opposed to the wave heights, EWAM estimates wave periodes slightly lower compared to GFS.')
            ])
    elif item == 'dir':
        return html.Div([
            dcc.Graph(
              figure= px.line(com_d,                 
              title="Comparison of wave direction"),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(com_d,
                              title=" "),
              className='six columns'
              ),
            html.P('Wave directions are similar for both models and coming from north orientations, as to be expected. In the GFS model there are clearly some errors for the wave direction as there are outliers located at -1, which indicates waves coming from the south.')
            ])
    elif item == 'fl':
        return html.Div([
            dcc.Graph(
              figure= px.line(com_f,                 
              title="Comparison of wave energy flux"),
              className='six columns'
              ), 
            dcc.Graph( 
              figure= px.box(com_f,
                              title=" "),
              className='six columns'
              ),
            html.P('The mean value of the wave energy flux is very close in both models. The STD is significantly higher in the GFS model, and in this model the outliers are also much higher then in the EWAM model.')
            ])


@app.callback(
    Output("box-plot", "figure"), 
    [Input("y-axis", "value")])

def generate_chart(y):
    fig = px.box(Model.df, y=y)
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)