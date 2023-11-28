from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
import os



#df_template = pd.read_csv(os.path.join(dir,'app\\dataframe_template.csv'),index_col=0)

df_template = pd.read_csv('dataframe_template.csv',index_col=0)

number_variable_df = pd.read_csv('app_number_variable_list.csv',index_col=0)

intubation_pipe = joblib.load("model/adaboost_median_impute_feature_oversample_intubation_dummy_sigmoid_calibration.pickle")
mortality_pipe = joblib.load("model/adaboost_median_impute_feature_oversample_death_isotonic_calibration.pickle")
cardiac_pipe = joblib.load("model/rf_selection_median_impute_final_oversample_cardio_complication_sigmoid_calibration.pickle")

MIN_PROGRESS_BAR = 4

app = Dash(external_stylesheets=[dbc.themes.SANDSTONE,dbc.icons.FONT_AWESOME])

server = app.server

card_list = []
for ele in number_variable_df.index:
    card_row = dbc.Row(
            [
                dbc.Label(number_variable_df.loc[ele,"label"],class_name="bold", width=5),
                dbc.Col(
                    [dbc.Input(
                        id='{}-input'.format(ele), 
                        type='number',
                        placeholder="Between {}-{}".format(number_variable_df.loc[ele,"min"],number_variable_df.loc[ele,"max"]),
                ),
                ]),
            ],
            className='mb-2'
        )
    card_list.append(card_row)

boolean_var_dict = {
    'healthworker': 'Healthcare worker',
    'smoker':'Smoker',
    'copd':'COPD',
    'vaccinated':'Vaccinated for COVID-19',
    'troponin':'Troponin above upper limit normal',
    'asthma':'History of asthma',
    'coronaryhistory':'Medicial history of coronary disease',
    'hypertension':'History of hypertension',
    'pacemaker':'Pacemaker'
}

boolean_card_list = []
for k,v in boolean_var_dict.items():
    card_row = html.Div(
            [
                dbc.Label(v),
                dbc.Switch(
                    id='{}-input'.format(k),
                    value=False,
                ),
            ],
            className='mb-2'
        )
    boolean_card_list.append(card_row)

controls = dbc.Card(
    [dbc.Row(
        [
            dbc.Col(card_list,lg=6),
            dbc.Col(
                [
        html.Div(
                [
                    dbc.Label('Dialysis'),
                    dbc.RadioItems(
                        id='dialysis-input',
                        options=[
                            {'label':'None','value':0},
                            {'label':'Currently on dialysis','value':1},
                            {'label':'Previously on dialysis','value':2}
                        ],
                        value=0,
                        inline=True
                    ),
                ],
                className='mb-2'
            ),
            html.Div(
                [
                    dbc.Label(
                        'Chest xray'),
                    dbc.RadioItems(
                        id='chestxray-input',
                        options = [
                            {'label':'No xray available','value':0},
                            {'label':'Features of COVID-19 present','value':1},
                            {'label':'No features of COVID-19 present','value':2}
                        ],
                        value= 0
                    ),
                ],
                className='mb-2'
            ),
            html.Div(
                [
                    dbc.Label(
                        'Sex'),
                    dbc.RadioItems(
                        id='sex-input',
                        options = ['Male', 'Female'],
                        value= 'Male',
                        inline=True
                    ),
                ],
                className='mb-2'
            )]
        + boolean_card_list
            )

        ]
    )]
        + [
        html.Hr(),
            html.Div(
                [
                    dbc.Button(
                        "Calculate", id="example-button", className="d-grid gap-2 col-6 mx-auto", color='primary',n_clicks=0
                    ),
                ]
            )
        ],
    body=True,
)

intubation_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("Intubation", className="card-subtitle mb-2"),
            dbc.Progress(
                value=MIN_PROGRESS_BAR, id="intubation-prob", animated=True, striped=True,style={"height": "20px","bg-color":'white'}, color='primary'
            )
        ]
    ),
    className="mb-3"
)

mortality_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("In-Hospital Mortality", className="card-subtitle mb-2"),
            dbc.Progress(
                value=MIN_PROGRESS_BAR, id="mortality-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            )
        ]
    ),
    className="mb-3"
)


cardiac_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("Cardiac Complication", className="card-subtitle mb-2"),
            dbc.Progress(
                value=MIN_PROGRESS_BAR, id="cardiac-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            )
        ]
    ),
    className="mb-3"
)

risk_score = dbc.Card(
    [
        dbc.CardBody([
            html.H4("Predicted Probability", className="card-title"),
            html.Hr(),
            mortality_result,
            intubation_result,
            cardiac_result
        ])
    ]
)

offcanvas = html.Div(
    [
        dbc.Button(id="open-offcanvas", n_clicks=0,children=html.I(className = "fa-solid fa-circle-info fa-xl"),
                style={'color':'white'}),
        dbc.Offcanvas(
            [
                html.P(
                    "Some description of the project. "
                ),
                html.P(
                    [
                        "Paper  ", html.A("link", href="https://github.com/harrisonnguyen/",)
                    ]
                ),
                html.P(
                    [
                        "Code of the metholody and application can be found on ", 
                        html.A("github", href="https://github.com/harrisonnguyen/cardiac_covid_model")
                    ]
                )
                ],
            id="offcanvas",
            title="Information",
            is_open=False,
        ),
    ]
)


navbar = dbc.NavbarSimple(
    children=[
        offcanvas
    ],
    brand="AUS-COVID Risk Prediction Algorithm",
    color="primary",
    dark=True,
    class_name='mb-3',
    style={'border-radius': '10px'}
)

app.layout = html.Div(
    [

    dbc.Container(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(md=1),
                dbc.Col(controls, md=10),
                dbc.Col(md=1),
                #dbc.Col(dcc.Graph(id='graph-content'), md=4)

            ],
            align="center",
            class_name='mb-3',
        ),
        dbc.Row([
                dbc.Col(xxl=3),
                dbc.Col([
                        risk_score
                ], xxl=6,align='centre'),    
            ],
            align="center",class_name='mx-auto',
        )
        #html.Hr(),
        

    ],
        fluid='md',
    ),
    dbc.Container([
    dbc.Row(
            [
                dbc.Col(dbc.Table(id='table-content', striped=True, bordered=True, hover=True), md=12),
            ],
            align="center",
        )
    ],
    fluid=True)
    ]

)

@callback(
    Output('table-content', 'children'),
    Output('intubation-prob', 'value'),
    Output('intubation-prob', 'label'),
    Output('mortality-prob', 'value'),
    Output('mortality-prob', 'label'),
    Output('cardiac-prob', 'value'),
    Output('cardiac-prob', 'label'),
    Input('example-button', 'n_clicks'),
    State('age-input', 'value'),
    State('respiratory-input', 'value'),
    State('spo2-input', 'value'),
    State('lymphocytes-input', 'value'),
    State('wcc-input', 'value'),
    State('platelets-input', 'value'),
    State('ldh-input', 'value'),
    State('creatinine-input', 'value'),
    State('alt-input', 'value'),
    State('crp-input', 'value'),
    State('dialysis-input', 'value'),
    State('chestxray-input', 'value'),
    State('sex-input', 'value'),
    State('healthworker-input', 'value'),
    State('smoker-input', 'value'),
    State('copd-input', 'value'),
    State('vaccinated-input', 'value'),
    State('troponin-input', 'value'),
    State('asthma-input', 'value'),
    State('coronaryhistory-input', 'value'),
    State('hypertension-input', 'value'),
    State('pacemaker-input', 'value'),
    prevent_initial_call=True
)
def predict_risk(n_clicks,age,respiratory,spo2,lymphocytes,wcc,platelets,ldh,
                 creatinine,alt,crp,dialysis,chestxray,sex,healthworker,smoker,copd,
                 vaccinated,troponin,asthma,coronaryhistory,hypertension,pacemaker):
    first_row = 0
    df_template.loc[first_row,'age'] = age
    df_template.loc[first_row,'respiratory_rate'] = respiratory
    df_template.loc[first_row,'spo2'] = spo2
    df_template.loc[first_row,'lymphocytes'] = lymphocytes
    df_template.loc[first_row,'total_wcc'] = wcc
    df_template.loc[first_row,'platelets'] = platelets
    df_template.loc[first_row,'ldh'] = ldh
    df_template.loc[first_row,'creatinine'] = creatinine
    df_template.loc[first_row,'alt'] = alt
    df_template.loc[first_row,'crp'] = crp

    
    # dummy variables
    if dialysis == 0:
        df_template.loc[first_row,'dialysis_2.0'] = 0
        df_template.loc[first_row, 'dialysis_3.0'] = 1
    elif dialysis == 1:
        df_template.loc[first_row, 'dialysis_2.0'] = 0
        df_template.loc[first_row, 'dialysis_3.0'] = 0
    else:
        df_template.loc[first_row, 'dialysis_2.0'] = 1
        df_template.loc[first_row, 'dialysis_3.0'] = 0


    if chestxray == 0:
        df_template.loc[first_row, 'chest_xray_2'] = 0
        df_template.loc[first_row, 'chest_xray_3'] = 0
    elif chestxray == 1:
        df_template.loc[first_row, 'chest_xray_2'] = 1
        df_template.loc[first_row, 'chest_xray_3'] = 0
    else:
        df_template.loc[first_row, 'chest_xray_2'] = 0
        df_template.loc[first_row, 'chest_xray_3'] = 1

    if sex == 'Male':
        df_template.loc[first_row, 'sex_2'] = 0
    else:
        df_template.loc[first_row, 'sex_2'] = 1
    
    if healthworker:
        df_template.loc[first_row, 'healthcare_worker_2.0'] = 0
    else:
        df_template.loc[first_row, 'healthcare_worker_2.0'] = 1

    if smoker:
        df_template.loc[first_row, 'smoker_2.0'] = 0
    else:
        df_template.loc[first_row, 'smoker_2.0'] = 1
    
    if copd:
        df_template.loc[first_row, 'copd_2.0'] = 0
    else:
        df_template.loc[first_row, 'copd_2.0'] = 1
    
    if vaccinated:
        df_template.loc[first_row, 'vaccinated_1'] = 1
    else:
        df_template.loc[first_row, 'vaccinated_1'] = 0

    if asthma:
        df_template.loc[first_row, 'asthma_2.0'] = 0
    else:
        df_template.loc[first_row, 'asthma_2.0'] = 1
    
    if troponin:
        df_template.loc[first_row, 'troponin_uln_2.0'] = 0
    else:
        df_template.loc[first_row, 'troponin_uln_2.0'] = 1
    
    if coronaryhistory:
        df_template.loc[first_row, 'coronary_med_history_col_True'] = 1
    else:
        df_template.loc[first_row, 'coronary_med_history_col_True'] = 0
    
    if hypertension:
        df_template.loc[first_row, 'hypertension_2.0'] = 0
    else:
        df_template.loc[first_row, 'hypertension_2.0'] = 1
    
    if pacemaker:
        df_template.loc[first_row, 'pacemaker_1'] = 1
    else:
        df_template.loc[first_row, 'pacemaker_1'] = 0

    if (
        check_age_validity(age) or  
        check_respiratory_validity(respiratory) or  
        check_spo2_validity(spo2) or
        check_lymphocytes_validity(lymphocytes) or
        check_wcc_validity(wcc) or
        check_platelets_validity(platelets) or
        check_ldh_validity(ldh) or
        check_creatinine_validity(creatinine) or
        check_alt_validity(alt) or
        check_crp_validity(crp)):
        return (
            dbc.Table.from_dataframe(df_template),
            MIN_PROGRESS_BAR,
            "",
            MIN_PROGRESS_BAR,
            "",
            MIN_PROGRESS_BAR,
            ""
    )


    # do prediction
    intubation_pred = round(intubation_pipe.predict_proba(df_template)[0,1],2)*100
    intubation_prob = "{:.0%}".format(intubation_pred/100)

    mortality_pred = round(mortality_pipe.predict_proba(df_template)[0,1],2)*100
    mortality_prob = "{:.0%}".format(mortality_pred/100)

    cardiac_pred = round(cardiac_pipe.predict_proba(df_template)[0,1],2)*100
    cardiac_prob = "{:.0%}".format(cardiac_pred/100)

    # adjust the progress bar so that its not empty 
    # visual reasons
    if intubation_pred < MIN_PROGRESS_BAR:
        intubation_pred = MIN_PROGRESS_BAR
    if mortality_pred < MIN_PROGRESS_BAR:
        mortality_pred = MIN_PROGRESS_BAR
    if cardiac_pred < MIN_PROGRESS_BAR:
        cardiac_pred = MIN_PROGRESS_BAR
    print(df_template)
    

    return (
        dbc.Table.from_dataframe(df_template),
        intubation_pred,
        intubation_prob,
        mortality_pred,
        mortality_prob,
        cardiac_pred,
        cardiac_prob
    )


@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


""""
Input checking
"""
@app.callback(
    Output("age-input", "invalid"),
    Input("age-input", "value"),
)
def check_age_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["age","min"] or value > number_variable_df.loc["age","max"]
        return is_invalid
    return False


@app.callback(
    Output("respiratory-input", "invalid"),
    Input("respiratory-input", "value"),
)
def check_respiratory_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["respiratory","min"] or value > number_variable_df.loc["respiratory","max"]
        return is_invalid
    return False

@app.callback(
    Output("spo2-input", "invalid"),
    Input("spo2-input", "value"),
)
def check_spo2_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["spo2","min"] or value > number_variable_df.loc["spo2","max"]
        return is_invalid
    return False
@app.callback(
    Output("lymphocytes-input", "invalid"),
    Input("lymphocytes-input", "value"),
)
def check_lymphocytes_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["lymphocytes","min"] or value > number_variable_df.loc["lymphocytes","max"]
        return is_invalid
    return False
@app.callback(
    Output("wcc-input", "invalid"),
    Input("wcc-input", "value"),
)
def check_wcc_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["wcc","min"] or value > number_variable_df.loc["wcc","max"]
        return is_invalid
    return False
@app.callback(
    Output("platelets-input", "invalid"),
    Input("platelets-input", "value"),
)
def check_platelets_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["platelets","min"] or value > number_variable_df.loc["platelets","max"]
        return is_invalid
    return False
@app.callback(
    Output("ldh-input", "invalid"),
    Input("ldh-input", "value"),
)
def check_ldh_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["ldh","min"] or value > number_variable_df.loc["ldh","max"]
        return is_invalid
    return False
@app.callback(
    Output("creatinine-input", "invalid"),
    Input("creatinine-input", "value"),
)
def check_creatinine_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["creatinine","min"] or value > number_variable_df.loc["creatinine","max"]
        return is_invalid
    return False
@app.callback(
    Output("alt-input", "invalid"),
    Input("alt-input", "value"),
)
def check_alt_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["alt","min"] or value > number_variable_df.loc["alt","max"]
        return is_invalid
    return False
@app.callback(
    Output("crp-input", "invalid"),
    Input("crp-input", "value"),
)
def check_crp_validity(value):
    if value:
        is_invalid = value < number_variable_df.loc["crp","min"] or value > number_variable_df.loc["crp","max"]
        return is_invalid
    return False



if __name__ == '__main__':
    app.run(debug=True)