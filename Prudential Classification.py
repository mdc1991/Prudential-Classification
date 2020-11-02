#Prudential Classification Challenge
import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.figure_factory as ff
import category_encoders as ce
import xgboost as xgb

from plotly.subplots import make_subplots
from ipywidgets import widgets
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from time import time

train_df = pd.read_csv(r'C:\Users\markc\OneDrive\Documents\Python\100Hours\Classification\Prudential\train.csv')
test_df = pd.read_csv(r'C:\Users\markc\OneDrive\Documents\Python\100Hours\Classification\Prudential\test.csv')

#data types

cat_features = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']

cont_features = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']

disc_features = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']

dummy_features = ['Medical_Keyword_' + str(i) for i in range(1, 49)]

#plot count of responses
bar_df = train_df['Response'].value_counts().reset_index().sort_values(by='Response', ascending=False)

bar_df['Pct'] = bar_df['Response'] / len(train_df)

x_bar = bar_df['index'].values
y_bar = bar_df['Pct'].values
text_bar = ['{0:.1%}'.format(i) for i in y_bar]

fig_bar = go.Figure()

fig_bar.add_trace(
    go.Bar(
        x=x_bar,
        y=y_bar,
        text=text_bar,
        textposition='outside',
        textfont=dict(
            color='black'
        )     
    )
)

fig_bar.update_traces(
    marker_color='rgb(121,194,255)',
    marker_line_color='rgb(0, 78, 143)',
    marker_line_width=1.5,
    opacity=0.6,
)

fig_bar.update_layout(
    title=dict(
        text='% Share of Response',
        x=0.5,
        y=0.9
    ),
    xaxis_title='Response',
    yaxis_title='% Share',
    xaxis=dict(
        type='category'
        ),
    plot_bgcolor='white',
    width=800,
    height=600
)

fig_bar.show()

#Data Visualisation -> Continuous features
responses = train_df['Response'].unique()
responses.sort()

subplot_titles = ['Response: ' + str(i) for i in range(1, 9)]
subplot_cols = [1, 2, 1, 2, 1, 2, 1, 2]
subplot_rows = [1, 1, 2, 2, 3, 3, 4, 4]

interactive_features = ['Ins_Age', 'Ht', 'Wt', 'BMI']

#create base graph

feature_widget = widgets.Dropdown(
    options=interactive_features,
    value=interactive_features[1],
    description='Feature:'
)

#create base chart data
int_fig = go.FigureWidget(
    make_subplots(4, 2, subplot_titles=subplot_titles)
)

for i in range(len(responses)):
    hist_data = train_df.loc[train_df['Response'] == responses[i], feature_widget.value].values

    int_fig.add_trace(
                go.Histogram(
                    x=hist_data
                ),
                col=subplot_cols[i],
                row=subplot_rows[i]
        )

int_fig.update_traces(xbins={'start' : 0, 'end' : 1, 'size' : 0.1})

int_fig.update_layout(title=dict(
    text='Distribution of ' + feature_widget.value + ' for each response 1-8',
    x=0.5,
    y=0.9),
    showlegend=False)

#create interactive element

def validate():
    if feature_widget.value in interactive_features:
        return True
    else: 
        return False

def response(change):
    if validate():
         
         for i in range(len(responses)):
            int_hist_data = train_df.loc[train_df['Response'] == i, feature_widget.value].values
            
            with int_fig.batch_update():
                int_fig.data[i]['x'] = int_hist_data
         
         int_fig.update_traces(xbins={'start' : 0, 'end' : 1, 'size' : 0.1})   
         int_fig.update_layout(
             title=dict(
                 text='Distribution of ' + feature_widget.value + ' for each response 1-8',
                 x=0.5,
                 y=0.9
             )
         )

feature_widget.observe(response, names='value')

container = widgets.HBox([feature_widget])
widgets.VBox([container, int_fig])

#evaluate missing values

def graph_missing_values(df):
    missing_values = (df.isnull().sum() / len(df))
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    missing_value_df = pd.DataFrame(missing_values, columns=['Missing pct'])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=missing_value_df.index,
        y=missing_value_df['Missing pct'].values
    ))

    fig.update_layout(title='Missing Value Percentage by Feature', xaxis_title='Feature', yaxis_title='% Missing')

    if missing_value_df.empty:
        print("No Missing Values")
    else: return fig

all_data = pd.concat([train_df, test_df])

graph_missing_values(all_data).show()

#Classification task using trees, therefore we can handle missing values -> impute -100 for a missing value
impute_to_missing = -100
all_data = all_data.fillna(impute_to_missing)

#feature engineering
#split product_info_2 into string and number elements

all_data['Product_Info_2_char'] = all_data['Product_Info_2'].str[0]
all_data['Product_Info_2_num'] = all_data['Product_Info_2'].str[1].astype(int)

#interaction between Age and BMI
all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

#medical keywords are 1-hot encoded -> add the sum of all medical keywords for each in instance
all_data['Medical_Keyword_Count'] = all_data[dummy_features].apply(lambda x: sum(x), axis=1)

#create feature with count encoded Medical Keywords

count_encoder = ce.CountEncoder()

fit_dummy_features = count_encoder.fit(train_df[dummy_features].astype(str))
dummy_features_encoded = fit_dummy_features.transform(all_data[dummy_features].astype(str))

all_data = pd.concat([all_data, dummy_features_encoded.add_suffix('_count')], axis=1)

obj_columns = list(all_data.select_dtypes('object').columns)

all_data = pd.concat([all_data, pd.get_dummies(all_data[obj_columns])], axis=1)
all_data = all_data.drop(obj_columns, axis=1)

#split back for ML training

n_train = train_df.shape[0]

all_data = all_data.drop(['Id', 'Response'], axis=1)

X = all_data[:n_train]
X_test = all_data[n_train:]

y = train_df['Response'].copy()

#plotting confusion matrix for each classifier

def plot_conf_mx(conf_mx, classifier):
    conf_mx_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / conf_mx_sums

    matrix_headings = list(responses)
    matrix_values = norm_conf_mx
    matrix_text = ['{:.1%}'.format(i) for i in matrix_values.ravel()]
    matrix_text = np.array(matrix_text).reshape(matrix_values.shape)

    fig = ff.create_annotated_heatmap(
        z=matrix_values, 
        x=matrix_headings, 
        y=matrix_headings, 
        annotation_text=matrix_text, 
        colorscale='Blues', 
        hoverinfo='z', 
        showscale=True
    )

    fig.update_layout(
        title=dict(
            text='Confusion Maxtrix for Classifier: ' + classifier,
            x=0.5,
            y=0.9
        )
    )

    return fig


forest_clf = RandomForestClassifier(n_estimators=500, random_state=42)

y_prob_forest = cross_val_predict(forest_clf, X, y, cv=5)
forest_conf_mx = confusion_matrix(y, y_prob_forest)
plot_conf_mx(forest_conf_mx, 'Random Forest').show()

svc_clf = Pipeline([
    ('std_scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])

y_prob_svc = cross_val_predict(svc_clf, X, y, cv=5)
svc_conf_mx = confusion_matrix(y, y_prob_svc)
plot_conf_mx(svc_conf_mx, 'SVC').show()

logistic_clf = LogisticRegression()
y_prob_logistic = cross_val_predict(logistic_clf, X, y, cv=5)
logistic_conf_mx = confusion_matrix(y, y_prob_logistc)
plot_conf_mx(logistic_conf_mx, 'Logistic').show()

xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=8, seed=42)

#transform data to optimized XGBoost format
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=8, seed=42)
y_prob_xgb = cross_val_predict(xgb_clf, X, y, cv=5)

xgb_conf_mx = confusion_matrix(y, y_prob_xgb)
plot_conf_mx(xgb_conf_mx, 'XGBoost')

voting_clf = VotingClassifier(
    estimators=[
        ('rf_clf', forest_clf),
        ('logistic_clf', logistic_clf),
        ('SVC_clf', SVC_clf),
        ('xgb_clf', xgb_clf)
    ]
)

y_prob_voting = cross_val_predict(voting_clf, X, y, cv=3) #reduced cv to save runtime
voting_conf_mx = confusion_matrix(y, y_prob_voting)
plot_conf_mx(voting_conf_mx, 'Voting')

#submit predictions for XGBoost
xgb_clf.fit(X, y)
xgb_predictions = xgb_clf.predict(X_test)
xgb_df = pd.DataFrame({'Id' : test_df['Id'], 'Response' : xgb_predictions})
xgb_df.to_csv(r'/home/mark/Python/xgb_predictions.csv')
#xgb_df.to_csv(r'/mnt/fromlinux/xgb_predictions.csv')

#submit predictions for voting classifier
voting_clf.fit(X, y)

voting_predictions = voting_clf.predict(X_test)
voting_df = pd.DataFrame({'Id' : test_df['Id'], 'Response' : voting_predictions})
voting_df.to_csv(r'/home/mark/Python/voting_predictions.csv')

#voting_df.to_csv(r'/mnt/fromlinux/voting_predictions.csv')