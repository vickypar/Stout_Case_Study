#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


#pip install dash --user
# pip install nbformat
#! pip install plotly
#! pip install kaleido
#pip install xgboost


# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objs as go
import nbformat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import ensemble, metrics, tree, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA


# # Load Dataset

# In[2]:


loans = pd.read_csv("loans_full_schema.csv")


# In[3]:


loans


# In[4]:


print(loans.info())
print(loans.isna().sum())


# In[5]:


print('Plot a simple heatmap to visualize missing data\n')
plt.figure(figsize=(14, 8))
mis_val_heatmap = sns.heatmap(loans.isnull(), yticklabels=False, cbar=False, cmap='Pastel1')
plt.title('Heatmap of Missing Values', fontsize=18)
plt.xlabel('Variables', fontsize=12)
plt.xticks(rotation=30, wrap=True, fontsize=8)
mis_val_heatmap.set_xticklabels(loans.columns, rotation=80, ha="center")
plt.tight_layout()
plt.show()


# In[6]:


print("Return the unique values from each column in the dataset: \n")
for i in loans:
    print("Column: {}\n---------------------------------".format(i))
    print(loans[i].value_counts(), '\n')


# In[7]:


loans["emp_title"].nunique()


# # Visualizations

# ## #1 Pie Chart of "homeownership"

# In[8]:


# Create pie chart for unique values of "homeownership"
data = go.Pie(labels=loans["homeownership"].unique(), values=loans["homeownership"].value_counts(), hole=.3)
fig = go.Figure([data])

# Add title and x-axis labels to the figure
fig.update_layout(title="The ownership status of the applicant's residence.")
fig.show()


# In[9]:


# Store Figure
fig.write_html("docs/homeownership.html")


# ## #2 Pie Chart of "application_type"

# In[10]:


# Create pie chart for unique values of "application_type"
data = go.Pie(labels=loans["application_type"].unique(), values=loans["application_type"].value_counts(), hole=.3)
fig = go.Figure([data])

# Add title and x-axis labels to the figure
fig.update_layout(title="The type of application: either 'individual' or 'joint'.")
fig.show()

# Store Figure
fig.write_html("docs/application_type.html")


# ## #3 Boxplots of "Annual_income" and "Annual_income_joint" for joint applications

# In[11]:


fig = go.Figure()

ver = loans["annual_income"].loc[loans['application_type'] == "joint"]
fig.add_trace(go.Box(y=ver, name="annual_income"))
ver = loans["annual_income_joint"].loc[loans['application_type'] == "joint"]
fig.add_trace(go.Box(y=ver, name="annual_income_joint"))


fig.update_layout(title="Boxplots of 'Annual_income' and 'Annual_income_joint' for joint applications.",
                   yaxis_title='Income')
fig.show()


# ## #4 Bar Chart of "state"

# In[12]:


fig = go.Figure(go.Bar(
            x=loans["state"].value_counts(),
            y=loans["state"].unique(),
            orientation='h'))

fig.update_layout(title="Barchart of states.",
                  xaxis_title='Percentage of applicants',
                   yaxis_title='State')

fig.show()

# Store Figure
fig.write_html("docs/state.html")


# ## #5 Line Chart of "interest_rate" VS "annual_income"

# In[13]:


loans["verified_income"].unique()


# In[14]:


fig = go.Figure()

ver = loans.loc[loans['verified_income'] == 'Verified']
fig.add_trace(go.Scatter(x=ver["annual_income"], y=ver["interest_rate"],
                    mode='markers',
                    name='Verified Income'))

no_ver = loans.loc[loans['verified_income'] == 'Not Verified']
fig.add_trace(go.Scatter(x=no_ver["annual_income"], y=no_ver["interest_rate"],
                    mode='markers',
                    name='Not Verified Income'))

source_ver = loans.loc[loans['verified_income'] == 'Source Verified']
fig.add_trace(go.Scatter(x=source_ver["annual_income"], y=source_ver["interest_rate"],
                    mode='markers',
                    name='Source Verified Income'))


fig.update_layout(title='Interest rate as a function of the annual income of the applicant.',
                   xaxis_title='Annual Income',
                   yaxis_title='Interest Rate')
fig.show()

# Store Figure
fig.write_html("docs/interest_rate_VS_annual_income.html")


# ## #6 Boxplots of "interest_rate" for different values of "delinq_2y"

# In[15]:


fig = go.Figure()

for i in [0,1,2,3,4,5,6,7,8,9,10,13]:
    ver = loans["interest_rate"].loc[loans['delinq_2y'] == i]
    fig.add_trace(go.Box(y=ver, name=str(i)))


fig.update_layout(title="Boxplots of interest rate for different values of applicant' delinquencies in the last 2 years.",
                   xaxis_title='Number of delinquencies',
                   yaxis_title='Interest Rate')
fig.show()

# Store Figure
fig.write_html("docs/delinquencies.html")


# ## #7 Boxplot of "interest_rate"

# In[16]:


fig = go.Figure()

fig.add_trace(go.Box(y=loans["interest_rate"]))

fig.update_layout(title="Boxplot of interest rate for all applications.",
                   yaxis_title='Interest Rate')
fig.show()

# Store Figure
fig.write_html("docs/interest_rate.html")


# ## #8 Boxplots of interest rate for different loan purposes

# In[17]:


fig = go.Figure()

for i in loans['loan_purpose'].unique():
    ver = loans["interest_rate"].loc[loans['loan_purpose'] == i]
    fig.add_trace(go.Box(y=ver, name=str(i)))


fig.update_layout(title="Boxplots of interest rate for different loan purposes.",
                   xaxis_title='Loan Purpose',
                   yaxis_title='Interest Rate')
fig.show()

# Store Figure
fig.write_html("docs/loan_purpose.html")


# In[18]:


loans["loan_purpose"].value_counts()


# # Model Creation

# ## Preprocessing

# In[19]:


# Factorize categorical variables
for category in ["disbursement_method", "state", "homeownership", "verified_income", "verification_income_joint", "loan_purpose",
                "application_type", "grade", "sub_grade", "issue_month", "loan_status", "initial_listing_status"]:
    loans[category] = pd.DataFrame(pd.factorize(loans[category])[0], columns = [category])


# In[20]:


# Fill nan values of joint variables using values from the equivalent individual ones

loans.annual_income_joint.fillna(loans.annual_income, inplace=True)
loans.verification_income_joint.fillna(loans.verified_income, inplace=True)
loans.debt_to_income_joint.fillna(loans.debt_to_income, inplace=True)

# Drop columns where missing values exist in more than 50% of the observations
loans.drop(['months_since_last_delinq', 'months_since_90d_late'], axis=1, inplace=True)

# Drop categorical features where there are more than 4000 unique values
loans.drop(['emp_title'], axis=1, inplace=True)

# Train - Test split
X = loans.loc[:, loans.columns != 'interest_rate']
y = loans["interest_rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Impute numerical missing values
num_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X_train = num_imp.fit_transform(X_train)
X_test = num_imp.transform(X_test)


# In[21]:


# Normalize data 
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Hyperparameter Selection

# In[22]:


#=========== SELECT HYPERPARAMETERS =====================#
# In this phase, hyperparameters for each algorithm will be selected based on a validation set

# More specifically, training set will be further split into train and validation set
# the algorithm will be trained using the new training set and it will be tested using the validation set

x_train2, x_val, y_train2, y_val = train_test_split(X_train, y_train, random_state = 123)


#======================================== XGBoost ============================================================#
rmse = []
for i in range (15,70,5):
    model = XGBRegressor(n_estimators=i, max_depth = 4)
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    rmse.append(mean_squared_error(y_val, predict_test2, squared=False))

# Create the figure    
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.Series(range(15,70,5)), y=rmse))
fig.update_layout(title='XGBoost Algorithm.',
                   xaxis_title='Number of Estimators',
                   yaxis_title='Root Mean Square Error (RMSE)')
fig.show()

# Store Figure
fig.write_html("docs/xgboost.html")

print("XGBoost minimun rmse: ", min(rmse), " for ", 15 + 5*rmse.index(min(rmse)), " estimators.")

#======================================== k-NN ============================================================#
rmse = []
for i in range (2,40,2):
    model = KNeighborsRegressor(n_neighbors= i, weights='distance', p=2)
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    rmseAd = mean_squared_error(y_val, predict_test2, squared=False)
    rmse.append(rmseAd)

# Create the figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.Series(range(2,40,2)), y=rmse))
fig.update_layout(title='K-Nearest Neighbors Algorithm.',
                   xaxis_title='Number of Neighbors',
                   yaxis_title='Root Mean Square Error (RMSE)')
fig.show()

# Store Figure
fig.write_html("docs/knn.html")

print("KNeighborsRegressor minimun rmse: ", min(rmse), " for ", 2+2*rmse.index(min(rmse)), " neighbors.")

#======================================== Gradient Boosting ============================================================#
rmse = []
for i in range (10,100,5):
    model = ensemble.GradientBoostingRegressor(n_estimators = i, random_state = 0)
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    rmseAd = mean_squared_error(y_val, predict_test2, squared=False)
    rmse.append(rmseAd)

# Create the figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.Series(range(10,100,5)), y=rmse))
fig.update_layout(title='Gradient Boosting Algorithm.',
                   xaxis_title='Number of Estimators',
                   yaxis_title='Root Mean Square Error (RMSE)')
fig.show()

# Store Figure
fig.write_html("docs/gradient_boosting.html")

print("Gradient Boosting minimun rmse: ", min(rmse), " for ", 10+5*rmse.index(min(rmse)), " estimators.")

#======================================== AdaBoost ============================================================#
rmse = []
for i in range (15,90,5):
    dt = tree.DecisionTreeRegressor(max_depth=4, random_state = 0)
    model = ensemble.AdaBoostRegressor(base_estimator = dt, n_estimators = i)
    model.fit(x_train2,y_train2)
    predict_test2 = model.predict(x_val)
    rmseAd = mean_squared_error(y_val, predict_test2, squared=False)
    rmse.append(rmseAd)

# Create the figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.Series(range(15,90,5)), y=rmse))
fig.update_layout(title='AdaBoostRegressor Algorithm.',
                   xaxis_title='Number of Estimators',
                   yaxis_title='Root Mean Square Error (RMSE)')
fig.show()

# Store Figure
fig.write_html("docs/adaboost.html")

print("AdaBoostRegressor minimun rmse: ", min(rmse), " for ", 15+5*rmse.index(min(rmse)), " estimators.")


# In[23]:


# =============== DATA MINING ==================== ##
# ========== Define the models that will be used ===== #

#XGBoost
modelXGB = XGBRegressor(n_estimators = 40, max_depth = 4, random_state = 0)

#K-NN
modelKnn = KNeighborsRegressor(n_neighbors= 12, weights='distance', p=2)

#Gradient Boosting Classifier
modelGBC = ensemble.GradientBoostingRegressor(n_estimators = 60, random_state = 0)

#AdaBoost
dt = tree.DecisionTreeRegressor(max_depth=4, random_state = 0)
modelAda = ensemble.AdaBoostRegressor(base_estimator = dt, n_estimators = 35, random_state=0)

# Define a list of algorithms' names and the algorithms
algo_names = ["XGBoost", "K-Nearest Neighbors", "Gradient Boosting", "AdaBoost"]
algorithms = [modelXGB, modelKnn, modelGBC, modelAda]


#Train each algorithm in the training set and evaluate it using the test set
#in terms of rmse, r-square and mae.

fig = go.Figure()
metrics = ['RMSE', 'R-Square', 'MAE']

for i in range(0,len(algorithms)):
    algorithms[i].fit(X_train, y_train)
    y_pred = algorithms[i].predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r_square = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    metr = [rmse, r_square, mae]
    fig.add_trace(go.Bar(name=algo_names[i], x=metrics, y=metr))
    print(algo_names[i])
    print("RMSE: %2f" % rmse)
    print("R Square: %2f" % r_square)
    print("MAE: %2f" % mae)
    print("=================================================")

# ============ Plot the bar chart ===========================================


fig.update_layout(title="Performance of all of the regressors.",
                   xaxis_title='Evaluation Metrics')
fig.show()

# Store Figure
fig.write_html("docs/results.html")


# In[ ]:




