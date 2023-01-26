import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns


df = pd.read_csv('high_diamond_ranked_10min.csv')
column_names = list(df.columns)

# Creating a dataframe that only consists of blue statistics
blue_stats = []
for name in column_names:
    if "blue" in name:
        blue_stats.append(name)
df_blue = df[blue_stats]
print(blue_stats)


#Method 1: Normal Multivariable Regression
independent_var = blue_stats[1:]
indep_var_stats = df_blue[independent_var]
# independent_var = ['blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood', 'blueKills', 'blueDeaths', 
# 'blueAssists', 'blueEliteMonsters', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 'blueTotalGold', 
# 'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled', 'blueGoldDiff', 
# 'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin']

dependent_var = blue_stats[0]
dep_var_stats = df_blue[dependent_var]
# dependent_var = blueWins

# Generating the regression coefficients using the scikit-learn library
regr = linear_model.LinearRegression()
regr.fit(indep_var_stats, dep_var_stats)
coefficients = regr.coef_
print("Accuracy: ", regr.score(indep_var_stats, dep_var_stats))
# coefficients = [-3.44700531e-04  4.90228971e-04  2.05088877e-02 -6.12293690e-03
# -8.57586566e-03 -2.26548510e-03  3.77195228e-02  6.20789862e-02
# -2.43594635e-02 -8.07197190e-02  3. 88138024e-05 -7.36219229e-03
# -3.81065774e-06 -7.61019706e-04  3.86169910e-04  5.43087331e-05
#  4.40778246e-05 -7.61019706e-05  3.88138026e-06]

# Creating a dictionary that contains the factors as keys and their corresponding regression coefficients as values
var_coefficients = {}
for num, var in enumerate(independent_var):
    var_coefficients[var] = coefficients[num]
'''
blueWardsPlaced : -0.00034470
blueWardsDestroyed : 0.00049023
blueFirstBlood : 0.02050889
blueKills : -0.00612294
blueDeaths : -0.00857587
blueAssists : -0.00226549
blueEliteMonsters : 0.03771952
blueDragons : 0.06207899
blueHeralds : -0.02435946
blueTowersDestroyed : -0.08071972
blueTotalGold : 0.00003881
blueAvgLevel : -0.00736219
blueTotalExperience : -0.00000381
blueTotalMinionsKilled : -0.00076102
blueTotalJungleMinionsKilled : 0.00038617
blueGoldDiff : 0.00005431
blueExperienceDiff : 0.00004408
blueCSPerMin : -0.00007610
blueGoldPerMin : 0.00000388
'''

# Finding the factor that has the highest regression coefficient
highest_coefficient = 0
highest_correlation = ''
for key, value in var_coefficients.items():
    if value > highest_coefficient:
        highest_coefficient = value
        highest_correlation = key

# Different method of obtaining regression coefficients
indep_var_stats = sm.add_constant(indep_var_stats)
model = sm.OLS(dep_var_stats, indep_var_stats).fit()
predictions = model.predict(indep_var_stats)
print_model = model.summary()
#print(print_model)



#Method 2: Eliminating Factors (added `_cleaned` to all variable names)
independent_var_cleaned = blue_stats[1:-4]   
indep_var_stats_cleaned = df[independent_var_cleaned]

regr_cleaned = linear_model.LinearRegression()
regr_cleaned.fit(indep_var_stats_cleaned, dep_var_stats)
coefficients_cleaned = regr_cleaned.coef_

var_coefficients_cleaned = {}
for num, var in enumerate(independent_var_cleaned):
    var_coefficients_cleaned[var] = coefficients_cleaned[num]

indep_var_stats_cleaned = sm.add_constant(indep_var_stats_cleaned)
model_cleaned = sm.OLS(dep_var_stats, indep_var_stats_cleaned).fit()
predictions_cleaned = model_cleaned.predict(indep_var_stats_cleaned)
print_model_cleaned = model_cleaned.summary()
#print(print_model_cleaned)



#Method 3: Multi-class Logistic Regression
df_logistic = df[blue_stats[0:-4]]
print(df_logistic)
x_train, x_test, y_train, y_test = train_test_split(df_logistic.drop('blueWins', axis=1), df_logistic['blueWins'])
LogReg = LogisticRegression(solver='lbfgs', max_iter=1000)
LogReg.fit(x_train, y_train)
coefficients_logistic = LogReg.coef_
print("Accuracy: ", LogReg.score(x_test, y_test))

var_coefficients_logit = {}
for num, var in enumerate(blue_stats[1:-4]):
    var_coefficients_logit[var] = coefficients_logistic[0][num]
for key, value in var_coefficients_logit.items():
    print(key, " : ", value)

    

# Data Visualizations

df_blueKills = df['blueKills']
df_blueDeaths = df['blueDeaths']


'''
fig, ax = plt.subplots(figsize=(7,7))
g = sns.regplot(x=df_blueKills, y=df['blueAvgLevel'], ax=ax, label = 'Average Level', color = 'turquoise')
ax2 = ax.twinx()
f = sns.regplot(x=df_blueKills, y=df_blueDeaths, ax=ax2, label = 'Number of Deaths', color = 'gold')
plt.setp(g.collections, alpha=.3)
plt.setp(g.lines, alpha=.3) 
plt.setp(f.collections, alpha=.3)
plt.setp(f.lines, alpha=.3)   
ax.set(xlabel = "Number of Kills", ylabel = 'Average Level')
ax2.set(ylabel = 'Number of Deaths')
fig.legend()
plt.show()
'''


'''
fig, ax = plt.subplots(figsize=(10, 7))
sns.regplot(df, x = df_blueDeaths, y = df['blueWins'], logistic = True, label = 'Deaths', color = 'black')
sns.regplot(df, x = df['blueAssists'], y = df['blueWins'], logistic = True, label = 'Assists', color = 'gold')
sns.regplot(df, x = df_blueKills, y = df['blueWins'], logistic = True, label = 'Kills', color = 'red')
sns.regplot(df, x = df['blueAvgLevel'], y = df['blueWins'], logistic = True, label = 'Average Level', color = 'turquoise')
ax.set(ylabel='Probability of Winning', xlabel='Number of (Deaths/Assists/Kills/Levels)')
ax.legend()
plt.show()
'''
