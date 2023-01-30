# League of Legends Analysis

**League of Legends** is one of the most popular videos games in the world. Developed and published in 2009 by Riot Games, League of Legends is a free Multiplayer Online Battle Arena game. In the game, two teams (blue and red) face off against one another, with the main objective being to destroy the opposing team's base, known as the Nexus. Due to the competitive nature of the game, League of Legends players regularly employ in-game strategies and tactics in hopes of increasing their chances of winning. As a result, the goal of this analysis will be to identify the factor that has the largest correlation to winning a match. The results of this analysis can potentially help beginners identify which factor(s) of the game they should prioritize in order to give themselves a better chance of winning.

The dataset used to perform this analysis was downloaded from [Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min).

## Initial Analysis
Since the dataset contains a column for `blueWins`, the first step in my analysis was to isolate the statistics for the blue team. The reason behind this decision was because I wanted to focus on the factors that players in the blue team had control over. My next step involved using the `scikit-learn` Python library to perform a multivariable linear regression. This yielded the following output: 
```
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
```
```
Accuracy:  0.2781881699438524
```

A problem that I noticed with this regression was that the accuracy of the model was very low, with a score of roughly 28%. In response, I used the `statsmodels` library to run another regression with the purpose of obtaining another sample. This model also yielded a 28% coefficient of determination (as seen below), which signified that something was wrong. 

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               blueWins   R-squared:                       0.278
Model:                            OLS   Adj. R-squared:                  0.277
Method:                 Least Squares   F-statistic:                     237.6
Date:                Wed, 25 Jan 2023   Prob (F-statistic):               0.00
Time:                        22:18:24   Log-Likelihood:                -5559.8
No. Observations:                9879   AIC:                         1.115e+04
Df Residuals:                    9862   BIC:                         1.128e+04
Df Model:                          16                                         
Covariance Type:            nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                            0.1861      0.185      1.006      0.314      -0.176       0.549
blueWardsPlaced                 -0.0003      0.000     -1.444      0.149      -0.001       0.000
blueWardsDestroyed               0.0005      0.002      0.246      0.806      -0.003       0.004
blueFirstBlood                   0.0205      0.010      2.141      0.032       0.002       0.039
blueKills                       -0.0061      0.005     -1.156      0.248      -0.017       0.004
blueDeaths                      -0.0086      0.004     -2.376      0.018      -0.016      -0.002
blueAssists                     -0.0023      0.002     -1.119      0.263      -0.006       0.002
blueEliteMonsters                0.0377      0.005      7.590      0.000       0.028       0.047
blueDragons                      0.0621      0.007      8.559      0.000       0.048       0.076
blueHeralds                     -0.0244      0.008     -2.988      0.003      -0.040      -0.008
blueTowersDestroyed             -0.0807      0.022     -3.639      0.000      -0.124      -0.037
blueTotalGold                 3.881e-05   1.48e-05      2.621      0.009    9.79e-06    6.78e-05
blueAvgLevel                    -0.0074      0.032     -0.227      0.821      -0.071       0.056
blueTotalExperience          -3.811e-06   1.12e-05     -0.339      0.734   -2.58e-05    1.82e-05
blueTotalMinionsKilled          -0.0008      0.000     -2.251      0.024      -0.001   -9.82e-05
blueTotalJungleMinionsKilled     0.0004      0.001      0.659      0.510      -0.001       0.002
blueGoldDiff                  5.431e-05   8.31e-06      6.538      0.000     3.8e-05    7.06e-05
blueExperienceDiff            4.408e-05   6.25e-06      7.055      0.000    3.18e-05    5.63e-05
blueCSPerMin                  -7.61e-05   3.38e-05     -2.251      0.024      -0.000   -9.82e-06
blueGoldPerMin                3.881e-06   1.48e-06      2.621      0.009    9.79e-07    6.78e-06
==============================================================================
Omnibus:                     3953.484   Durbin-Watson:                   1.991
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              495.092
Skew:                           0.006   Prob(JB):                    3.11e-108
Kurtosis:                       1.903   Cond. No.                     1.47e+20
==============================================================================
```

## Finding a Solution
From here, I tried various strategies in hopes of increasing the coefficient of determination. The first thing I did was narrow down the factors by eliminating those that might have had a high degree of collinearity with another factor; this included `blueGoldPerMin`, `blueCSPerMin`, `blueExperienceDiff`, and `blueGoldDiff`. Nonetheless, I still was not able to obtain the result that I was looking for, as the coefficient of determination did not budge. 

At this point, knowing that the `blueWins` data was binary, I decided that it would be more appropriate if I used the `scikit-learn` library to run a multi-class logistic regression on the remaining factors; this yielded the following coefficients: 
```
blueWardsPlaced  :  -0.0021983962913219063
blueWardsDestroyed  :  -0.00565297385525353
blueFirstBlood  :  0.024646631810065673
blueKills  :  0.18309956960175328
blueDeaths  :  -0.35320204808619476
blueAssists  :  -0.03543093423176732
blueEliteMonsters  :  0.06530735666787041
blueDragons  :  0.050740532814767494
blueHeralds  :  0.014566823849407409
blueTowersDestroyed  :  0.01007489959068934
blueTotalGold  :  0.0003735805410723446
blueAvgLevel  :  -0.03766042577788362
blueTotalExperience  :  -0.0002502353738830023
blueTotalMinionsKilled  :  -0.0033029551340255656
blueTotalJungleMinionsKilled  :  0.01131535522446172
```
```
Accuracy:  0.7226720647773279
```
The accuracy of this model was roughly 72%, a significant jump from the accuracies of the previous regression models. From this point onwards, any strategy that I tried would simply decrease the accuracy of the model. 
## Conclusion
Based on the output above, since the `blueKills` factor has the largest positive coefficient, my conclusion is that the factor with the largest correlation to winning a game in League of Legends is the **number of enemies killed by a team**. 

<br/>

## Data Visualizations
### Average Level vs. Number of Kills AND Number of Deaths vs. Number of Kills
![Regressions between variables](https://github.com/Kevinl0378/league-of-legends-analysis/blob/main/Regressions%20between%20variables.png)

### Individual Logistic Regressions 
![Logistic Regressions](https://github.com/Kevinl0378/league-of-legends-analysis/blob/main/Logistic%20Regressions.png)

