# League of Legends Analysis

**League of Legends** is one of the most popular videos games in the world. Developed and published in 2009 by Riot Games, League of Legends is a free Multiplayer Online Battle Arena game. In the game, two teams (blue and red) face off against one another, with the main objective being to destroy the opposing team's base, known as the Nexus. League of Legends players regularly employ in-game strategies and tactics in hopes of increasing their chances of winning. As a result, the goal of this analysis will be to identify the factor that has the largest correlation to winning a match. 

The dataset used to perform this analysis was downloaded from [Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min).

The first step in my analysis was to isolate the statistics for the blue team. The reason behind this decision was because I wanted to focus on the factors that players in the blue team had control over. The next step involved using the scikit-learn Python library to perform a multivariable linear regression. A problem that I noticed with this regression was that the accuracy of the model was very low, with a score of roughly 20%. In response, I tried using the statsmodel library to run another regression. However, this model also yielded a 20% coefficient of determination, which signified that something was wrong. 

From here, I tried various strategies in hopes of increasing the coefficient of determination. The first thing I did was narrow down the factors by eliminating those with a high degree of collinearity; this included CS per minute, gold per minute, etc. Nonetheless, I still was not able to obtain the result that I was looking for, which prompted me to continue my search. 

At this point, knowing that the blueWins data was binary, I decided that it would be more appropriate if I used the scikit-learn library to run a multi-class logistic regression on the remaining factors; this yielded the following coefficients. 

The accuracy of this model was roughly 70%, a significant jump from the accuracies of the previous regression models. From this point onwards, any strategy that I tried would simply decrease the accuracy of the model. As a result, since the blueKills factor has the largest coefficient, my conclusion is that the factor with the largest corelation to winning a game in League of Legends is the number of enemies killed by a team. 
