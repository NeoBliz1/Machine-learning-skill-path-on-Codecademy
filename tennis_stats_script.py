import seaborn
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## load and investigate the data here:
tennis_stats = pd.read_csv('tennis_stats.csv')
dataFrame = pd.DataFrame(tennis_stats)
print(dataFrame.head())

## perform exploratory analysis here:

figure_exploratory_analysis = plt.figure(figsize=(13,9))
scat1 = figure_exploratory_analysis.add_subplot(221)
scat2 = figure_exploratory_analysis.add_subplot(222)
scat3 = figure_exploratory_analysis.add_subplot(223)
scat4 = figure_exploratory_analysis.add_subplot(224)


# BreakPointsOpportunities vs Winnings
scat1.set_xlabel('Winnings: total winnings in USD($) in a year')
scat1.set_ylabel('BreakPointsOpportunities')
scat1.set_title('BreakPointsOpportunities vs Winnings')
scat1.scatter(dataFrame['BreakPointsOpportunities'],dataFrame['Winnings'], alpha=0.4)



# Aces vs Winnings  

scat2.set_xlabel('Wins: number of matches won in a year')
scat2.set_ylabel('Aces')
scat2.set_title('Aces vs Wins')
scat2.scatter(dataFrame['Aces'], dataFrame['Wins'], alpha=0.4)


# TotalPointsWon vs Losses  

scat3.set_xlabel('Losses: number of matches lost in a year')
scat3.set_ylabel('TotalPointsWon')
scat3.set_title('TotalPointsWon vs Losses')
scat3.scatter(dataFrame['TotalPointsWon'], dataFrame['Losses'], alpha=0.4)



# ReturnGamesPlayed vs Winnings   

scat4.set_xlabel('Winnings: total winnings in USD($) in a year')
scat4.set_ylabel('returnGamesPlayed')
scat4.set_title('ReturnGamesPlayed vs Winnings')
scat4.scatter(dataFrame['ReturnGamesPlayed'], dataFrame['Winnings'], alpha=0.4)

plt.subplots_adjust(hspace=0.8)
plt.show()


## perform single feature linear regressions here:

def round_half_up(n, decimals=0):
		multiplier = 10 ** decimals
		return math.floor(n*multiplier + 0.5) / multiplier

figure_linear_regressions = plt.figure(figsize=(13,9))
scat1 = figure_linear_regressions.add_subplot(221)
scat2 = figure_linear_regressions.add_subplot(222)
scat3 = figure_linear_regressions.add_subplot(223)
scat4 = figure_linear_regressions.add_subplot(224)

lm = LinearRegression()

# BreakPointsOpportunities and Winnings  

x_train_breakPO, x_test_breakPO, y_train_winnings, y_test_winnings = train_test_split(dataFrame[['BreakPointsOpportunities']], dataFrame[['Winnings']], train_size = 0.8)

breakPO_winnings_regr = lm.fit(x_train_breakPO, y_train_winnings)
y1_predict_winnings = breakPO_winnings_regr.predict(x_test_breakPO)

train_winnings_breakPO_score =  round_half_up(breakPO_winnings_regr.score(x_train_breakPO, y_train_winnings), 2)
test_winnings_breakPO_score =  round_half_up(breakPO_winnings_regr.score(x_test_breakPO, y_test_winnings), 2)
scat1_xlabel = ['Actual winnings (', 'Train score:', train_winnings_breakPO_score, ' Test score:', test_winnings_breakPO_score, ')']
scat1_xlabel = ' '.join(map(str, scat1_xlabel))
scat1.set_xlabel(scat1_xlabel)
scat1.set_ylabel('Predicted winnings')
scat1.set_title('BreakPointsOpportunities')
scat1.scatter(y_test_winnings, y1_predict_winnings, alpha=0.4)

# Aces and Wins 
x_train_aces, x_test_aces, y_train_wins, y_test_wins = train_test_split(dataFrame[['Aces']], dataFrame[['Winnings']], train_size = 0.8)

ace_wins_regr = lm.fit(x_train_aces, y_train_wins)
y2_predict_wins = ace_wins_regr.predict(x_test_aces)

train_wins_aces_score =  round_half_up(ace_wins_regr.score(x_train_aces, y_train_wins), 2)
test_wins_aces_score =  round_half_up(ace_wins_regr.score(x_test_aces, y_test_wins), 2)
scat2_xlabel = ['Actual wins (', 'Train score:', train_wins_aces_score, ' Test score:', test_wins_aces_score, ')']
scat2_xlabel = ' '.join(map(str, scat2_xlabel))
scat2.set_xlabel(scat2_xlabel)
scat2.set_ylabel('Predicted wins.')
scat2.set_title('Aces')
scat2.scatter(y_test_wins, y2_predict_wins, alpha=0.4)

# TotalPointsWon and Losses
x_train_totalPW, x_test_totalPW, y_train_losses, y_test_losses = train_test_split(dataFrame[['TotalPointsWon']], dataFrame[['Losses']], train_size = 0.8)

totalPW_losses_regr = lm.fit(x_train_totalPW, y_train_losses)
y3_predict_losses = totalPW_losses_regr.predict(x_test_totalPW)

train_losses_totalPointsWon_score =  round_half_up(totalPW_losses_regr.score(x_train_totalPW, y_train_losses), 2)
test_losses_totalPointsWon_score =  round_half_up(totalPW_losses_regr.score(x_test_totalPW, y_test_losses), 2)
scat3_xlabel = ['Actual winnings (', 'Train score:', train_losses_totalPointsWon_score, ' Test score:', test_losses_totalPointsWon_score, ')']
scat3_xlabel = ' '.join(map(str, scat3_xlabel))
scat3.set_xlabel(scat3_xlabel)
scat3.set_ylabel('Predicted winnings')
scat3.set_title('TotalPointsWon')
scat3.scatter(y_test_losses, y3_predict_losses, alpha=0.4)

# ReturnGamesPlayed and Winnings
x_train_returnGP, x_test_returnGP, y_train_winnings, y_test_winnings = train_test_split(dataFrame[['ReturnGamesPlayed']], dataFrame[['Winnings']], train_size = 0.8)

returnGP_winnings_regr = lm.fit(x_train_returnGP, y_train_winnings)
y4_predict_winnings = returnGP_winnings_regr.predict(x_test_returnGP)

train_winnings_returnGP_score =  round_half_up(returnGP_winnings_regr.score(x_train_returnGP, y_train_winnings), 2)
test_winnings_returnGP_score =  round_half_up(returnGP_winnings_regr.score(x_test_returnGP, y_test_winnings), 2)
scat4_xlabel = ['Actual winnings (', 'Train score:', train_winnings_returnGP_score, ' Test score:', test_winnings_returnGP_score, ')']
scat4_xlabel = ' '.join(map(str, scat4_xlabel))
scat4.set_xlabel(scat4_xlabel)
scat4.set_ylabel('Predicted winnings.')
scat4.set_title('ReturnGamesPlayed')
scat4.scatter(y_test_winnings, y4_predict_winnings, alpha=0.4)

plt.subplots_adjust(hspace=0.8)
plt.show()


## perform two feature linear regressions here:

figure_TF_linear_regressions = plt.figure(figsize=(13,6))
scat1 = figure_TF_linear_regressions.add_subplot(121)
scat2 = figure_TF_linear_regressions.add_subplot(122)

# winnings, breakPointsOpportunities and returnGamesPlayed prediction
X_bPO_bPF = dataFrame[['BreakPointsOpportunities', 'BreakPointsFaced']]
y_winnings = dataFrame[['Winnings']]
x_train_bPO_bPF, x_test_bPO_bPF, y_train_winnings, y_test_winnings = train_test_split(X_bPO_bPF, y_winnings, train_size = 0.8)

winnings_bPO_bPF_regr = lm.fit(x_train_bPO_bPF, y_train_winnings)
y1_predict_winnings = winnings_bPO_bPF_regr.predict(x_test_bPO_bPF)

train_winnings_bPO_bPF_score =  round_half_up(winnings_bPO_bPF_regr.score(x_train_bPO_bPF, y_train_winnings), 2)
test_winnings_bPO_bPF_score =  round_half_up(winnings_bPO_bPF_regr.score(x_test_bPO_bPF, y_test_winnings), 2)
scat1_x2F_label = ['Actual winnings (', 'Train score:', train_winnings_bPO_bPF_score, ' Test score:', test_winnings_bPO_bPF_score, ')']
scat1_x2F_label = ' '.join(map(str, scat1_x2F_label))
scat1.set_xlabel(scat1_x2F_label)
scat1.set_ylabel('Predicted winnings')
scat1.set_title('breakPointsOpportunities and returnGamesPlayed')
scat1.scatter(y_test_winnings, y1_predict_winnings, alpha=0.4)

# ReturnGamesPlayed, ServiceGamesPlayed and winnings prediction
X_returnGP_serviceGP = dataFrame[['ReturnGamesPlayed', 'ServiceGamesPlayed']]
y_winnings = dataFrame[['Winnings']]
x_train_returnGP_serviceGP, x_test_returnGP_serviceGP, y_train_winnings, y_test_winnings = train_test_split(X_returnGP_serviceGP, y_winnings, train_size = 0.8)

winnings_returnGP_serviceGP_regr = lm.fit(x_train_returnGP_serviceGP, y_train_winnings)
y2_predict_winnings = winnings_returnGP_serviceGP_regr.predict(x_test_returnGP_serviceGP)

train_winnings_returnGP_serviceGP_score =  round_half_up(winnings_returnGP_serviceGP_regr.score(x_train_returnGP_serviceGP, y_train_winnings), 2)
test_winnings_returnGP_serviceGP_score =  round_half_up(winnings_returnGP_serviceGP_regr.score(x_test_returnGP_serviceGP, y_test_winnings), 2)
scat2_x2F_label = ['Actual winnings (', 'Train score:', train_winnings_returnGP_serviceGP_score, ' Test score:', test_winnings_returnGP_serviceGP_score, ')']
scat2_x2F_label = ' '.join(map(str, scat2_x2F_label))
scat2.set_xlabel(scat2_x2F_label)
scat2.set_ylabel('Predicted winnings')
scat2.set_title('ReturnGamesPlayed and ServiceGamesPlayed')
scat2.scatter(y_test_winnings, y2_predict_winnings, alpha=0.4)

plt.show()


## perform multiple feature linear regressions here:
### multiple regression for 8 futures
figure_multi_linear_regressions = plt.figure(figsize=(13,6))
scat1 = figure_multi_linear_regressions.add_subplot(121)
scat2 = figure_multi_linear_regressions.add_subplot(122)

X_8_multi_features = dataFrame[['Aces',	'BreakPointsFaced',	'BreakPointsOpportunities',	'DoubleFaults',	'ReturnGamesPlayed',	'ServiceGamesPlayed',	'TotalPointsWon',	'TotalServicePointsWon']]
y8_winnings = dataFrame[['Winnings']]
x8_train, x8_test, y8_train, y8_test = train_test_split(X_8_multi_features, y8_winnings, train_size = 0.8)

regr8 = lm.fit(x8_train, y8_train)
y8_predict = regr8.predict(x8_test)

train_8_score =  round_half_up(regr8.score(x8_train, y8_train), 2)
test_8_score =  round_half_up(regr8.score(x8_test, y8_test), 2)
scat1_x8label = ['Actual futures (', 'Train score:', train_8_score, ' Test score:', test_8_score, ')']
scat1_x8label = ' '.join(map(str, scat1_x8label))
scat1.set_xlabel(scat1_x8label)
scat1.set_ylabel('Predicted winnings')
scat1.set_title('Predicted vs Actual winnings 8 futures')
scat1.scatter(y8_test, y8_predict, alpha=0.4)

### multiple regression for all futures
X_AF_multi_features = dataFrame[['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon', 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon', 'TotalServicePointsWon']]
y_AF_winnings = dataFrame[['Winnings']]
x_AF_train, x_AF_test, y_AF_train, y_AF_test = train_test_split(X_AF_multi_features, y_AF_winnings, train_size = 0.8)

regrAF = lm.fit(x_AF_train, y_AF_train)
y_AF_winnings_predict = regrAF.predict(x_AF_test)

train_AF_score =  round_half_up(regrAF.score(x_AF_train, y_AF_train), 2)
test_AF_score =  round_half_up(regrAF.score(x_AF_test, y_AF_test), 2)
scat2_xAFlabel = ['Actual winnings (', 'Train score:', train_AF_score, ' Test score:', test_AF_score, ')']
scat2_xAFlabel = ' '.join(map(str, scat2_xAFlabel))
scat2.set_xlabel(scat2_xAFlabel)
scat2.set_ylabel('Predicted winnings')
scat2.set_title('Predicted vs Actual winnings all futures')
scat2.scatter(y_AF_test, y_AF_winnings_predict, alpha=0.4)

plt.show()
plt.clf()
plt.cla()