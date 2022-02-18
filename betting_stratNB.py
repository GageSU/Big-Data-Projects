from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
data= pd.read_csv("./ufc-master.csv")

features = ["R_odds","B_odds","B_current_lose_streak","lose_streak_dif","win_dif",\
            "loss_dif","total_round_dif","ko_dif","sub_dif","height_dif","reach_dif",\
            "age_dif","sig_str_dif","avg_sub_att_dif","avg_td_dif"]
target = ["Winner"]
data = data[features+target].dropna()
data["Winner_clean"] = [0 if x=="Red" else 1 if x=="Blue" else 2 for x in data.Winner]
data= data.drop("Winner", axis=1)
#print("data= ",data)
X= data[features]
Y= data["Winner_clean"]

model =GaussianNB()

def getStrategyPayout(X, Y, test_perc,model, stake_per_bet,capital,prob_threshold,scale=True):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_perc, random_state=1)
    original_odds_test = X_test[['R_odds','B_odds']] 
    #original_odds_train = X_train[['R_odds','B_odds']]
    original_odds_test.reset_index(inplace=True)
    original_odds_test= original_odds_test.drop('index',axis=1)
    
    if scale:        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    def getModelOutput(X_train, X_test, y_train, y_test,model,original_odds_test):
        classifier = model
        classifier.fit(X_train, y_train)
        predictions_probs = pd.DataFrame(classifier.predict_proba(X_test), columns=classifier.classes_)
        from sklearn.metrics import accuracy_score
        y_pred = classifier.predict(X_test)
        print(f"Accuracy: {round(accuracy_score(y_test, y_pred)*100,2)}%")
        actual_results = y_test.values 
        out= pd.DataFrame(data={'Red Odds':original_odds_test.R_odds,'Red Win%':predictions_probs[0],\
                      'Blue Odds':original_odds_test.B_odds,'Blue Win%':predictions_probs[1],'Actual Result':actual_results})
        return([out,classifier])  


    
    def getPayout(odd,stake): 
        if odd<0:
            payout = -stake*100/odd
        else:
            payout = odd*stake/100
        return(payout)
    modelOutput_full = getModelOutput(X_train, X_test, y_train, y_test,model,original_odds_test)
    output = modelOutput_full[0]
    model_return = modelOutput_full[1]
    viable_bets = output.loc[np.logical_or(output['Red Win%'].values>prob_threshold,output['Blue Win%'].values>prob_threshold),:]
    print(f'Bets made in {len(viable_bets)} fights out of {len(output)}')
    
    win_counter = 0
    gain_counter = 0
    loss_counter = 0

    init_capital = capital
    for row in viable_bets.iterrows():
        if row[1]['Red Win%']>row[1]['Blue Win%']: ### Bet on Red 
            if row[1]['Actual Result']==0: ### Bet on red, red wins
                win_counter = win_counter + 1
                prof = getPayout(row[1]['Red Odds'],stake_per_bet)
                gain_counter = gain_counter + prof
                capital = capital+ prof
            else:   ### Bet on red, red loses
                loss_counter = loss_counter + stake_per_bet                
                capital = capital - stake_per_bet
        else: ### Bet on Blue
            if row[1]['Actual Result']==1: ###Bet on Blue, Blue wins
                win_counter = win_counter + 1
                prof = getPayout(row[1]['Blue Odds'],stake_per_bet)
                gain_counter = gain_counter + prof
                capital = capital+ prof
            else: ###Bet on Blue, Blue loses
                loss_counter = loss_counter + stake_per_bet  
                capital = capital - stake_per_bet
                
    print(f'Won {win_counter} out of {len(viable_bets)} bets made- Win Percentage: {round(100*win_counter/len(viable_bets),2)}%')
    print(f'Average Win Size: {round(gain_counter/win_counter,2)}')
    print(f'Average Loss Size: {round((loss_counter/(len(viable_bets)-win_counter)),2)}')
    print(f'Total Profit earned: {round(capital-init_capital,2)}')
    return([scaler, model_return,output])
nb_output = getStrategyPayout(X,Y,0.2,model,stake_per_bet =100, capital=10000, prob_threshold=0.6)
print("Naive Bayes Results")