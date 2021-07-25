import os
import joblib
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from datetime import date
from sklearn.metrics import mean_squared_error, r2_score

# option
warnings.simplefilter('ignore')





def result(model, hour):

    # load trained model
    clf_arrival = joblib.load(config.output + f'{model}_arrival_{hour}.bin') 
    clf_departure = joblib.load(config.output + f'{model}_departure_{hour}.bin') 

    # load data
    data_arrival_raw = pd.read_csv( os.path.join(config.input_dir, 'arrival_train.csv'), index_col = 0)
    data_arrival = data_arrival_raw[data_arrival_raw['taf'] == 6].reset_index(drop = True)   
    data_arrival = data_arrival.drop('taf', axis = 1)
    data_departure_raw = pd.read_csv( os.path.join(config.input_dir, 'departure_train.csv'), index_col = 0)
    data_departure = data_departure_raw[data_departure_raw['taf'] == 6].reset_index(drop = True)            # 맨 처음은 6시간 taf 쓰기
    data_departure = data_departure.drop('taf', axis = 1)

    # prediction
    X_a = data_arrival.drop('label', axis = 1).values
    y_a = data_arrival['label']
    pred_a = clf_arrival.predict(X_a)    
    X_d = data_departure.drop('label', axis = 1).values
    y_d = data_departure['label']
    pred_d = clf_departure.predict(X_d)


    """ Feature Importance """
    # arrival
    imp_arrival = pd.DataFrame({'feature': data_arrival.drop('label', axis = 1).columns, 'importance': clf_arrival.feature_importances_})
    imp_arrival = imp_arrival.sort_values('importance').set_index('feature')
    imp_arrival.plot(kind='barh', figsize = (20,20))
    plt.legend(loc='lower right')
    plt.savefig(save_dir + f'/arrival_{hour}hour_feature_importance.png')

    # departure
    imp_departure = pd.DataFrame({'feature': data_departure.drop('label', axis = 1).columns, 'importance': clf_departure.feature_importances_})
    imp_departure = imp_departure.sort_values('importance').set_index('feature')
    imp_departure.plot(kind='barh', figsize = (20,20))
    plt.legend(loc='lower right')
    plt.savefig(save_dir + f'/departure_{hour}hour_feature_importance.png')


    """ Distribution """
    # arrival
    fig_arrival = plt.figure(figsize=(20, 10))
    ax1_arrival = fig_arrival.add_subplot(1, 2, 1)
    ax2_arrival = fig_arrival.add_subplot(1, 2, 2)

    sns.scatterplot(data = pd.concat([y_a, pd.DataFrame({'Prediction_arrival' : pred_a})], axis = 1), 
                    x = "label", y = "Prediction_arrival" , ax = ax1_arrival)
    sns.distplot(y_a, label = 'AAR', kde = False, ax = ax2_arrival)
    sns.distplot(pd.DataFrame({'Prediction_arrival' : pred_a}), label = 'Prediction', kde = False, ax = ax2_arrival)
    ax1_arrival.set(xlabel='Actual Arrival Rate', ylabel='Arrival Prediction')
    ax2_arrival.set(xlabel='Arrivals per hour', ylabel='Count')
    plt.legend() 
    plt.savefig(save_dir + f'/arrival_{hour}hour_distribution.png')

    # departure
    fig_departure = plt.figure(figsize=(20, 10))
    ax1_departure = fig_departure.add_subplot(1, 2, 1)
    ax2_departure = fig_departure.add_subplot(1, 2, 2)
    
    sns.scatterplot(data = pd.concat([y_d, pd.DataFrame({'Prediction_departure' : pred_d})], axis = 1), 
                    x = "label", y = "Prediction_departure" , ax = ax1_departure)
    sns.distplot(y_d, label = 'ADR', kde = False, ax = ax2_departure)
    sns.distplot(pd.DataFrame({'Prediction_departure' : pred_d}), label = 'Prediction', kde = False, ax = ax2_departure)
    ax1_departure.set(xlabel='Actual Departure Rate', ylabel='Departure Prediction')
    ax2_departure.set(xlabel='Departures per hour', ylabel='Count')
    plt.legend() 
    plt.savefig(save_dir + f'/departure_{hour}hour_distribution.png')




"""
X_train_a = train_data_arrival.drop('label', axis = 1).values
y_train_a = train_data_arrival['label'].values

train_pred_a = clf.predict(X_train_a)
train_rmse_a = np.sqrt(mean_squared_error(y_train_a, train_pred_a))
print(train_rmse_a)
"""



# *** training data에는 AAR, ADR 둘다 없어야 함
# 0부터 시작
# max capacity 선 추가
# 원래 demand에 대한 predict 점 추가




def max_capacity(example, hour, model):    # 50까지 늘림
    
    # load trained model
    clf_arrival = joblib.load(config.output + f'{model}_arrival_{hour}.bin') 
    clf_departure = joblib.load(config.output + f'{model}_departure_{hour}.bin') 

    # load data
    data_arrival_raw = pd.read_csv( os.path.join(config.input_dir, 'arrival_train.csv'), index_col = 0)
    data_arrival = data_arrival_raw[data_arrival_raw['taf'] == 6].reset_index(drop = True)   
    data_arrival = data_arrival.drop('taf', axis = 1)
    data_departure_raw = pd.read_csv( os.path.join(config.input_dir, 'departure_train.csv'), index_col = 0)
    data_departure = data_departure_raw[data_departure_raw['taf'] == 6].reset_index(drop = True)            # 맨 처음은 6시간 taf 쓰기
    data_departure = data_departure.drop('taf', axis = 1)

    # Data_raw : Data_#과 동일   ->  data_arrival / data_departure
    # Data_m : Data_# 에서 AAR, ADR 뺀 것  -> data_arrival.drop('label', axis = 1) / data_departure.drop('label', axis = 1)

    # extra
    demand = 80
    
    """ arrival """
    data_a = data_arrival.drop('label', axis = 1).to_numpy()[example:example+1]
    XX_a = np.zeros((1,len(data_a.T)))
    
    original_demand_a = int(data_a[0][0])
    for i in range(0,demand+1):
        XX_a = np.append(XX_a, data_a, axis = 0)
        XX_a[i,0] = XX_a[i,0] + i - original_demand_a
    
    XX_a[0] = XX_a[1]   
    XX_a[0,0] = 0   
    XXX_a = XX_a[0:demand+1]
    XXXX_a = np.arange(XXX_a[0,0], XXX_a[demand,0]+1, 1)
    YYYY_a = clf_arrival.predict(XXX_a[0:demand+1])
    max_aar = float(max(YYYY_a))
    actual_aar = int(data_arrival['label'][example:example+1])
    ead = int(data_arrival['EAD'][example:example+1])
    prediction_a = float(clf_arrival.predict(data_arrival.drop('label', axis = 1)[example:example+1]))
    

    """ departure """
    data_d = data_departure.drop('label', axis = 1).to_numpy()[example:example+1]
    XX_d = np.zeros((1,len(data_d.T)))
    
    original_demand_d = int(data_d[0][1])
    for i in range(0,demand+1):
        XX_d = np.append(XX_d, data_d, axis = 0)
        XX_d[i,1] = XX_d[i,1] + i - original_demand_d
        
    XX_d[0] = XX_d[1]           
    XX_d[0,1] = 0
    XXX_d = XX_d[0:demand+1]
    XXXX_d = np.arange(XXX_d[0,1], XXX_d[demand,1]+1, 1)
    YYYY_d = clf_departure.predict(XXX_d[0:demand+1])
    max_adr = float(max(YYYY_d))
    actual_adr = int(data_departure['label'][example:example+1])
    edd = int(data_departure['EDD'][example:example+1])
    prediction_d = float(clf_departure.predict(data_departure.drop('label', axis = 1)[example:example+1])) 


    """ capacity """
    # arrival과 departure을 수요의 비율로 늘렸을 때
    max_cap = int(max(data_arrival['label'] + data_departure['label']))
    max_capacity = np.zeros(demand+1)
    for i in range(0,demand*2+1,2):
        if round(i*data_a[0,0]/(data_a[0,0]+data_d[0,1])) >=75:
            capa_arr = YYYY_a[75]
        elif round(i*data_d[0,1]/(data_a[0,0]+data_d[0,1])) >=75:
            capa_dep = YYYY_d[75]
        else:
            capa_arr = YYYY_a[round(i*data_a[0,0]/(data_a[0,0]+data_d[0,1]))]
            capa_dep = YYYY_d[round(i*data_d[0,1]/(data_a[0,0]+data_d[0,1]))]
        max_capacity[int(i/2)] = capa_arr + capa_dep
    
    
    """ plot """
    plt.figure(figsize=(15,15))
    plt.title('Maximum Capacity', fontsize=30)
    plt.xlabel('Estimated Demands', fontsize=25)
    plt.ylabel('Capacity', fontsize=25)
    ax = plt.subplot()

    plt.plot(XXXX_a, YYYY_a, linewidth=4, label = 'Arrival')    # Arrival
    plt.plot(XXXX_d, YYYY_d, linewidth=4, label = 'Departure')    # Departure
    plt.plot(XXXX_a[:-5] + XXXX_d[:-5], max_capacity[:-5], linewidth=4, label = 'Total')    # Capacity - 안 예뻐서 뒤에 55개 자름
    plt.plot(XXXX_a[:-5]+XXXX_d[:-5], [max_cap]*(demand-4), linewidth=4, label = f'Empirical Maximum = {max_cap}')    # 데이터 상 max capacity
    
    plt.plot(data_a[0,0], prediction_a,'ob', markersize = 5)    # 원래 arrival 예측값
    plt.plot(data_d[0,1], prediction_d,'or', markersize = 5)    # 원래 departure 예측값
    plt.plot(data_a[0,0]+data_d[0,1], prediction_a + prediction_d,'yo', markersize = 8)    # 원래 capacity 예측값
    plt.legend(prop={'size': 20}, loc = 'upper left')

    # text
    capacity_text = f""" 
    * Predicted Max AAR: {max_aar:.1f}\n 
    Predicted AAR : {prediction_a:.1f}\n 
    Actual AAR : {actual_aar}\n 
    EAD : {ead}\n\n
    
    * Predicted Max ADR: {max_adr:.1f}\n
    Predicted ADR : {prediction_d:.1f}\n
    Actual ADR : {actual_adr}\n
    EDD : {edd}\n\n
    
    * Predicted Max Capacity: {max_aar + max_adr:.1f}\n
    Predicted Rate : {prediction_a + prediction_d:.1f}\n
    Actual Rate : {actual_aar + actual_adr}\n
    Demand : {ead + edd}\n
    """

    # 위치조정
    plt.text(1.1, 0.05, capacity_text,
             fontsize=15, style='italic', transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.05, 'pad': 6})
    
    plt.savefig(save_dir + f'/{hour}hour_maximum_capaicty.png', bbox_inches='tight', pad_inches=1)

    



"""
#############################################################################################################
#############################################################################################################
#############################################################################################################

#ngboost

def plot_result(prediction, start=0, end=10):   

    fig, ax = plt.subplots(figsize=(22, 10))
    plt.fill_between(prediction['Date'][start:end], prediction['95% Predictions_lower'][start:end],  prediction['95% Predictions_upper'][start:end], 
                     label = '95% Prediction Interval', color='gray', alpha=0.5)
    plt.plot(prediction['Date'][start:end], prediction['Predictions'][start:end], label = 'Predictions', lw=2)
    plt.scatter(prediction['Date'][start:end], prediction['Predictions'][start:end], lw=3)
    plt.scatter(prediction['Date'][start:end], prediction['Actual AAR'][start:end], label = 'Actual AAR', color='r', lw=3)

    ax.legend(fontsize = 15)
    #plt.title('Hourly Power Consumption Actual vs. Predicted Values with Prediction Intervals')
    plt.xlabel('Time')
    plt.ylabel('Arrivals per hour')
    plt.show()

plot_result(prediction, start = 8230, end = 8300)

"""




# today
today = date.today()
today = today.strftime("%b.%d.%Y")

# make folder
save_dir = config.result_dir + today
if not os.path.exists(save_dir):
    os.makedirs(save_dir)




if __name__ =='__main__':

    # make Argparser 
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--model', type = str)
    parser.add_argument('--hour', type = int)

    # save input to args
    args = parser.parse_args()

    # print arguments
    print('\nresult.py')
    print(f'model : {args.model}')
    print(f'Prediction hour : {args.hour} \n')
    
    # result
    if args.hour > 24:
        print('Exceeding 24h is not exist \n')
    else :
        result(
                 model = args.model,
                 hour = args.hour
                 )
    
    # max capacity
    print('Maximum Capacity')
    inst = input('Select instance : ')
    prediction_hour = input('Prediction hour : ')
    if int(prediction_hour) > 24:
        print('Exceeding 24h is not exist \n')
    else:
        max_capacity(example = int(inst), hour = int(prediction_hour), model = args.model, )