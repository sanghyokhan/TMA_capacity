import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Feature Importance
imp = pd.DataFrame({'feature': Data_m.columns, 'importance': reg_arrival.feature_importances_})
imp = imp.sort_values('importance').set_index('feature')
imp.plot(kind='barh', figsize = (20,20))
plt.legend(loc='lower right')



#############################################################################################################
#############################################################################################################
#############################################################################################################



fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


sns.scatterplot(data = pred_test_d, x = "ADR", y = "Prediction_departure" , ax = ax1)
sns.distplot(pred_test_d["ADR"], label = 'ADR', kde = False, ax = ax2)
sns.distplot(pred_test_d["Prediction_departure"], label = 'Prediction', kde = False, ax = ax2)

ax1.set(xlabel='Actual Departure Rate', ylabel='Departure Prediction')
ax2.set(xlabel='Departures per hour', ylabel='Count')
#ax1.set_title('aaa')
#ax2.set_title('bbb')

plt.legend()
plt.show()



#############################################################################################################
#############################################################################################################
#############################################################################################################



# *** training data에는 AAR, ADR 둘다 없어야 함
# 0부터 시작
# max capacity 선 추가
# 원래 demand에 대한 predict 점 추가

def max_capacity(Data_raw, Data_m, example):    # 50까지 늘림
    
    # extra
    demand = 80
    

    # arrival
    data_a = Data_m.to_numpy()[example:example+1]
    XX_a = np.zeros((1,len(data_a.T)))
    
    original_demand_a = int(data_a[0][0])
    for i in range(0,demand+1):
        XX_a = np.append(XX_a, data_a, axis = 0)
        XX_a[i,0] = XX_a[i,0] + i - original_demand_a
    
    XX_a[0] = XX_a[1]   
    XX_a[0,0] = 0   
    XXX_a = XX_a[0:demand+1]
    XXXX_a = np.arange(XXX_a[0,0], XXX_a[demand,0]+1, 1)
    YYYY_a = reg_arrival.predict(XXX_a[0:demand+1])
    max_aar = float(max(YYYY_a))
    actual_aar = int(Data_raw.drop('ADR', axis=1)['AAR'][example:example+1])
    ead = int(Data_raw.drop('ADR', axis=1)['EAD'][example:example+1])
    prediction_a = float(reg_arrival.predict(Data_m[example:example+1]))
    
    
    
    # departure
    data_d = Data_m.to_numpy()[example:example+1]
    XX_d = np.zeros((1,len(data_d.T)))
    
    original_demand_d = int(data_d[0][1])
    for i in range(0,demand+1):
        XX_d = np.append(XX_d, data_d, axis = 0)
        XX_d[i,1] = XX_d[i,1] + i - original_demand_d
        
    XX_d[0] = XX_d[1]           
    XX_d[0,1] = 0
    XXX_d = XX_d[0:demand+1]
    XXXX_d = np.arange(XXX_d[0,1], XXX_d[demand,1]+1, 1)
    YYYY_d = reg_departure.predict(XXX_d[0:demand+1])
    max_adr = float(max(YYYY_d))
    actual_adr = int(Data_raw.drop('AAR', axis=1)['ADR'][example:example+1])
    edd = int(Data_raw.drop('AAR', axis=1)['EDD'][example:example+1])
    prediction_d = float(reg_departure.predict(Data_m[example:example+1])) 

    
    
    # capacity - arrival과 departure을 수요의 비율로 늘렸을 때
    max_cap = int(max(Data_raw['AAR'] + Data_raw['ADR']))
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
    
    
    
    # plot
    plt.figure(figsize=(15,15))
    plt.title('Maximum Capacity', fontsize=30)
    plt.xlabel('Estimated Demands', fontsize=25)
    plt.ylabel('Capacity', fontsize=25)
    ax = plt.subplot()

    plt.plot(XXXX_a, YYYY_a, label = 'Arrival')    # Arrival
    plt.plot(XXXX_d, YYYY_d, label = 'Departure')    # Departure
    plt.plot(XXXX_a[:-5] + XXXX_d[:-5], max_capacity[:-5], label = 'Total')    # Capacity - 안 예뻐서 뒤에 55개 자름
    plt.plot(XXXX_a[:-5]+XXXX_d[:-5], [max_cap]*(demand-4), label = f'Empirical Maximum = {max_cap}')    # 데이터 상 max capacity
    
    plt.plot(data_a[0,0], prediction_a,'ob', markersize = 10)    # 원래 arrival 예측값
    plt.plot(data_d[0,1], prediction_d,'or', markersize = 10)    # 원래 departure 예측값
    plt.plot(data_a[0,0]+data_d[0,1], prediction_a + prediction_d,'yo', markersize = 15)    # 원래 capacity 예측값
    plt.legend(prop={'size': 20}, loc = 'upper left')

    plt.text(0.95, 0.04,    # 위치조정
             f"""
             * Predicted Max AAR: {max_aar:.1f}\n
               Predicted AAR : {prediction_a:.1f}\n
               Actual AAR : {actual_aar} \n               
               EAD : {ead} \n\n
             
             * Predicted Max ADR: {max_adr:.1f}\n
               Predicted ADR : {prediction_d:.1f}\n
               Actual ADR : {actual_adr} \n               
               EDD : {edd} \n\n
             
             * Predicted Max Capacity: {max_aar + max_adr:.1f}\n
               Predicted Rate : {prediction_a + prediction_d:.1f}\n
               Actual Rate : {actual_aar + actual_adr} \n               
               Demand : {ead + edd}'
             """,
             fontsize=15, style='italic', transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0, 'pad': 5})
    
    plt.show()   
    
    return Data_raw[example:example+1]   #result_


# 원하는 시간의 Max Capacity 그래프
max_capacity(Data_raw, Data_m, 154)     # 숫자에 원하는 Data index(0-8760) 넣기



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