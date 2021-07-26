import numpy as np
import pandas as pd
import datetime
import warnings

print('It will take a few minutes...')

# options
pd.set_option('max_columns',100)
warnings.simplefilter('ignore')

# Load Data
Wx = pd.concat([pd.read_csv('.\\Weather\\RKSI_air_stcs2019%02d.csv' %(i)) for i in range(1,13)], ignore_index=True)
taf = pd.read_csv('.\\Weather\\TAF_data.csv') 
WINTEMP_Osan = pd.DataFrame(pd.read_csv('.\\WINTEMP\\UPPER_SONDE_47122_STD_2019_2019_2020.csv'))
#   types = {'WDIR': int, 'WSPD': int, 'WG' : int, 'VIS' : int, 'WC' : str, 'CLA_1LYR':str, 'BASE_1LYR':int, 'CLA_2LYR':str, 'BASE_2LYR':int, 'CLA_3LYR':str, 'BASE_3LYR':int}



"Time"
# 24h -> 00h conversion
def twentyfour_to_zero(date):                                             
    if date[8:10] == '24':
        return pd.to_datetime(date[:-2], format = '%Y%m%d') + pd.Timedelta(days=1)   # +1d 00h 
    else:
        return pd.to_datetime(date, format = '%Y%m%d%H')

Wx.TM = Wx.TM.astype(str)       # Wx.TM : int -> string
Wx.TM = Wx.TM.apply(twentyfour_to_zero)  

# Day name of the week 
DayName = Wx['TM'].dt.day_name()               
DayName = (DayName.replace('Monday', '1').replace('Tuesday', '2').replace('Wednesday', '3')
           .replace('Thursday', '4').replace('Friday', '5').replace('Saturday', '6').replace('Sunday', '7'))
DayName = pd.DataFrame({'DayName' : DayName})

# Divide time
year = Wx.TM.dt.year        # 항목별로 추출 -> Timestamp type X
month = Wx.TM.dt.month
day = Wx.TM.dt.day
hour = Wx.TM.dt.hour
Time = pd.DataFrame({'year' : year, 'month' : month, 'day' : day, 'hour' : hour})

# Time Concatenate
Time = Time.join(DayName)



"TAF"
# fill NaN
taf = taf.fillna({'WDIR': 0, 'WSPD': 0, 'WG' : 0, 'VIS' : 9999, 'WC' : 0, 
            'CLA_1LYR':0, 'BASE_1LYR':400, 'CLA_2LYR':0, 'BASE_2LYR':400, 'CLA_3LYR':0, 'BASE_3LYR':400})
taf = taf.rename(columns = {'Unnamed: 0' : 'Time'})
taf['Time'] = pd.to_datetime(taf['Time'])
taf['issue_time'] = pd.to_datetime(taf['issue_time'])

# Same criteria
taf = taf.replace('FEW', 2)
taf = taf.replace('SCT', 4)
taf = taf.replace('BKN', 7)
taf = taf.replace('OVC', 8)

taf = taf.replace('FC', 10)   #  10 - Tornado                   
taf = taf.replace('VA', 10)
taf = taf.replace('TS', 9)    #   9 - TS                      
taf = taf.replace('SQ', 8)    #   8 - Squall                  
taf = taf.replace('PL', 7)    #   7 - Snow                      
taf = taf.replace('SN', 7)
taf = taf.replace('SG', 7)
taf = taf.replace('GR', 6)    #   6 - Freezing~, Heavy rain     
taf = taf.replace('GS', 6)
taf = taf.replace('IC', 6)
taf = taf.replace('RA', 5)    #   5 - Rain                      
taf = taf.replace('FG', 4)    #   4 - Fog                       
taf = taf.replace('DZ', 3)    #   3 - Drizzle, Haze, Duststorm  
taf = taf.replace('BR', 3)
taf = taf.replace('DS', 3)
taf = taf.replace('SS', 3)
taf = taf.replace('HZ', 3)
taf = taf.replace('SA', 3)
taf = taf.replace('DU', 3)    #   2 - others                    
taf = taf.replace('FU', 3)    #   1 - Nothing                      

taf.BASE_1LYR = taf.BASE_1LYR.replace('0', 400)
taf.BASE_2LYR = taf.BASE_2LYR.replace('0', 400)
taf.BASE_3LYR = taf.BASE_3LYR.replace('0', 400)

# TAF Dataframe initialize
TAF_6 = pd.DataFrame([], columns = ['WDIR_t6', 'WSPD_t6', 'WG_t6', 'VIS_t6', 'WC_t6', 'CLA_1LYR_t6', 'BASE_1LYR_t6',
                                    'CLA_2LYR_t6', 'BASE_2LYR_t6','CLA_3LYR_t6', 'BASE_3LYR_t6',], index = Wx.TM)
TAF_12 = pd.DataFrame([], columns = ['WDIR_t12', 'WSPD_t12', 'WG_t12', 'VIS_t12', 'WC_t12', 'CLA_1LYR_t12', 'BASE_1LYR_t12',
                                     'CLA_2LYR_t12', 'BASE_2LYR_t12','CLA_3LYR_t12', 'BASE_3LYR_t12',], index = Wx.TM)
TAF_18 = pd.DataFrame([], columns = ['WDIR_t18', 'WSPD_t18', 'WG_t18', 'VIS_t18', 'WC_t18', 'CLA_1LYR_t18', 'BASE_1LYR_t18',
                                     'CLA_2LYR_t18', 'BASE_2LYR_t18','CLA_3LYR_t18', 'BASE_3LYR_t18',], index = Wx.TM)
TAF_24 = pd.DataFrame([], columns = ['WDIR_t24', 'WSPD_t24', 'WG_t24', 'VIS_t24', 'WC_t24', 'CLA_1LYR_t24', 'BASE_1LYR_t24',
                                     'CLA_2LYR_t24', 'BASE_2LYR_t24','CLA_3LYR_t24', 'BASE_3LYR_t24',], index = Wx.TM)

taf_sort = taf.sort_values('Time')

for i in range(len(Wx)):
    temp_date = datetime.datetime(2019,1,1,0,0) + datetime.timedelta(hours = i)   
    temp_taf_date = taf_sort[taf_sort['Time'] == temp_date].reset_index()  
    total_taf_eachdate = temp_taf_date.drop('index',1).drop('Time',1).sort_values('issue_time').reset_index().drop('index',1)

    # 6시간 전 예보
    taf_6 = total_taf_eachdate[(total_taf_eachdate['issue_time'] < temp_date) 
                               & (total_taf_eachdate['issue_time'] >= temp_date - datetime.timedelta(hours=6))][0:1].drop('issue_time',1)
    if taf_6.empty:
        TAF_6[temp_date:temp_date] = TAF_6[temp_date:temp_date]
    else:
        TAF_6[temp_date:temp_date] = taf_6

    # 12시간 전 예보
    taf_12 = total_taf_eachdate[(total_taf_eachdate['issue_time'] < temp_date - datetime.timedelta(hours=6)) 
                                & (total_taf_eachdate['issue_time'] >= temp_date - datetime.timedelta(hours=12))][0:1].drop('issue_time',1)
    if taf_12.empty:
        TAF_12[temp_date:temp_date] = TAF_12[temp_date:temp_date]
    else:
        TAF_12[temp_date:temp_date] = taf_12

    # 18시간 전 예보
    taf_18 = total_taf_eachdate[(total_taf_eachdate['issue_time'] < temp_date - datetime.timedelta(hours=12)) 
                                & (total_taf_eachdate['issue_time'] >= temp_date - datetime.timedelta(hours=18))][0:1].drop('issue_time',1)
    if taf_18.empty:
        TAF_18[temp_date:temp_date] = TAF_18[temp_date:temp_date]
    else:
        TAF_18[temp_date:temp_date] = taf_18

    # 24시간 전 예보
    taf_24 = total_taf_eachdate[(total_taf_eachdate['issue_time'] < temp_date - datetime.timedelta(hours=18)) 
                                & (total_taf_eachdate['issue_time'] >= temp_date - datetime.timedelta(hours=24))][0:1].drop('issue_time',1)
    if taf_24.empty:
        TAF_24[temp_date:temp_date] = TAF_24[temp_date:temp_date]
    else:
        TAF_24[temp_date:temp_date] = taf_24
               
TAF_6 = TAF_6.fillna({'WDIR_t6': 0, 'WSPD_t6': 0, 'WG_t6' : 0, 'VIS_t6' : 9999, 'WC_t6' : 0, 
                      'CLA_1LYR_t6':0, 'BASE_1LYR_t6':400, 'CLA_2LYR_t6':0, 'BASE_2LYR_t6':400,
                      'CLA_3LYR_t6':0, 'BASE_3LYR_t6':400})
TAF_12 = TAF_12.fillna({'WDIR_t12': 0, 'WSPD_t12': 0, 'WG_t12' : 0, 'VIS_t12' : 9999, 'WC_t12' : 0, 
                        'CLA_1LYR_t12':0, 'BASE_1LYR_t12':400, 'CLA_2LYR_t12':0, 'BASE_2LYR_t12':400,
                        'CLA_3LYR_t12':0, 'BASE_3LYR_t12':400})
TAF_18 = TAF_18.fillna({'WDIR_t18': 0, 'WSPD_t18': 0, 'WG_t18' : 0, 'VIS_t18' : 9999, 'WC_t18' : 0, 
                        'CLA_1LYR_t18':0, 'BASE_1LYR_t18':400, 'CLA_2LYR_t18':0, 'BASE_2LYR_t18':400,
                        'CLA_3LYR_t18':0, 'BASE_3LYR_t18':400})
TAF_24 = TAF_24.fillna({'WDIR_t24': 0, 'WSPD_t24': 0, 'WG_t24' : 0, 'VIS_t24' : 9999, 'WC_t24' : 0, 
                        'CLA_1LYR_t24':0, 'BASE_1LYR_t24':400, 'CLA_2LYR_t24':0, 'BASE_2LYR_t24':400,
                        'CLA_3LYR_t24':0, 'BASE_3LYR_t24':400})
    
TAF_6 = TAF_6.reset_index().drop('TM',1)
TAF_12 = TAF_12.reset_index().drop('TM',1)
TAF_18 = TAF_18.reset_index().drop('TM',1)
TAF_24 = TAF_24.reset_index().drop('TM',1)



"METAR"
# Present Weather code

#     <Wx> Present weather (code 4677)
#      00-49 : No Precipitation at the station at the time of the observation
#      00-19 : No precipitation, fog, ice fog (except for 11 and 12), duststorm, 
#              sandstorm, drifting or blowing snow at the station* at the time of 
#              observation or, except for 09 and 17, during the preceding hour
#      20-29 : Precipitation, fog, ice fog or thunderstorm at the station during
#              the preceding hour but not at the time of observation
#      30-39 : Duststorm, sandstorm, drifting or blowing snow
#      40-49 : Fog or ice fog at the time of observation
#      50-59 : Drizzle
#      60-69 : Rain
#      70-79 : Solid precipitation not in showers
#      80-99 : Showery precipitation, or precipitation with current 
#              or recent thunderstorm
#
#     Order of severity (Code 4677)
#      10 - Tornado                   (19)
#       9 - TS                        (13, 17, 91-99)
#       8 - Squall                    (18)
#       7 - Snow                      (70-78, 83-90) 
#       6 - Freezing~, Heavy rain     (56-57, 66-69, 79)
#       5 - Rain                      (58-59, 60-65, 80-82) 
#       4 - Fog                       (40-49)
#       3 - Drizzle, Haze, Duststorm  (5, 10, 30-39, 50-55)
#       2 - others                    (0-4, 6-9, 11-12, 14-16)
#       1 - Nothing                   (NaN)
#                                     (20번대는 제외)

def present_weather(weather):
    if weather == 19:
        weather = 10
    elif weather == 13 or weather == 17 or weather in range(91,100):
        weather = 9
    elif weather == 18:
        weather = 8
    elif weather in range(70, 79) or weather in range(83,91):
        weather = 7
    elif weather in range(56, 58) or weather in range(66,70) or weather == 79:
        weather = 6
    elif weather in range(58, 60) or weather in range(60,66) or weather in range(80-83):
        weather = 5
    elif weather in range(40, 50):
        weather = 4
    elif weather == 5 or weather == 10 or weather in range(30, 40) or weather in range(50,56):
        weather = 3
    elif weather in range(0,5) or weather in range(6,10) or weather in range(11, 13) or weather in range(14,17):
        weather = 2        
    else:
        weather = 1
    return weather

Wx['WC'] = Wx['WC'].apply(present_weather)

# Fill NA/NaN
Wx = Wx.fillna({'WS_GST' : 0, 'RVR1' : 1000, 'RVR2' : 1000, 'RVR3' : 1000, 'RN' : 0, 'BASE_1LYR' : 400, 'BASE_2LYR' : 400, 'BASE_3LYR' : 400,'BASE_4LYR' : 400,
                'CLA_1LYR' : 0,'CLA_2LYR' : 0,'CLA_3LYR' : 0,'CLA_4LYR' : 0})

# Mean RVR
RVR = Wx.loc[:,'RVR1':'RVR3'].min(axis=1)
RVR = pd.DataFrame({'RVR':RVR})
Wx = Wx.join(RVR)

# Drop useless column
Wx = Wx.drop('CLF_1LYR', axis=1)                        
Wx = Wx.drop('CLF_2LYR', axis=1) 
Wx = Wx.drop('CLF_3LYR', axis=1)                        
Wx = Wx.drop('CLF_4LYR', axis=1)
Wx = Wx.drop('RVR1', axis=1)                        
Wx = Wx.drop('RVR2', axis=1)                        
Wx = Wx.drop('RVR3', axis=1)
Wx = Wx.drop('RVR4', axis=1)                        
Wx = Wx.drop('TM', axis=1)      # datetime 버리기

METAR = Wx


"Airport Condition / Previous Airpot Condition"
# Ceiling
CIG = np.zeros((len(Wx),1))
for i in range(len(Wx)):
    if Wx.CLA_1LYR[i] >= 5:
        CIG[i] = Wx.BASE_1LYR[i]
    elif Wx.CLA_2LYR[i] >= 5:
        CIG[i] = Wx.BASE_2LYR[i]
    elif Wx.CLA_3LYR[i] >= 5:
        CIG[i] = Wx.BASE_3LYR[i]
    elif Wx.CLA_4LYR[i] >= 5:
        CIG[i] = Wx.BASE_4LYR[i]
    else: 
        CIG[i] = 400

# VFR or IFR
P_Airp = np.zeros((len(Wx),1))            # VFR = 1 / MVFR = 2 / IFR = 3 / LIFR = 4
for i in range(len(Wx)):
    if Wx.VIS[i] > 600 and CIG[i] > 30:
        P_Airp[i] = 1
    elif Wx.VIS[i] >= 480 and Wx.VIS[i] <= 600 or CIG[i] >= 10 and CIG[i] <= 30:
        P_Airp[i] = 2
    elif Wx.VIS[i] >= 160 and Wx.VIS[i] < 480 or CIG[i] >= 5 and CIG[i] < 30:
        P_Airp[i] = 3
    else :
        P_Airp[i] = 4

Arpt_cond = P_Airp.flatten()    # Arpt_cond = P_Airp.reshape(8760,)    flatten() : 2D array -> 1D array / (100,1) -> (100,)
Arpt_cond = pd.DataFrame({'Arpt_cond' : Arpt_cond})
P_Airp = P_Airp.flatten()       # P_Airp = P_Airp.reshape(8760,)
P_Airp = np.roll(P_Airp, 1)     # 맨 처음 data에는 0 넣음
P_Airp[0] = 1
P_Airp = pd.DataFrame({'P_Airp' : P_Airp})



"Actual Arrival Rate (AAR) / Estimate Arrival Demand (EAD)"
EAD = np.zeros(len(Wx))  
AAR = np.zeros(len(Wx))

start = '2019-01-01'
end = '2019-12-31'
j = 0

datelist = pd.date_range(start, end).astype(str).to_list()
for date in datelist:
    Arr_data = pd.read_csv(f'.\\FOIS\\Arrival_{date}.csv', index_col= 0)
    Arr_data['Arrival_SCH'] = date + Arr_data['Arrival_SCH']        # 해당 날짜 더하기
    Arr_data['Arrival_SCH'] = pd.to_datetime(Arr_data['Arrival_SCH'], format='%Y-%m-%d%H:%M')        # datetime으로 변환
    Arr_data['Arrival_ATA'] = date + Arr_data['Arrival_ATA']        # 해당 날짜 더하기
    Arr_data['Arrival_ATA'] = pd.to_datetime(Arr_data['Arrival_ATA'], format='%Y-%m-%d%H:%M')        # datetime으로 변환
    for i in range(24):
        EAD[j] = len(Arr_data[(Arr_data['Arrival_SCH'] >= datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i))
                               & (Arr_data['Arrival_SCH'] < datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i+1))])
        AAR[j] = len(Arr_data[(Arr_data['Arrival_ATA'] >= datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i))
                               & (Arr_data['Arrival_ATA'] < datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i+1))])
        j = j+1



"Actual Departure Rate (ADR) / Estimate Departure Demand (EDD)"
EDD = np.zeros(len(Wx))  
ADR = np.zeros(len(Wx))

start = '2019-01-01'
end = '2019-12-31'
j = 0

datelist = pd.date_range(start, end).astype(str).to_list()
for date in datelist:
    dep_data = pd.read_csv(f'.\\FOIS\\Departure_{date}.csv', index_col= 0)
    dep_data['Departure_SCH'] = date + dep_data['Departure_SCH']        # 해당 날짜 더하기
    dep_data['Departure_SCH'] = pd.to_datetime(dep_data['Departure_SCH'], format='%Y-%m-%d%H:%M')        # datetime으로 변환
    dep_data['Departure_ATD'] = date + dep_data['Departure_ATD']        # 해당 날짜 더하기
    dep_data['Departure_ATD'] = pd.to_datetime(dep_data['Departure_ATD'], format='%Y-%m-%d%H:%M')        # datetime으로 변환
    for i in range(24):
        EDD[j] = len(dep_data[(dep_data['Departure_SCH'] >= datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i))
                               & (dep_data['Departure_SCH'] < datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i+1))])
        ADR[j] = len(dep_data[(dep_data['Departure_ATD'] >= datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i))
                               & (dep_data['Departure_ATD'] < datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(hours = i+1))])
        j = j+1



"Previous hour's AAR / ADR"
# Previous AAR
P_AAR = np.roll(AAR,1)                  # 맨 처음 data에는 0 넣음
P_AAR[0] = 0
P_AAR = pd.DataFrame({'P_AAR' : P_AAR})

# Previous ADR
P_ADR = np.roll(ADR,1)
P_ADR[0] = 0
P_ADR = pd.DataFrame({'P_ADR' : P_ADR})



"Wind/Temp"
WINTEMP_Osan['Time'] = pd.to_datetime(WINTEMP_Osan['Time'], format='%d/%m/%Y %H:%M', infer_datetime_format=True)
WINTEMP_Osan_app_alt = WINTEMP_Osan[WINTEMP_Osan['Pressure'] > 300]
WINTEMP_Osan_app_alt = WINTEMP_Osan_app_alt.fillna(0)

WINTEMP = pd.DataFrame()

for i in range(0,6765,6):
    wintemp = WINTEMP_Osan_app_alt[i:i+6]
    data = pd.DataFrame(pd.concat([wintemp['WD'],wintemp['WS']], axis = 0)).reset_index().drop('index', axis=1).T
    WINTEMP = WINTEMP.append(pd.concat([data]*6))

for i in range(6765,len(WINTEMP_Osan_app_alt),6):   # 10월 10일 부터 2번씩 관측함
    wintemp = WINTEMP_Osan_app_alt[i:i+6]
    data = pd.DataFrame(pd.concat([wintemp['WD'],wintemp['WS']], axis = 0)).reset_index().drop('index', axis=1).T
    WINTEMP = WINTEMP.append(pd.concat([data]*12))
WINTEMP = WINTEMP.reset_index(drop=True)
WINTEMP = WINTEMP.rename(columns ={0:'WD_400',1:'WD_500',2:'WD_700',3:'WD_850',4:'WD_925',5:'WD_1000',
                                   6:'WS_400',7:'WS_500',8:'WS_700',9:'WS_850',10:'WS_925',11:'WS_1000' } )

WINTEMP = WINTEMP.fillna(0)



"Remainder"
# remainder = previos hour's demand - previous hour's actual AAR 
arrival_remainder = EAD - AAR
arrival_remainder = np.roll(arrival_remainder,1)  
arrival_remainder[0] = 0
arrival_remainder = pd.DataFrame({'Arrival_remainder' : arrival_remainder})
departure_remainder = EDD - ADR
departure_remainder = np.roll(departure_remainder,1)  
departure_remainder[0] = 0
departure_remainder = pd.DataFrame({'Departure_remainder' : departure_remainder})


# No. of cancellation
fois_arrival = pd.read_excel('.\\FOIS\\RKSI_19_20_arrival.xlsx')
fois_departure = pd.read_excel('.\\FOIS\\RKSI_19_20_departure.xlsx')

fois_arrival['Time'] = pd.to_datetime(fois_arrival['STA_DATE'].astype(str) + fois_arrival['STA'].astype(str).str.zfill(4), 
                                      format = '%Y%m%d%H%M') #STA를 4자리로 만들고 합친 뒤, datetime으로 바꿈
fois_arrival = fois_arrival.drop('STA_DATE', axis=1)
fois_arrival = fois_arrival.drop('STA', axis=1)

fois_departure['Time'] = pd.to_datetime(fois_departure['SCH_DATE'].astype(str) + fois_departure['SCH_TIME'].astype(str).str.zfill(4), 
                                        format = '%Y%m%d%H%M') #STA를 4자리로 만들고 합친 뒤, datetime으로 바꿈
fois_departure = fois_departure.drop('SCH_DATE', axis=1)
fois_departure = fois_departure.drop('SCH_TIME', axis=1)
#   fois_departure['DEP_STATUS'].value_counts()   # DEP, DLA, CNL, DIV
#   fois_arrival['ARR_STATUS'].value_counts()     # DEP, DLA, CNL, DIV, LND

cnl_a = np.zeros(len(Wx))
cnl_d = np.zeros(len(Wx))

for i in range(len(Wx)):
    time = datetime.datetime(2019,1,1) + datetime.timedelta(hours = i+1)
    # 해당되는 시간으로 나눔
    temp_a = fois_arrival[(fois_arrival['Time'] < time) & (fois_arrival['Time'] >= time-datetime.timedelta(hours=1))] 
    temp_d = fois_departure[(fois_departure['Time'] < time) & (fois_departure['Time'] >= time-datetime.timedelta(hours=1))]
    # 해당 시간에서 캔슬 수 카운트
    cnl_a[i] = sum((temp_a['ARR_STATUS'] == 'CNL')|(temp_a['ARR_STATUS'] == 'DIV'))      
    cnl_d[i] = sum((temp_d['DEP_STATUS'] == 'CNL'))


# subtract No. of cancellation
arrival_remainder['Arrival_remainder'] = arrival_remainder['Arrival_remainder'] - cnl_a
departure_remainder['Departure_remainder'] = departure_remainder['Departure_remainder'] - cnl_d


# Non negative values
arrival_remainder[arrival_remainder < 0] = 0
departure_remainder[departure_remainder < 0] = 0





###############################################################################################################################################

# Data Concatenate

###############################################################################################################################################

# Data Concatenate
Data = pd.DataFrame({'AAR':AAR, 'EAD':EAD, 'ADR':ADR, 'EDD':EDD})
Data = Data.join(Time)
Data = Data.join(Arpt_cond)
Data = Data.join(P_Airp)
Data = Data.join(P_AAR)
Data = Data.join(P_ADR)
Data = Data.join(arrival_remainder)
Data = Data.join(departure_remainder)
Data = Data.join(WINTEMP)
Data = Data.join(METAR)
Data = Data.join(TAF_6)
Data = Data.join(TAF_12)
Data = Data.join(TAF_18)
Data = Data.join(TAF_24)

print(Data.head())

save_csv = input("\n\n Save to 'data_raw.csv' ?  [y/n]  :  "  )
if save_csv == 'y':
    Data.to_csv('.\\data_raw.csv')
    print("\n'data_raw.csv' saved\n")
else:
    print('\nSave permission denied\n')
