# test data돌릴때 쓸 스크립트
# 현재 시간의 데이터를 입력하면 24시간 동안의 max capacity를 예측하도록

import os
import config
import re
import pandas as pd
from datetime import datetime

time = datetime.strptime(input('Current Time [yyyymmdd HHMM] : '), '%Y%m%d %H%M') or datetime.now()
EAD = input('\nExpected Arrival Demands : ')
EDD = input('Expected Departure Demands : ')
previous_AAR = input("\nPrevious hour's Actual Arrival Rate : ")
previous_ADR = input("previous hour's Actual Departure Rate : ")
previous_EAD = input("Previous hour's Arrival Demands : ")
previous_EDD = input("Previous hour's Departure Demands : ")
airport_condition = input('\nAirport Condition [ VFR / MVFR / IFR / LIFR ] : ')
previous_airport_condition = input("Previous hour's Airport Condition [ VFR / MVFR / IFR / LIFR ] : ")
METAR = input('\nMETAR : ')
TAF = input('TAF : ')
wintemp = input('\nWind Aloft at 1000hpa, 925hpa, 850hpa \n[ (1000hpa)WDWDWDWSWSWS-(925hpa)WDWDWDWSWSWS-(850hpa)WDWDWDWSWSWS ] \n : ')



# time
year = time.year
month = time.month
day = time.day
hour = time.hour
dayname = time.strftime("%A")
dayname = (dayname.replace('Monday', '1').replace('Tuesday', '2').replace('Wednesday', '3')
           .replace('Thursday', '4').replace('Friday', '5').replace('Saturday', '6').replace('Sunday', '7'))

# airport condition
arpt_wx = {'VFR' : 1, 'MVFR' : 2, 'IFR' : 3, 'LIFR' : 4}    # VFR = 1 / MVFR = 2 / IFR = 3 / LIFR = 4
arpt_cond = arpt_wx[airport_condition]  
p_arpt =  arpt_wx[previous_airport_condition]  

# remainder
arrival_remainder = int(previous_EAD) - int(previous_AAR)
departure_remainder = int(previous_EDD) - int(previous_ADR)
if arrival_remainder < 0:
    arrival_remainder = 0
if departure_remainder < 0:
    departure_remainder = 0

#wintemp
WD_1000 = wintemp[0:3]
WS_1000 = wintemp[3:6]
WD_925 = wintemp[7:10]
WS_925 = wintemp[10:13]
WD_850 = wintemp[14:17]
WS_850 = wintemp[17:20]

#TAF
pattern_issue = '[0-9].....Z'
pattern_date = '[0-9]+/[0-9]{2,4}'
pattern_taf = 'TX........Z|TX.......Z'             # BECMG, TEMPO 전 text 찾는데 사용
pattern_wind = '\s[0-9]...[0-9][A-Z]'              # 단, 100KT이상은 잡을 수 X        
pattern_vis = '\s[0-9]..[0-9]\s'               
pattern_cavok = '\sC[A-Z][A-Z][A-Z][A-Z]\s'
pattern_wc = 'NSW|TS|DZ|RA|SN|SG|IC|PL|GR|GS|BR|FG|FU|VA|DU|SA|HZ|SQ|FC|SS|DS'
pattern_cloud = '[FSBO][A-Z][A-Z]..[0-9]'
pattern_chg = 'BECMG|TEMPO'
regex = re.compile(pattern_date)
search = regex.search(TAF)




#METAR





# EAD	EDD	 year	month	day 	hour	DayName 	Arpt_cond	P_Airp	P_AAR	P_ADR	Arrival_remainder	Departure_remainder	
# WD_850 WD_925	WD_1000	WS_850	WS_925	WS_1000	WD	WSPD	WS_GST	VIS	WC	RN	CA_TOT	CLA_1LYR	BASE_1LYR	CLA_2LYR	BASE_2LYR	CLA_3LYR	BASE_3LYR	CLA_4LYR	BASE_4LYR	RVR	WDIR_t	WSPD_t	WG_t	VIS_t	WC_t	CLA_1LYR_t	BASE_1LYR_t	CLA_2LYR_t	BASE_2LYR_t	CLA_3LYR_t	BASE_3LYR_t

input_data = {'EAD' : int(EAD), 
              'EDD' : int(EDD),
              'year' : year,
              'month' : month,
              'day' : day,
              'hour' : hour,
              'DayName' : dayname,
              'Arpt_cond ' : arpt_cond,
              'P_Airp' : p_arpt,
              'P_AAR' : int(previous_AAR),
              'P_ADR' : int(previous_ADR),
              'Arrival_remainder' : arrival_remainder,
              'Departure_remainder' : departure_remainder,
              'WD_850' : int(WD_850),
              'WD_925' : int(WD_925),
              'WD_1000' : int(WD_1000),
              'WS_850' : int(WS_850),
              'WS_925' : int(WS_925),
              'WS_1000' : int(WS_1000)
              }

df = pd.DataFrame(input_data)

print(df)