# test data돌릴때 쓸 스크립트
# 현재 시간의 데이터를 입력하면 24시간 동안의 max capacity를 예측하도록

import os
import config
import pandas as pd
from datetime import datetime

time = datetime.strptime(input('Current Time [yyyy/mm/dd/HHMM] : '), '%Y/%m/%d/%H%M') or datetime.now()
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
wintemp = input('\nWind Aloft at 1000hpa, 925hpa, 850hpa \n[ 1000hpa WDWDWDWSWSWS - 925hpa WDWDWDWSWSWS - 850hpa WDWDWDWSWSWS ] \n : ')


# EAD	EDD	 year	month	day 	hour	DayName 	Arpt_cond	P_Airp	P_AAR	P_ADR	Arrival_remainder	Departure_remainder	
# WD_850	WD_925	WD_1000	WS_850	WS_925	WS_1000	WD	WSPD	WS_GST	VIS	WC	RN	CA_TOT	CLA_1LYR	BASE_1LYR	CLA_2LYR	BASE_2LYR	CLA_3LYR	BASE_3LYR	CLA_4LYR	BASE_4LYR	RVR	WDIR_t	WSPD_t	WG_t	VIS_t	WC_t	CLA_1LYR_t	BASE_1LYR_t	CLA_2LYR_t	BASE_2LYR_t	CLA_3LYR_t	BASE_3LYR_t

print(time)

time ->
arpt_cond ->
p_arpt -> 
arrival_remainder -> 
departure_remainder ->
METAR
TAF
wintemp




input_data = pd.DataFrame({'EAD' : int(EAD), 
                            'EDD' : int(EDD),
                            'year' : ddddd,
                            'month' : ,
                            'day' : ,
                            'hour' : ,
                            'DayName' : ,
                            'Arpt_cond ' : 

                            })