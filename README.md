# TMA_capacity

<b>Workflow<b>



<data 폴더>
* 필요한 경우 각각 데이터(TAF 등)에서 csv파일 로 만듦
* Data_Concatenate에서 모든 데이터를 하나의 Dataframe으로 만들고 Data_raw.csv로 저장

<eda 폴더>
* data_engineering에서 필요한 데이터만 두고 Data.csv로 저장 - 현재 생략(data폴더에서 바로 Data_csv 생성)

<train 폴더>
* lgbr로 학습하고, 분석
* lgbr_a.pkl, lgbr_d.pkl 로 저장

<test 폴더>
* test
