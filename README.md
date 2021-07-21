# TMA_capacity

<b> Workflow </b>
<br>
<br>

<data>
* 필요한 경우 각각 데이터(TAF 등)에서 csv파일 로 만듦 <br>
* Data_Concatenate에서 모든 데이터를 하나의 Dataframe으로 만들고 'data_raw.csv' 로 저장 (8760 rows × 92 columns) <br>
<br>

<eda>
* data_engineering에서 필요한 데이터만 두고 data.csv로 저장 - 현재 아무 내용도 없음!

![Input_data](https://user-images.githubusercontent.com/85796140/125006955-4577da00-e09a-11eb-8fe5-5e415e4441b1.png)
<br>

<train>
* lgbr로 학습하고, 분석
* lgbr_a.pkl, lgbr_d.pkl 로 model 저장
* lgbr_pval_a.csv, lgbr_pval_d.csv, lgbr_predict_a.csv,lgbr_predict_d.csv 로 CV data 저장 -> stacking
<br>

<evaluate>
* evaluate
