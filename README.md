
# TMA_capacity

<br>

![Input_data](https://user-images.githubusercontent.com/85796140/126852091-31d6562d-7469-41b6-958d-464f91aaa36e.png)

<br>
<br>

data <br>
* 필요한 경우 각각 데이터(TAF 등)에서 csv파일 로 만듦 <br>
* data_Concatenate,ipynb에서 모든 데이터를 하나의 Dataframe으로 만들고 'data_raw.csv' 로 저장 (8760 rows × 92 columns) <br>
* data_engineering.ipynb에서 EDA 후, 'data.csv', 'data_6.csv', 'data_12.csv', 'data.csv_18', 'data_24.csv' 저장 <br>
<br>

input <br>
* data_engineering.ipynb에서 training, test data set으로 나누어 저장 <br>
* P_AAR, P_ADR은 원래 Previous AAR/ADR로 직전 시간의 AAR/ADR을 가져옴  ->  2시간 이상의 시간 예측 시, Previous AAR/ADR에서 직전 시간 Predicted previous AAR/ADR로 바뀜
* Remainder는 원래 직전 시간 "수요 - 실제 AAR/ADR - 캔슬된 비행 수 (+ 0이하 값은 0으로 바꿈)"  ->  2시간 이상의 시간 예측 시, "수요 - Predicted AAR/ADR" 로 바뀜
<br>

models -> github tracking X -> models 폴더 하나 만들기<br>
* 학습된 model 저장 <br>
* lgbr_pval_a.csv, lgbr_pval_d.csv, lgbr_predict_a.csv,lgbr_predict_d.csv 로 CV data 저장 -> stacking <br>
<br>

notebooks <br>
* Jupyter notebook 저장 <br>
<br>

result <br>
* 결과 저장 <br>
<br>

src <br>
* .py 저장 <br>
* model 추가 : hyperparameters.py, models.py, train.py에 각각 추가하기
<br>
