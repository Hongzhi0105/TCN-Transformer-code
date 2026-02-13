# config.py

OrigData_path = '../CMAQ_Oringin_Data'
Train_Data_path = './CMAQ_Train_Data'
Trained_Model_path = './CMAQ_Model'

HIST_DIR = './HIST_DIR'   #PM2.5_obs歷史數據資料夾
year_of_data = 2016       #定義目前數據資料年分，以此判定平閏年，設定暫定time的數據長度

#####################################################################
#         You can change variables that under this hint             #
#                        Auguments Cross test                       #
#####################################################################

# Please set only one site.
site_sets = [42]

# Input_timestep (past)
Input_timesteps = 72

# Output_timestep (future N)
Output_timesteps = 72  

# -------------------------------
# Shifted Mode Feature Setting
# -------------------------------
# Past segment: t-72 ~ t-1 (可含 obs 與 sim；你目前只放 obs)
X_PAST_COLS = ['PM2.5_obs','PM2.5', 'O3', 'SO2', 'NO2', 'NOx',
    'TEMP2_JD', 'WSPD10', 'WDIR10', 'PRSFC', 'PBL', 'QV', 'CFRAC']

#X_PAST_COLS = ['PM2.5_obs']
# Future segment: t0 ~ t+N-1 (只放 sim，不能含 *_obs)
X_FUTURE_COLS = [
    'PM2.5', 'O3', 'SO2', 'NO2', 'NOx',
    'TEMP2_JD', 'WSPD10', 'WDIR10', 'PRSFC', 'PBL', 'QV', 'CFRAC'
]

# Union channels (固定順序，避免權重不可用)
X_UNION_COLS = X_PAST_COLS + X_FUTURE_COLS

# Model input dimension must match union cols
INPUT_DIM = len(X_UNION_COLS)

# Target column
Y_COL = 'obs_diff'
Need_element_Y = [Y_COL]

# 用來從 txt 撈資料的 cls 清單（至少要包含 past + future 欄位）
Need_element_X = sorted(list(set(X_PAST_COLS + X_FUTURE_COLS)))

# 你原本的 OUTPUT_DIM 這裡不再固定，用 Output_timesteps 決定
OUTPUT_DIM = Output_timesteps

# -------------------------------
# For Log Name (keep)
# -------------------------------
LossFunction = "RMSE"
ActivateFunction = "Relu"

# Train / Valid
Train_ratio = 8
Valid_ratio = 2

# Train Month / Predict Month（目前主要只用於命名，不做過濾）
Train_month = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
predict_month = [1]

#####################################################################
#               Other variable Save Name                            #
#####################################################################

_fileName_sets = ["201601d4_sim.met", "201601d4_sim.met2", "201601d4_sim_obs"]

# In dataset.py
train_data_name = f'{Train_Data_path}/{site_sets[0]}_training_data.pt'

LogName = f'''{site_sets[0]}_{Input_timesteps}_{Output_timesteps}_{LossFunction}_{ActivateFunction}_{Train_ratio}_{Valid_ratio}_predict{predict_month[0]}_{Need_element_Y[0]}'''

best_model_path = f'{Trained_Model_path}/{LogName}_best_model.pth'
pre_tru = f'{Trained_Model_path}/{LogName}_pre_tru_{Need_element_Y[0]}'
