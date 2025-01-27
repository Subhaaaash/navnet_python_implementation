import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Permute, Multiply, Lambda, RepeatVector, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

imu_data = pd.read_csv('IMU_trajectory1.csv')
dvl_data = pd.read_csv('DVL_trajectory1.csv')

imu_features = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]', 'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
imu_data = imu_data[imu_features].values
scaler_imu = MinMaxScaler(feature_range=(0, 1))
imu_data_scaled = scaler_imu.fit_transform(imu_data)

dvl_features = ['DVL X [m/s]', 'DVL Y [m/s]', 'DVL Z [m/s]']
dvl_data = dvl_data[dvl_features].values
scaler_dvl = MinMaxScaler(feature_range=(0, 1))
dvl_data_scaled = scaler_dvl.fit_transform(dvl_data)

def create_sequences(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step_imu = 100 
time_step_dvl = 1   

X_imu, y_imu = create_sequences(imu_data_scaled, time_step_imu)
X_dvl, y_dvl = create_sequences(dvl_data_scaled, time_step_dvl)

def build_lstm_with_attention(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(units=64, return_sequences=True)(inputs)
    lstm_out = LSTM(units=64, return_sequences=True)(lstm_out)

    # Attention mechanism
    attention_weights = Dense(1, activation='tanh')(lstm_out)  
    attention_weights = Flatten()(attention_weights)           
    attention_weights = Dense(input_shape[0], activation='softmax')(attention_weights)  
    attention_weights = RepeatVector(64)(attention_weights)   
    attention_weights = Permute([2, 1])(attention_weights)    

    context_vector = Multiply()([lstm_out, attention_weights])  
    context_vector = Lambda(lambda x: K.sum(x, axis=1))(context_vector)  

    output = Dense(units=input_shape[1])(context_vector)  

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

model_imu = build_lstm_with_attention((X_imu.shape[1], X_imu.shape[2]))
model_imu.fit(X_imu, y_imu, epochs=10, batch_size=32, verbose=1)

model_dvl = build_lstm_with_attention((X_dvl.shape[1], X_dvl.shape[2]))
model_dvl.fit(X_dvl, y_dvl, epochs=10, batch_size=32, verbose=1)

pred_imu = model_imu.predict(X_imu)
pred_dvl = model_dvl.predict(X_dvl)

pred_imu_rescaled = scaler_imu.inverse_transform(pred_imu)
pred_dvl_rescaled = scaler_dvl.inverse_transform(pred_dvl)

actual_imu = scaler_imu.inverse_transform(y_imu)
actual_dvl = scaler_dvl.inverse_transform(y_dvl)

plt.figure(figsize=(14, 12))

# IMU predictions vs actual
for i, feature in enumerate(imu_features):
    plt.subplot(3, 2, i + 1)
    plt.plot(pred_imu_rescaled[:, i], label=f'Predicted {feature}', color='red')
    plt.plot(actual_imu[:, i], label=f'Actual {feature}', color='blue', linestyle='--')
    plt.title(f'IMU Data: Predicted vs Actual ({feature})')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))

# DVL predictions vs actual
for i, feature in enumerate(dvl_features):
    plt.subplot(3, 1, i + 1)
    plt.plot(pred_dvl_rescaled[:, i], label=f'Predicted {feature}', color='blue')
    plt.plot(actual_dvl[:, i], label=f'Actual {feature}', color='green', linestyle='--')
    plt.title(f'DVL Data: Predicted vs Actual ({feature})')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

plt.tight_layout()
plt.show()