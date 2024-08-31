import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./datasets/tabela-fipe-historico-precos.csv', delimiter=',')

# 2. Converter 'codigoFipe' para tipo numérico e remover valores nulos
df['codigoFipe'] = pd.to_numeric(df['codigoFipe'], errors='coerce')

# 3. Codificação de variáveis categóricas
label_encoders = {}
for column in ['marca', 'modelo', 'mesReferencia']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 4. Separar as features (X) e o target (y)
X = df[['codigoFipe', 'marca', 'modelo', 'anoModelo', 'mesReferencia', 'anoReferencia']]
y = df['valor']

# Normalização do alvo (y) para o intervalo de 0 a 10
scaler_y = MinMaxScaler(feature_range=(0, 10))
y_normalizado = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 5. Dividir o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_normalizado, test_size=0.2, random_state=42)

# 6. Treinamento do modelo com suporte a variáveis categóricas
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 7. Avaliação do modelo
y_pred_normalizado = model.predict(X_test)

# Desnormalizar o resultado para avaliar o modelo
y_test_desnormalizado = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_desnormalizado = scaler_y.inverse_transform(y_pred_normalizado.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_test_desnormalizado, y_pred_desnormalizado)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# 8. Exemplo de predição
def safe_label_transform(label_encoder, value):
    try:
        return label_encoder.transform([value])[0]
    except ValueError:
        return 0  # Exemplo: assume '0' como um valor padrão

exemplo = [[123456, 
            safe_label_transform(label_encoders['marca'], 'Chevrolet'),
            safe_label_transform(label_encoders['modelo'], 'Onix'),
            2024, 
            safe_label_transform(label_encoders['mesReferencia'], 'Agosto'),
            2024]]

# Predição do valor normalizado e desnormalização
predicao_normalizada = model.predict(exemplo)
predicao = scaler_y.inverse_transform(predicao_normalizada.reshape(-1, 1)).flatten()
print(f'Predição do valor: {predicao[0]}')
