import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
import xgboost as xgb
import numpy as np

# Carregar o dataset
df = pd.read_csv('./datasets/casas_ponta_grossa.csv', delimiter=',')

# Verificando a presença de NaNs
print("Quantidade de valores ausentes:")
print(df.isna().sum())

# Retirando valores NaN:
df = df[df['Quartos'].notna()]
df = df[df['Banheiros'].notna()]
df = df[df['Suítes'].notna()]
df = df[df['Vagas de Garagem'].notna()]
df = df[df['Bairro'].notna()]
df = df[df['Área Total'].notna()]
dfFiltrado = df[df['Valor Venda'].notna()]

# Filtrando as casas com valor de venda entre 200k e 1 milhão
dfFiltrado = dfFiltrado[(dfFiltrado['Valor Venda'] >= 200000) & (dfFiltrado['Valor Venda'] <= 1000000)]

# Verificando as estatísticas descritivas para detectar valores extremos
print("\nEstatísticas Descritivas:")
print(dfFiltrado.describe())

# Plotando gráficos para verificar as distribuições dos dados

# Histograma de 'Valor Venda' e 'Área Total'
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(dfFiltrado['Valor Venda'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribuição do Valor de Venda')
plt.xlabel('Valor de Venda')
plt.ylabel('Frequência')

plt.subplot(1, 2, 2)
plt.hist(dfFiltrado['Área Total'], bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribuição da Área Total')
plt.xlabel('Área Total')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

# Gráfico de dispersão entre 'Área Total' e 'Valor Venda'
plt.figure(figsize=(8, 6))
plt.scatter(dfFiltrado['Área Total'], dfFiltrado['Valor Venda'], alpha=0.5, color='orange')
plt.title('Dispersão entre Área Total e Valor de Venda')
plt.xlabel('Área Total')
plt.ylabel('Valor de Venda')
plt.show()

# Verificando a correlação entre variáveis numéricas
correlation_matrix = dfFiltrado[['Área Total', 'Quartos', 'Banheiros', 'Suítes', 'Vagas de Garagem', 'Valor Venda']].corr()
print("\nMatriz de Correlação:")
print(correlation_matrix)

# Heatmap de correlação
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(np.arange(correlation_matrix.shape[1]), correlation_matrix.columns, rotation=45)
plt.yticks(np.arange(correlation_matrix.shape[0]), correlation_matrix.index)
plt.title('Matriz de Correlação das Variáveis')
plt.tight_layout()
plt.show()

# Engenharia de Features
# Transformando a coluna 'Bairro' com LabelEncoder
le = LabelEncoder()
dfFiltrado['Bairro'] = le.fit_transform(dfFiltrado['Bairro'])

# Normalizando as variáveis numéricas
scaler = QuantileTransformer(output_distribution='normal')  # Usando QuantileTransformer para robustez
dfFiltrado[['Área Total', 'Quartos', 'Banheiros', 'Suítes', 'Vagas de Garagem']] = scaler.fit_transform(
    dfFiltrado[['Área Total', 'Quartos', 'Banheiros', 'Suítes', 'Vagas de Garagem']]
)

# Criando novas colunas
dfFiltrado['Preco por m2'] = dfFiltrado['Valor Venda'] / dfFiltrado['Área Total']
dfFiltrado['Faixa de Preço'] = pd.cut(dfFiltrado['Valor Venda'], bins=[1000000, 2000000, 3500000, 5000000], labels=['Baixo', 'Médio', 'Alto'])
dfFiltrado['Quartos por m2'] = dfFiltrado['Quartos'] / dfFiltrado['Área Total']
dfFiltrado['Banheiros por m2'] = dfFiltrado['Banheiros'] / dfFiltrado['Área Total']
dfFiltrado['Vagas de Garagem por m2'] = dfFiltrado['Vagas de Garagem'] / dfFiltrado['Área Total']

# Novas colunas sugeridas
dfFiltrado['Área Total * Quartos'] = dfFiltrado['Área Total'] * dfFiltrado['Quartos']
dfFiltrado['Área Total * Banheiros'] = dfFiltrado['Área Total'] * dfFiltrado['Banheiros']
dfFiltrado['Diferença Preço - Média Bairro'] = dfFiltrado['Valor Venda'] - dfFiltrado.groupby('Bairro')['Valor Venda'].transform('mean')
dfFiltrado['Bairro por Faixa de Preço'] = dfFiltrado['Bairro'] * dfFiltrado['Faixa de Preço'].cat.codes
dfFiltrado['Área por Quartos'] = dfFiltrado['Área Total'] / dfFiltrado['Quartos']
dfFiltrado['Área por Banheiros'] = dfFiltrado['Área Total'] / dfFiltrado['Banheiros']

# Convertendo a coluna 'Faixa de Preço' para valores numéricos (inteiros)
dfFiltrado['Faixa de Preço'] = dfFiltrado['Faixa de Preço'].cat.codes

# Definindo as features (X) e o alvo (y)
X = dfFiltrado[['Quartos', 'Banheiros', 'Suítes', 'Vagas de Garagem', 'Bairro', 'Área Total', 'Preco por m2', 'Faixa de Preço', 
                'Quartos por m2', 'Banheiros por m2', 'Vagas de Garagem por m2', 'Área Total * Quartos', 
                'Área Total * Banheiros', 'Diferença Preço - Média Bairro', 'Bairro por Faixa de Preço', 
                'Área por Quartos', 'Área por Banheiros']]
y = dfFiltrado['Valor Venda']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tuning de Hiperparâmetros com RandomizedSearchCV para o modelo XGBoost
param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10, cv=5, verbose=1, scoring='neg_mean_absolute_error', random_state=42)
xgb_random_search.fit(X_train, y_train)

# Melhor modelo XGBoost
best_xgb_model = xgb_random_search.best_estimator_

# Previsões e avaliação para XGBoost
y_pred_xgb = best_xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"XGBoost - RMSE: {rmse_xgb}")
print(f"XGBoost - MAE: {mae_xgb}")

# Visualizando a importância das features com XGBoost
xgb.plot_importance(best_xgb_model)
plt.title('Importância das Features - XGBoost')
plt.tight_layout()
plt.show()
