import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np

# Carregar o dataset
df = pd.read_csv('./datasets/casas_ponta_grossa.csv', delimiter=',')

# Retirando valores NaN:
df = df[df['Quartos'].notna()]
df = df[df['Banheiros'].notna()]
df = df[df['Suítes'].notna()]
df = df[df['Vagas de Garagem'].notna()]
df = df[df['Bairro'].notna()]
df = df[df['Área Total'].notna()]
dfFiltrado = df[df['Valor Venda'].notna()]

Qtdquartos = dfFiltrado['Quartos'].value_counts()

Qtdbanheiros = dfFiltrado['Banheiros'].value_counts()

QtdSuítes = dfFiltrado['Suítes'].value_counts()

Qtdvagas_de_garagem = dfFiltrado['Vagas de Garagem'].value_counts()

Qtdbairro = dfFiltrado['Bairro'].value_counts()

##############
QtdArea_Total = dfFiltrado['Área Total'].value_counts()
# Definindo as bins e as labels
bins1 = [0, 100, 200, 400, 600, 800, 1100]  # Bins em metros quadrados
labels1 = ['0m-100m', '100m-200m', '200m-400m', '400m-600m', '600m-800m', '800m-1100m']  # Labels apropriados
# Criando a coluna de categorias
dfFiltrado['Categoria1'] = pd.cut(dfFiltrado['Área Total'], bins=bins1, labels=labels1, right=False)
# Contando a frequência em cada categoria
QtdArea_Total = dfFiltrado['Categoria1'].value_counts()
##############

##############
QtdValorVenda = dfFiltrado['Valor Venda'].value_counts()
# Definindo as bins
bins2 = [0, 200000, 400000, 600000, 800000, 1000000]
labels2 = ['<200k', '200k-400k', '400k-600k', '600k-800k', '800k-1M']
# Criando a coluna de categorias
dfFiltrado['Categoria2'] = pd.cut(dfFiltrado['Valor Venda'], bins=bins2, labels=labels2)
# Contando a frequência em cada categoria
QtdValorVenda = dfFiltrado['Categoria2'].value_counts()
##############


# Gráfico 1: Quartos
plt.figure(figsize=(10,6)) 
Qtdquartos.plot(kind='bar', alpha=0.7, color='blue')  
plt.title('Contagem de Valores de Quartos', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 2: Banheiros
plt.figure(figsize=(10,6)) 
Qtdbanheiros.plot(kind='bar', alpha=0.7, color='blue')  
plt.title('Contagem de Valores de Banheiros', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 3: Suítes
plt.figure(figsize=(10,6)) 
QtdSuítes.plot(kind='bar', alpha=0.7, color='blue') 
plt.title('Contagem de Valores de Suítes', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 4: Vagas de Garagem
plt.figure(figsize=(10,6)) 
Qtdvagas_de_garagem.plot(kind='bar', alpha=0.7, color='blue')
plt.title('Contagem de Valores de Vagas de Garagem', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 5: Bairro
plt.figure(figsize=(10,6)) 
Qtdbairro.plot(kind='bar',alpha=0.7, color='blue') 
plt.title('Contagem de Valores de Bairro', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 6: Plotando o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(QtdArea_Total.index, QtdArea_Total.values, alpha=0.7, color='blue')
plt.title('Frequência de Área Total por Categoria', fontsize=16)
plt.xlabel('Categoria de Área Total (m²)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
plt.tight_layout()

# Gráfico 7: Criando gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(QtdValorVenda.index, QtdValorVenda.values, alpha=0.7, color='blue')
plt.title('Contagem de Valores de Venda por Categoria', fontsize=16)
plt.xlabel('Categoria de Venda', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
plt.tight_layout()

# Suponha que 'df' seja o DataFrame com os dados carregados
X = dfFiltrado[['Quartos', 'Banheiros', 'Suítes', 'Vagas de Garagem', 'Área Total']]
y = dfFiltrado['Valor Venda']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliação usando RMSE
# RSME preve o quanto as previsões estão variando do valor real em reais.
rmse = np.sqrt(mean_squared_error(y_test, y_pred, squared=False))
print(f"RMSE: {rmse}")

# MAE preve o quanto as previsões medias em unidade estão variando do valor real em reais.
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

# Exibe todos os gráficos
plt.show()