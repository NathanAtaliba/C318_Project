import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Carregar o dataset
df = pd.read_csv('./datasets/tabela-fipe-historico-precos.csv', delimiter=',')

