import pandas as pd
import matplotlib.pyplot as plt
# Carregar o dataset
df = pd.read_csv('./datasets/casas_ponta_grossa.csv', delimiter=',')

Qtdquartos = df['Quartos'].value_counts()

Qtdbanheiros = df['Banheiros'].value_counts()

QtdSuítes = df['Suítes'].value_counts()

Qtdvagas_de_garagem = df['Vagas de Garagem'].value_counts()

Qtdbairro = df['Bairro'].value_counts()

QtdArea_Total = df['Área Total'].value_counts()

QtdValorVenda = df['Valor Venda'].value_counts()

# Gráfico 1: Quartos
plt.figure(figsize=(10,6)) 
Qtdquartos.plot(kind='bar') 
plt.title('Contagem de Valores de Quartos', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 2: Banheiros
plt.figure(figsize=(10,6)) 
Qtdbanheiros.plot(kind='bar') 
plt.title('Contagem de Valores de Banheiros', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 3: Suítes
plt.figure(figsize=(10,6)) 
QtdSuítes.plot(kind='bar') 
plt.title('Contagem de Valores de Suítes', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 4: Vagas de Garagem
plt.figure(figsize=(10,6)) 
Qtdvagas_de_garagem.plot(kind='bar') 
plt.title('Contagem de Valores de Vagas de Garagem', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 5: Bairro
plt.figure(figsize=(10,6)) 
Qtdbairro.plot(kind='bar') 
plt.title('Contagem de Valores de Bairro', fontsize=16)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 6: Área Total (Histograma)
plt.figure(figsize=(10,6)) 
plt.hist(QtdArea_Total, bins=20)
plt.title('Distribuição de Área Total', fontsize=16)
plt.xlabel('Área Total', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Gráfico 7: Valor Venda (Histograma)
plt.figure(figsize=(10,6)) 
plt.hist(QtdValorVenda, bins=20)
plt.title('Distribuição de Valor de Venda', fontsize=16)
plt.xlabel('Valor de Venda', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.tight_layout()

# Exibe todos os gráficos
plt.show()