# C318_Project
Repositorio criado para trabalho em grupo da disciplina de C318

## 1- Dataset:
Foi usado um dataset para avaliar o valor das casas em ponta grossa:
As variaveis do dataset são:

    #1 -> Referência
    #2 -> Quartos
    #3 -> Banheiros
    #4 -> Suítes
    #5 -> Vagas de Garagem
    #6 -> Bairro,Área Total
    #7 -> Valor Venda

## 2- Foi avaliado grafico de cada variavel para verificar anormalidade:

Percebemos que tinham linhas que não continham os valores de alguma variavel, resolvemos descarta-las e seguir com apenas as que tinham linhas completa.


## 3- Foi avaliado novos graficos para verificar se o dataset estava normalizado:

De fato agora os valores estavam melhores e foram retirados as linhas que estavam sem valores.

## 4- Treinamento do modelo:

Utilizamos o XGBoost para treinar o modelo, usando o grid para ter os melhores parametros

## 5- Importancia de cada caracteristica:
A importancia de cada caracteristica para o preço:

    #1 - Área total 
    #2 - Banheiros
    #3 - Garagem
    #4 - Quartos
    #5 - Suítes

## 6- Metricas:
    Utilizamos as metricas RSME e MAE.
    O RSME É a raiz quadrada da média dos erros quadráticos. O RMSE penaliza erros maiores mais fortemente do que o MAE, devido à elevação ao quadrado. Isso torna o RMSE útil para cenários onde erros grandes são mais indesejáveis.

    O MAE É a média dos erros absolutos entre as previsões e os valores reais. O MAE dá uma ideia de quanto, em média, as previsões estão afastadas dos valores reais, sem considerar a direção do erro (positiva ou negativa). Ele é mais interpretável, pois é uma média das distâncias absolutas.

## 7- Resultados:
    
### Primeiro resultado:
    
    RMSE: 533.1920646953762
    MAE: 172451.31370192306

### Segundo resultado:

Os resultados não agradaram muito, pela margem de erro, decidimos dividir o dataset em dois. e adicionar colunas novas para ajudar na avaliação da casas preço.

### Colunas criadas:

    1-Preco por m2:
    Descrição: Calcula o preço de venda por metro quadrado.
    Fórmula: Valor Venda / Área Total.
    Motivo: Ajuda a normalizar os preços considerando o tamanho da casa.

    2-Faixa de Preço
    Descrição: Classifica os imóveis em faixas de preço categóricas.
    Fórmula: Usa pd.cut() com intervalos específicos para agrupar os valores de venda.
    Motivo: Fornece uma visão categorizada do valor de venda, útil para capturar padrões em diferentes intervalos de preço.

    3-Quartos por m2
    Descrição: Calcula a proporção de quartos por metro quadrado.
    Fórmula: Quartos / Área Total.
    Motivo: Identifica imóveis que oferecem mais ou menos espaço por quarto.

    4-Banheiros por m2
    Descrição: Calcula a proporção de banheiros por metro quadrado.
    Fórmula: Banheiros / Área Total.
    Motivo: Similar ao anterior, mas considerando os banheiros.

    5-Vagas de Garagem por m2
    Descrição: Calcula a proporção de vagas de garagem por metro quadrado.
    Fórmula: Vagas de Garagem / Área Total.
    Motivo: Identifica imóveis que oferecem mais ou menos espaço para estacionamento.

    6-Área Total * Quartos
    Descrição: Multiplica a área total pelo número de quartos.
    Fórmula: Área Total * Quartos.
    Motivo: Captura uma interação entre o tamanho da casa e o número de quartos.
    
    7-Área Total * Banheiros
    Descrição: Multiplica a área total pelo número de banheiros.
    Fórmula: Área Total * Banheiros.
    Motivo: Similar ao anterior, mas considerando banheiros.
    
    8-Diferença Preço - Média Bairro
    Descrição: Calcula a diferença entre o valor de venda da casa e a média dos preços no bairro.
    Fórmula: Valor Venda - Média de Valor Venda do Bairro.
    Motivo: Considera o desvio de preço em relação ao padrão do bairro.
    
    9-Bairro por Faixa de Preço
    Descrição: Combina as categorias de bairro com as categorias de faixa de preço.
    Fórmula: Bairro * Faixa de Preço.
    Motivo: Cria uma interação que pode capturar padrões específicos para cada faixa de preço em diferentes bairros.
    
    10-Área por Quartos
    Descrição: Calcula a média de área por quarto.
    Fórmula: Área Total / Quartos.
    Motivo: Captura a relação entre o tamanho da casa e o número de quartos.
    
    11-Área por Banheiros
    Descrição: Calcula a média de área por banheiro.
    Fórmula: Área Total / Banheiros.
    Motivo: Similar ao anterior, mas para banheiros.

#### 1 DIVISÃO: R$200.000 a R$1.000.000
    RMSE: 27551.490013441387
    MAE: 16402.26690821256

#### 2 DIVISÃO: R$1.000.000 a R$5.000.000
    RMSE: 131955.3321247353
    MAE: 88781.97039473684

#### DIVISÃO GERAL: R$50.000 a R$5.000.000
	RMSE: 77419.83838332203
	MAE: 45303.575050850595




