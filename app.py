from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import time
import hashlib
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import sklearn.metrics as mt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score,  f1_score, classification_report
from sklearn.neighbors import RadiusNeighborsClassifier


# Função principal do Streamlit
def main():

    st.set_page_config(page_title='Trabalho 5 - Classificação', layout='wide')
    
    def highlight_md(val):
        color = 'yellow' if val == 'MD' else ''
        return f'background-color: {color}'
    
    def highlight_nan(val):
        color = 'red' if pd.isna(val) else ''
        return f'background-color: {color}'
    
    st.title("Análise de Dados e Agrupamento de Candidatos a Emprego\nAtividade 5 - Classificação")
        

    # Função para gerar o URL do Gravatar a partir do e-mail
    def get_gravatar_url(email, size=100):
        email_hash = hashlib.md5(email.strip().lower().encode('utf-8')).hexdigest()
        gravatar_url = f"https://www.gravatar.com/avatar/"
        return gravatar_url

    # Definir o e-mail e o tamanho da imagem
    email = "rodrigo.peixoto@estudante.ufjf.br"  # Substitua pelo seu e-mail
    size = 200  # Tamanho da imagem

    # Obter o URL do Gravatar
    gravatar_url = get_gravatar_url(email, size)

    # Função para verificar se o valor é numérico e não nulo
    def verificar_tipo_invalido(valor):
        if pd.isna(valor):  # Verifica se o valor é nulo
            return False
        try:
            float(valor)  # Tenta converter para float
            return False
        except ValueError:
            return True

    st.subheader('1. Carregamento dos dados e correção dos nomes das colunas - Trabalho 4')

    # Carregar os dados
    dados = pd.read_csv('data/trab4.csv')
    incialRow = len(dados)
    inicialColumns = len(dados.columns)

    # Remover espaços em branco no início e no final dos nomes das colunas e converter para Snake Case
    dados.columns = (dados.columns
                    .str.strip()                   # Remover espaços em branco no início e no final
                    .str.upper()                   # Converter para minúsculas
                    .str.replace(' ', '_')         # Substituir espaços por underscores
                    .str.replace('(', '')
                    .str.replace(')', '')
                    )
    
    # Remover espaços em branco no início e no final dos valores
    dados[dados.columns] = dados[dados.columns].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    with st.expander('1. Dados iniciais', expanded=False):
        with st.popover('Colunas'):
            st.write("Colunas: ", set(dados.columns))
        
        
        
        st.write(dados.head())
        st.write('Quantidade:', len(dados), "x", len(dados.columns))

    # Remover espaços em branco no início e no final dos nomes das colunas
        

    st.subheader('2. Dados Duplicados - Trabalho 4')
    with st.expander('Remoção de duplicatas', expanded=False):
        with st.popover('Código'):
            st.code('''
                
            ''')
            
        duplicatas = dados[dados.duplicated(keep=False)].sort_values(by=["CANDIDATE_ID"])      
       

        st.write(duplicatas.head(), expande=True)    
        st.write("Linhas: ", (dados.duplicated().sum()))
        
        st.write("**Eliminação dos dados duplicados**")
        
        dados.drop_duplicates(inplace=True)
        
        st.write(dados.head(), expanded=True)
        st.write("Linhas: ", len(dados), " x ", "QTD Inicial: ", incialRow)
      
    st.subheader('3. Tratamento `MD` - Trabalho 4')
    with st.expander('Código e visualização dos dados', expanded=False):
        colunas_com_md = dados.columns[dados.isin(['MD']).any()]
        linhas_com_md = dados[dados[colunas_com_md].isin(['MD']).any(axis=1)]
        
        with st.popover('Código'):
            st.write("Colunas com MD:", set(colunas_com_md))
            st.code(''' 
        coluna_alvo = 'PERFORMANCE'
        X = dados.drop(columns=[coluna_alvo])  # Excluindo a coluna alvo
        y = dados[coluna_alvo]
        
        X = pd.get_dummies(X, drop_first=True)

        # Dividir o conjunto de dados em linhas com e sem NaN na coluna alvo
        dados_completos = dados[dados[coluna_alvo].notna()]
        dados_incompletos = dados[dados[coluna_alvo].isna()]

        # Separar X e y para os dados completos
        X_completos = X.loc[dados_completos.index]
        y_completos = y.loc[dados_completos.index]

        # Separar X para os dados incompletos (sem y, porque estamos prevendo)
        X_incompletos = X.loc[dados_incompletos.index]

        # Dividir os dados completos em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X_completos, y_completos, test_size=0.2, random_state=42)

        # Criar e treinar o modelo de árvore de decisão
        modelo = DecisionTreeClassifier(random_state=42)
        modelo.fit(X_train, y_train)

        # Prever os valores faltantes nas linhas incompletas
        y_pred = modelo.predict(X_incompletos)

        # Substituir os valores NaN pelos valores previstos
        dados.loc[dados[coluna_alvo].isna(), coluna_alvo] = y_pred
        
        acuracia = modelo.score(X_test, y_test) ''')
              
        
        head_md = linhas_com_md[colunas_com_md].head(24)
        head_md = head_md.style.applymap(highlight_md)
        
        st.write(head_md, expanded=True)
        st.write("QTD: ", len(linhas_com_md))
        
        dados.replace("MD", np.nan, inplace=True)
        linhas_com_md = dados[dados.isnull().any(axis=1)]
        head_md = linhas_com_md[colunas_com_md].head(24)
        head_md = head_md.style.applymap(highlight_nan)
              
        st.info("**Substituindo os valores MD por NaN**")
        
        st.write(head_md, expanded=True)
        st.write("QTD: ", len(linhas_com_md))
        
    
        ####################################################
        coluna_alvo = 'PERFORMANCE'
        X = dados.drop(columns=[coluna_alvo])  # Excluindo a coluna alvo
        y = dados[coluna_alvo]

        # Codificar variáveis categóricas
        X = pd.get_dummies(X, drop_first=True)

        # Dividir o conjunto de dados em linhas com e sem NaN na coluna alvo
        dados_completos = dados[dados[coluna_alvo].notna()]
        dados_incompletos = dados[dados[coluna_alvo].isna()]

        # Separar X e y para os dados completos
        X_completos = X.loc[dados_completos.index]
        y_completos = y.loc[dados_completos.index]

        # Separar X para os dados incompletos (sem y, porque estamos prevendo)
        X_incompletos = X.loc[dados_incompletos.index]

        # Dividir os dados completos em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X_completos, y_completos, test_size=0.2, random_state=42)

        # Criar e treinar o modelo de árvore de decisão
        modelo = DecisionTreeClassifier(random_state=42)
        modelo.fit(X_train, y_train)

        # Prever os valores faltantes nas linhas incompletas
        y_pred = modelo.predict(X_incompletos)

        # Substituir os valores NaN pelos valores previstos
        dados.loc[dados[coluna_alvo].isna(), coluna_alvo] = y_pred
        
        acuracia = modelo.score(X_test, y_test)
        
        ############################################################################
        st.info("**Performance: DecisionTreeClassifier**")
        linhas_com_md = dados[dados.isnull().any(axis=1)]
        head_md = linhas_com_md[colunas_com_md].head(24)
        head_md = head_md.style.applymap(highlight_nan)
        st.write(head_md, expanded=True)
        
        st.write("Acurácia: ", acuracia)
        
        st.info("DecisionTreeRegressor: DOMAIN_SKILLS_1, ANALYTICAL_SKILLS_1, QUANTITATIVE_ABILITY_1")
        
        

        # Suponha que as variáveis numéricas que você deseja prever sejam as seguintes
        coluna_alvo = ['ANALYTICAL_SKILLS_1', 'DOMAIN_SKILLS_1', 'QUANTITATIVE_ABILITY_1']

        # Loop para prever valores faltantes em cada coluna alvo separadamente
        for coluna in coluna_alvo:
            
            # Escolher as colunas para usar como features (X) e a coluna alvo (y)
            X = dados.drop(columns=coluna_alvo)
            y = dados[coluna]

            # Codificar variáveis categóricas
            X = pd.get_dummies(X, drop_first=True)

            # Dividir o conjunto de dados em linhas com e sem NaN na coluna alvo
            dados_completos = dados[dados[coluna].notna()]
            dados_incompletos = dados[dados[coluna].isna()]

            # Separar X e y para os dados completos
            X_completos = X.loc[dados_completos.index]
            y_completos = y.loc[dados_completos.index]

            # Separar X para os dados incompletos (sem y, porque estamos prevendo)
            X_incompletos = X.loc[dados_incompletos.index]

            # Verificar se existem dados incompletos para prever
            if X_incompletos.shape[0] > 0:
                # Dividir os dados completos em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(X_completos, y_completos, test_size=0.2, random_state=42)

                # Criar e treinar o modelo de regressão com árvore de decisão
                modelo = DecisionTreeRegressor(random_state=42)
                modelo.fit(X_train, y_train)

                # Prever os valores faltantes nas linhas incompletas
                y_pred = modelo.predict(X_incompletos)

                # Substituir os valores NaN pelos valores previstos
                dados.loc[dados[coluna].isna(), coluna] = y_pred

                # Calcular o desempenho do modelo no conjunto de teste
                r2_score = modelo.score(X_test, y_test)
                st.write(f"Métrica do modelo para {coluna}: {r2_score:.2f}")

            # Exibir o DataFrame após a substituição
        st.write(dados[colunas_com_md].head(), expanded=True)

    st.subheader('4. Trabalho 4 - Tratamento dos dados, Normalização, MiniBatchKMeans, AgglomerativeClustering e DBSCAN')
    with st.expander("Informações do trabalho 4 anterior", expanded=False):             
        
            
        dropColumns = ['CANDIDATE_ID',
            'NAME',
            'NUMBER_OF_CHARACTERS_IN_ORIGINAL_NAME',
            'MONTH_OF_BIRTH', 'YEAR_OF_BIRTH']
        
        dados.drop(dropColumns, axis=1, inplace=True)
        
        st.write("Drop colunas: ", set(dropColumns))
        
        colunas_para_lstrip = ['10TH_COMPLETION_YEAR', '12TH_COMPLETION_YEAR', 'YEAR_OF_COMPLETION_OF_COLLEGE']

        # Aplicando lstrip('Y') a cada coluna da lista
        for coluna in colunas_para_lstrip:
            dados[coluna] = dados[coluna].str.lstrip('Y')

        # Exibindo o DataFrame após as alterações
        st.write("Remove o 'Y' das colunas de data: ", set(colunas_para_lstrip))
                
        #transformação das colunas categóricas
        colunas_categoricas = ['GENDER', 'STATE_LOCATION', 'DEGREE_OF_STUDY', 'SPECIALIZATION_IN_STUDY']
        dados_encoded =  pd.get_dummies(dados, columns=colunas_categoricas, prefix=colunas_categoricas)
        st.write("Transformação das colunas categóricas: ", set(colunas_categoricas))
        st.write(dados_encoded.head(), expanded=True)
        
        st.info("**Normalização dos dados**")
        dados_encoded = dados_encoded.dropna()
        dados_encoded['10TH_COMPLETION_YEAR'] = pd.to_numeric(dados_encoded['10TH_COMPLETION_YEAR'], errors='coerce')
        dados_encoded['12TH_COMPLETION_YEAR'] = pd.to_numeric(dados_encoded['12TH_COMPLETION_YEAR'], errors='coerce')
        dados_encoded['YEAR_OF_COMPLETION_OF_COLLEGE'] = pd.to_numeric(dados_encoded['YEAR_OF_COMPLETION_OF_COLLEGE'], errors='coerce')
        dados_encoded['QUANTITATIVE_ABILITY_1'] = pd.to_numeric(dados_encoded['QUANTITATIVE_ABILITY_1'], errors='coerce')
        dados_encoded['DOMAIN_SKILLS_1'] = pd.to_numeric(dados_encoded['DOMAIN_SKILLS_1'], errors='coerce')
        dados_encoded['ANALYTICAL_SKILLS_1'] = pd.to_numeric(dados_encoded['ANALYTICAL_SKILLS_1'], errors='coerce')

        # Converter essas colunas para float
        dados_encoded['10TH_COMPLETION_YEAR'] = dados_encoded['10TH_COMPLETION_YEAR'].astype(float)
        dados_encoded['12TH_COMPLETION_YEAR'] = dados_encoded['12TH_COMPLETION_YEAR'].astype(float)
        dados_encoded['YEAR_OF_COMPLETION_OF_COLLEGE'] = dados_encoded['YEAR_OF_COMPLETION_OF_COLLEGE'].astype(float)
        dados_encoded['QUANTITATIVE_ABILITY_1'] = dados_encoded['QUANTITATIVE_ABILITY_1'].astype(float)
        dados_encoded['DOMAIN_SKILLS_1'] = dados_encoded['DOMAIN_SKILLS_1'].astype(float)
        dados_encoded['ANALYTICAL_SKILLS_1'] = dados_encoded['ANALYTICAL_SKILLS_1'].astype(float)

        # Lista das colunas que você deseja converter para int
        colunas_para_converter = [
            'GENDER_A', 'GENDER_B',
            'STATE_LOCATION_A', 'STATE_LOCATION_AB', 'STATE_LOCATION_B', 'STATE_LOCATION_C',
            'STATE_LOCATION_D', 'STATE_LOCATION_E', 'STATE_LOCATION_F', 'STATE_LOCATION_G',
            'STATE_LOCATION_H', 'STATE_LOCATION_I', 'STATE_LOCATION_J', 'STATE_LOCATION_K',
            'STATE_LOCATION_L', 'STATE_LOCATION_M', 'STATE_LOCATION_N', 'STATE_LOCATION_O',
            'STATE_LOCATION_Q', 'STATE_LOCATION_R', 'STATE_LOCATION_S', 'STATE_LOCATION_T',
            'STATE_LOCATION_Y', 'STATE_LOCATION_Z',
            'DEGREE_OF_STUDY_W', 'DEGREE_OF_STUDY_X', 'DEGREE_OF_STUDY_Y',
            'SPECIALIZATION_IN_STUDY_B', 'SPECIALIZATION_IN_STUDY_C', 'SPECIALIZATION_IN_STUDY_D',
            'SPECIALIZATION_IN_STUDY_E', 'SPECIALIZATION_IN_STUDY_F', 'SPECIALIZATION_IN_STUDY_G',
            'SPECIALIZATION_IN_STUDY_H', 'SPECIALIZATION_IN_STUDY_I', 'SPECIALIZATION_IN_STUDY_J',
            'SPECIALIZATION_IN_STUDY_K', 'SPECIALIZATION_IN_STUDY_L'
        ]

        # Converter as colunas para int
        for coluna in colunas_para_converter:
            dados_encoded[coluna] = dados_encoded[coluna].astype(int)
        
        # Inicializar o MinMaxScaler com o intervalo desejado
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        dados_encoded = dados_encoded.dropna()
        dados_com_performance = dados_encoded.copy()
        dados_encoded.drop(['PERFORMANCE'], axis=1, inplace=True)

        # Aplicar o scaler aos dados
        dados_normalizados = scaler.fit_transform(dados_encoded)

        # Converter o array numpy resultante de volta em um DataFrame para fácil manipulação
        dados_normalizados_final = pd.DataFrame(dados_normalizados, columns=dados_encoded.columns)
        
        st.write(dados_normalizados_final.head(), expanded=True)
        
        st.info("Redução de dimensionalidade")
        # Suponha que você queira reduzir para 2 dimensões
        pca = PCA(n_components=2)
        dados_reduzidos = pca.fit_transform(dados_normalizados_final)

        # Se quiser visualizar a variância explicada por cada componente:
        print(pca.explained_variance_ratio_)

        # Se quiser transformar de volta para um DataFrame:
        dados_reduzidos_final = pd.DataFrame(dados_reduzidos, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

        st.write(dados_reduzidos_final.head(), expanded=True)
        ###########################################################
        '''from sklearn.cluster import MiniBatchKMeans
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.cluster import DBSCAN
        import matplotlib.pyplot as plt
        
        with st.sidebar.expander("`[4]` Agrupamento"):            
            linkageAgglomerativeClustering = st.selectbox('linkage', ['average','single', 'complete', 'ward'])
            affinityAgglomerativeClustering = st.selectbox('affinity', ['euclidean','l1', 'l2', 'manhattan', 'cosine'])

        
        # MiniBatchKMeans
        mb_kmeans = MiniBatchKMeans(n_clusters=4, batch_size=25)
        clusters_kmeans_n = mb_kmeans.fit_predict(dados_reduzidos_final)
        dados_reduzidos_final['Cluster_KMeans'] = clusters_kmeans_n
        dados_encoded['Cluster_KMeans'] = clusters_kmeans_n
        dados_com_performance['Cluster_KMeans'] = clusters_kmeans_n

        # Agglomerative Clustering
        agglomerative = AgglomerativeClustering(n_clusters=4, linkage=linkageAgglomerativeClustering, affinity=affinityAgglomerativeClustering)
        clusters_agglo_n = agglomerative.fit_predict(dados_reduzidos_final)
        dados_reduzidos_final['Cluster_Agglo'] = clusters_agglo_n
        dados_encoded['Cluster_Agglo'] = clusters_agglo_n
        dados_com_performance['Cluster_Agglo'] = clusters_agglo_n

        # DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        clusters_dbscan_n = dbscan.fit_predict(dados_reduzidos_final)
        dados_reduzidos_final['Cluster_DBSCAN'] = clusters_dbscan_n
        dados_encoded['Cluster_DBSCAN'] = clusters_dbscan_n
        dados_com_performance['Cluster_DBSCAN'] = clusters_dbscan_n

        # Supondo que você queira usar as duas primeiras colunas após a normalização e redução para plotagem
        coluna_x = dados_reduzidos_final.columns[0]  # Primeira coluna para o eixo x
        coluna_y = dados_reduzidos_final.columns[1]  # Segunda coluna para o eixo y

        # Plotar os clusters gerados pelos algoritmos
        plt.figure(figsize=(24, 8))

        # Gráfico para MiniBatchKMeans
        plt.subplot(1, 3, 1)
        plt.scatter(dados_reduzidos_final[coluna_x], dados_reduzidos_final[coluna_y], c=dados_reduzidos_final['Cluster_KMeans'], cmap='viridis', marker='o', edgecolor='k')
        plt.title('Clusters MiniBatchKMeans após Normalização')
        plt.xlabel(coluna_x)
        plt.ylabel(coluna_y)

        # Gráfico para Agglomerative Clustering
        plt.subplot(1, 3, 2)
        plt.scatter(dados_reduzidos_final[coluna_x], dados_reduzidos_final[coluna_y], c=dados_reduzidos_final['Cluster_Agglo'], cmap='viridis', marker='o', edgecolor='k')
        plt.title('Clusters Agglomerative Clustering após Normalização')
        plt.xlabel(coluna_x)
        plt.ylabel(coluna_y)

        # Gráfico para DBSCAN
        plt.subplot(1, 3, 3)
        plt.scatter(dados_reduzidos_final[coluna_x], dados_reduzidos_final[coluna_y], c=dados_reduzidos_final['Cluster_DBSCAN'], cmap='viridis', marker='o', edgecolor='k')
        plt.title('Clusters DBSCAN após Normalização')
        plt.xlabel(coluna_x)
        plt.ylabel(coluna_y)

        # Ajustar layout e mostrar os gráficos
        plt.tight_layout()
        st.pyplot(plt)'''
           
    st.subheader('5.1. Montar modelos que separem as instâncias com BP das instâncias com MP e LP.')
    with st.expander("Separando as instâncias", expanded=True):
        
        st.write(set(dados["PERFORMANCE"].unique()))
        st.info("Separação dos Grupos BP")
        st.info("Deseja-se montar modelos que separem as instâncias com BP das instâncias com MP e LP.")
        
        # Criar uma nova coluna 'is_BP' que será 1 para BP e 0 para MP ou LP
        dados["BP"] = dados['PERFORMANCE'].apply(lambda x: 1 if x == 'BP' else 0)
        # Codificar 'AP'
        dados["LP"] = dados['PERFORMANCE'].apply(lambda x: 1 if x == 'LP' else 0)
        # Codificar 'CP'
        dados["MP"] = dados['PERFORMANCE'].apply(lambda x: 1 if x == 'MP' else 0)
        
        # Verificar as mudanças]
        y = dados['BP']        
        
        with st.sidebar.expander("SMOTE"):
            
            chkSmote = st.checkbox("SMOTE HABILITADO")
            
        x = dados_normalizados_final
        
        if chkSmote:
                        
            ######################################
            # Supondo que X e y já foram definidos antes no código
            # Aplicar SMOTE para balancear as classes
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(x, y)

            # Atualizando as variáveis X e y para os dados balanceados
            x = X_resampled
            y = y_resampled

            # Aplicar normalização após o SMOTE
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        

        texto = """


**Aplicação do SMOTE no Código**

No meu projeto, foi necessário lidar com um conjunto de dados desequilibrado, onde uma das classes estava significativamente sub-representada em comparação às outras. Para mitigar o impacto desse desequilíbrio na performance dos modelos de machine learning, apliquei a técnica SMOTE (Synthetic Minority Over-sampling Technique).

O SMOTE foi utilizado após a divisão dos dados em variáveis dependentes (rótulos) e independentes, mas antes da normalização. A aplicação do SMOTE consiste em gerar novas amostras sintéticas da classe minoritária, equilibrando a quantidade de amostras entre as classes. Isso é feito através da interpolação de amostras existentes, criando novos dados que ajudam a evitar o viés dos modelos em favor da classe majoritária.

Após o balanceamento com o SMOTE, os dados foram normalizados usando a técnica de `StandardScaler`, que ajusta os dados para que tenham média 0 e desvio padrão 1. A normalização foi realizada após o SMOTE para garantir que as amostras sintéticas fossem criadas no espaço original dos dados, mantendo as características originais das variáveis.

**Benefícios da Aplicação do SMOTE**

1. **Redução de Viés do Modelo**: Ao balancear as classes, o SMOTE ajuda a evitar que o modelo aprenda a priorizar a classe majoritária, o que poderia resultar em um desempenho fraco na identificação da classe minoritária. Isso é especialmente importante em problemas onde a classe minoritária tem uma importância crítica, como em diagnósticos médicos ou detecção de fraudes.

2. **Melhoria na Performance**: Com um conjunto de dados mais balanceado, o modelo tem a oportunidade de aprender melhor as características das classes minoritárias, o que pode melhorar métricas de avaliação como a precisão, recall e F1-score.

3. **Manutenção das Características Originais**: Como o SMOTE é aplicado antes da normalização, as novas amostras geradas mantêm as relações e distâncias no espaço original dos dados, preservando a integridade dos dados durante o processo de normalização posterior.

Em resumo, a aplicação do SMOTE foi um passo crucial para garantir que o modelo desenvolvido seja capaz de identificar com precisão as instâncias de todas as classes, garantindo um desempenho robusto e confiável.

---
   
        """
        
        ######################################
        
        dados_normalizados_final =  pd.concat([dados_normalizados_final, dados[['BP']]], axis=1)    
        
                
        tudo = st.checkbox("Head", True)
        if not tudo :
            st.write(dados[['PERFORMANCE', 'BP']], expanded=True)
        else:
            st.write(dados[['PERFORMANCE', 'BP']].head(), expanded=True)
        
    st.subheader('Trabalho 4. PCA')
    with st.expander('Visualizar os pontos classificados no espaço PCA', expanded=False):
        
        
        # Slider para escolher o número de componentes principais
        st.write("Quantidade e componentes")
        valor_pca = st.slider(
            'PCA: n_components ',
            min_value=2,
            max_value=10,
            value=2,  
            step=1    
        )

        # Aplicação do PCA com o número de componentes escolhido
        pca = PCA(n_components=valor_pca)
        X_pca = pca.fit_transform(x)

        # Criação dinâmica do DataFrame com todos os componentes principais
        pca_columns = [f'PC{i+1}' for i in range(valor_pca)]
        pca_df = pd.DataFrame(data=X_pca, columns=pca_columns)
        pca_df['BP'] = y.values  # Adicionar a coluna de rótulos ao DataFrame

        # Define o tamanho do gráfico e posição da legenda
        altura_grafico = 700  # Ajuste a altura conforme necessário
        posicao_legenda = "right"  # Posição da legenda (pode ser "top", "bottom", "left", "right")

        # Plotagem baseada no número de componentes principais
        if valor_pca == 2:
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='BP', title='PCA: Visualização das Classes no Espaço 2D', height=altura_grafico, color_continuous_scale='Viridis')
        elif valor_pca == 3:
            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='BP', title='PCA: Visualização das Classes no Espaço 3D', height=altura_grafico, color_continuous_scale='Viridis')
        else:
            fig = px.scatter_matrix(pca_df, dimensions=pca_columns, color='BP', title=f'PCA: Visualização das Classes com {valor_pca} Componentes Principais', height=altura_grafico)

        
        # Ajustar a posição e o estilo da legenda
        fig.update_layout(
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(l=0, r=0, t=40, b=0)  # Ajuste das margens para dar mais espaço ao gráfico
        )
        # Exibe o gráfico interativo no Streamlit
        st.plotly_chart(fig)
    
    
    st.subheader('5.2. KNN')
    with st.expander('knn', expanded=True):
        dados_novos = dados
        with st.popover('Código'):
            st.code('''
        knn_model = KNeighborsClassifier(n_neighbors=valor_vizinho)  # Aqui k=3 é usado, mas pode ser ajustado

        # Treinar o modelo
        knn_model.fit(X_train_smote, y_train_smote)

        # Fazer previsões no conjunto de teste
        y_pred_knn = knn_model.predict(X_test)
        
        metrics_average = st.selectbox('metrics: average', ['micro','macro', 'weighted'])

        
        precisao = mt.precision_score(y_test, y_pred_knn, average=metrics_average)
        
        
        recall = mt.recall_score(y_test, y_pred_knn, average=metrics_average)
        f1 = mt.f1_score(y_test, y_pred_knn, average=metrics_average)
        revocacao = mt.recall_score(y_test, y_pred_knn, average=metrics_average)
        
        st.write("Precisão: Mede o quão bem o modelo evita falsos positivos")
        st.write("Precisão:", precisao)
        
        st.write("Revocação: Mede o quão bem o modelo captura todos os positivos reais")
        st.write("Recall:", recall)
        
        st.write("F1-score: Proporciona um equilíbrio entre precisão e revocação")
        st.write("F1 Score:", f1)
        
        st.write("Recall por classe:", revocacao)
        
        st.write("Relatório de classificação para KNN/classification_report:")
        st.text(classification_report(y_test, y_pred_knn))
                    ''')
               
        valor_selecionado2 = st.slider(
            'train_test_split: % test_size ',
            min_value=0.1,
            max_value=0.9,
            value=0.1,  # Valor inicial do slider
            step=0.1    # Incremento de 0.1
        )

        # Dividir os dados em conjunto de treino e teste
        X_train_smote, X_test,y_train_smote, y_test = train_test_split(x, y, test_size=valor_selecionado2, random_state=42, stratify=y)

        st.write("Quantidade e vizinho")
        valor_vizinho = st.slider(
            'KNeighborsClassifier: n_neighbors ',
            min_value=2,
            max_value=40,
            value=7,  
            step=1    
        )
        # Instanciar o modelo kNN, escolher o valor de k (n_neighbors)
        knn_model = KNeighborsClassifier(n_neighbors=valor_vizinho)  # Aqui k=3 é usado, mas pode ser ajustado

        # Treinar o modelo
        knn_model.fit(X_train_smote, y_train_smote)

        # Fazer previsões no conjunto de teste
        y_pred_knn = knn_model.predict(X_test)
        
        metrics_average = st.selectbox('metrics: average', ['micro','macro', 'weighted'])

        
        precisao = mt.precision_score(y_test, y_pred_knn, average=metrics_average)
        
        
        recall = mt.precision_score(y_test, y_pred_knn, average=metrics_average)
        f1 = mt.f1_score(y_test, y_pred_knn, average=metrics_average)
        revocacao = mt.recall_score(y_test, y_pred_knn, average=metrics_average)

        st.write("precision_score: Mede o quão bem o modelo evita falsos positivos")
        st.write("precision_score:", precisao)
        
        st.write("recall_score: Mede o quão bem o modelo captura todos os positivos reais")
        st.write("recall_score:", recall)
        
        st.write("F1-score: Proporciona um equilíbrio entre precisão e revocação")
        st.write("F1 Score:", f1)
        
        st.write("Recall por classe:", revocacao)
        
        st.write("Relatório de classificação para KNN/classification_report:")
        st.text(classification_report(y_test, y_pred_knn))
        
        ####################################
        st.markdown("---")
        st.info("KNN por raio")
        # Instanciar o modelo por raio e treinar
        # Selecionar o tipo de média para as métricas
        metrics_average = st.selectbox('Escolha a métrica de average', ['micro', 'macro', 'weighted'])
        
        valor_raio = st.slider('Escolha o raio para o modelo', min_value=3.0, max_value=10.0, step=0.1, value=3.1)
        raio_modelo = RadiusNeighborsClassifier(radius=valor_raio)
        raio_modelo.fit(X_train_smote, y_train_smote)

        # Fazer previsões no conjunto de teste
        y_pred_raio = raio_modelo.predict(X_test)
        
        # Calcular as métricas
        precisao = precision_score(y_test, y_pred_raio, average=metrics_average)
        recall = recall_score(y_test, y_pred_raio, average=metrics_average)
        f1 = f1_score(y_test, y_pred_raio, average=metrics_average)

        # Exibir as métricas
        st.write("precision_score:", precisao)
        st.write("recall_score:", recall)
        st.write("F1-Score:", f1)

        # Mostrar o relatório de classificação
        st.write("### Relatório de Classificação para Radius Neighbors")
        st.text(classification_report(y_test, y_pred_raio))

        # Aplicar PCA para reduzir as dimensões para 2D e visualizar
        pca = PCA(n_components=2)
        X_test_pca = pca.fit_transform(X_test)

        # Criar um DataFrame para facilitar a plotagem
        pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2'])
        pca_df['Classe Real'] = y_test
        pca_df['Classe Predita'] = y_pred_raio

        # Plotar as classes com diferentes formas geométricas
        st.write("### Gráfico PCA com Diferentes Formas para as Classes (Radius Neighbors)")

        fig, ax = plt.subplots()

        # Plotando as classes reais com diferentes formas
        for classe, marker, color in zip([0, 1], ['s', '^'], ['blue', 'red']):
            subset = pca_df[pca_df['Classe Real'] == classe]
            ax.scatter(subset['PC1'], subset['PC2'], marker=marker, color=color, label=f'Classe {classe} Real')

        # Plotando as classes preditas com diferentes formas
        for classe, marker, color in zip([0, 1], ['o', 'v'], ['cyan', 'magenta']):
            subset = pca_df[pca_df['Classe Predita'] == classe]
            ax.scatter(subset['PC1'], subset['PC2'], marker=marker, facecolors='none', edgecolors=color, label=f'Classe {classe} Predita')

        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_title('PCA: Comparação das Classes com Diferentes Formas (Radius Neighbors)')
        ax.legend()

        st.pyplot(fig)
     

    st.subheader("")
    with st.expander("", expanded=False):
        
        dados_normalizados_final.dropna(inplace=True)


        #são as features (todas as colunas exceto a variável alvo).
        x = dados_normalizados_final.iloc[:, :58]  
        
        #é a variável alvo (por exemplo, 'PERFORMANCE').
        y = dados_normalizados_final.iloc[:, 59]  
        
       
        # Dividir os dados entre classes para visualização
        x0 = x[y == 0]
        x1 = x[y == 1]
        
        st.write(x[y == 0])

        # Instanciar e treinar o modelo Naive Bayes
        modelo = GaussianNB()
        modelo.fit(x, y)

        # Escolher um novo ponto para prever (aqui só para exemplo, mantenha as 58 features)
        novo_ponto = x.mean().values.reshape(1, -1)  # Aqui eu usei a média das features como exemplo

        # Previsão do novo ponto
        yt = modelo.predict(novo_ponto)

        # Visualização usando matplotlib no Streamlit
        plt.figure(figsize=(10, 6))
        plt.plot(x0.iloc[:, 0], x0.iloc[:, 1], "bs", label="Classe 0")
        plt.plot(x1.iloc[:, 0], x1.iloc[:, 1], "g^", label="Classe 1")
        if yt[0] == 0:
            plt.plot(novo_ponto[:, 0], novo_ponto[:, 1], "rs", label="Novo ponto (Classe 0)")
        else:
            plt.plot(novo_ponto[:, 0], novo_ponto[:, 1], "r^", label="Novo ponto (Classe 1)")

        plt.title('Visualização do Modelo Naive Bayes')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        # Exibindo o gráfico no Streamlit
        st.pyplot(plt)

    st.subheader("5.3. Comparação dos tipos de modelos devem ser gerados e comparados")
    with st.expander("", expanded=True):
        
        # 2. Preparar as features (X) e a variável alvo (y)
        X = dados_normalizados_final.drop(columns=['BP'])  # Supondo que 'alvo' já foi criado anteriormente
        y = dados_normalizados_final['BP']
        st.write("Dimensões dos dados:")
        st.write(f"Features: {X.shape}")
        st.write(f"Variável Alvo: {y.shape}")
        
        with st.sidebar.expander("`[5.3]` Comparação"):
            valor_selecionado3 = st.slider(
                ' train_test_split: % test_size ',
                min_value=0.1,
                max_value=0.9,
                value=0.3,  # Valor inicial do slider
                step=0.1    # Incremento de 0.1
            )
        

        # 3. Dividir os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valor_selecionado3, random_state=42)
        st.write("Divisão dos dados:")
        st.write(f"Treinamento: {X_train.shape[0]} instâncias")
        st.write(f"Teste: {X_test.shape[0]} instâncias")

        # 4. Treinar o primeiro modelo: Naive Bayes
        st.write("## Modelo Naive Bayes")
        modelo_nb = GaussianNB()
        modelo_nb.fit(X_train, y_train)
        y_pred_nb = modelo_nb.predict(X_test)
        
        # Avaliar o modelo Naive Bayes
        acuracia_nb = accuracy_score(y_test, y_pred_nb)
        st.write(f"Acurácia do modelo Naive Bayes: {acuracia_nb:.2f}")
        st.write("Relatório de Classificação (Naive Bayes):")
        st.text(classification_report(y_test, y_pred_nb))
        
        # Exibir a matriz de confusão para Naive Bayes
        st.write("Matriz de Confusão (Naive Bayes):")
        cm_nb = confusion_matrix(y_test, y_pred_nb)
        fig_nb, ax_nb = plt.subplots()
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", ax=ax_nb)
        ax_nb.set_title("Matriz de Confusão - Naive Bayes")
        ax_nb.set_xlabel("Classe Prevista")
        ax_nb.set_ylabel("Classe Real")
        st.pyplot(fig_nb)

        st.info("Parametros de poda")

        max_depth = st.slider(
            "Profundidade Máxima da Árvore (max_depth)", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1
        )

        st.write("Define o número mínimo de amostras que um nó deve ter antes de ser dividido em subnós. Se um nó tem menos amostras do que o valor de min_samples_split, ele não será dividido.")
        min_samples_split = st.slider(
            "Número Mínimo de Amostras para Dividir um Nó (min_samples_split)", 
            min_value=2, 
            max_value=20, 
            value=8, 
            step=1
        )

        st.write("Define o número mínimo de amostras que um nó folha (nó final da árvore) deve conter. Se um nó folha tem menos amostras do que esse valor, ele não é considerado uma folha e a divisão não é realizada.")
        min_samples_leaf = st.slider(
            "Número Mínimo de Amostras em uma Folha (min_samples_leaf)", 
            min_value=1, 
            max_value=20, 
            value=5, 
            step=1
        )
        # 5. Treinar o segundo modelo: Árvore de Decisão
        st.write("## Modelo Árvore de Decisão")
        modelo_dt = DecisionTreeClassifier(
            random_state=42,
            max_depth=max_depth,  # Profundidade da árvore ajustada pelo slider
            min_samples_split=min_samples_split,  # Min amostras para dividir ajustado pelo slider
            min_samples_leaf=min_samples_leaf  # Min amostras em folha ajustado pelo slider
        )
        modelo_dt.fit(X_train, y_train)
        y_pred_dt = modelo_dt.predict(X_test)
        
        # Avaliar o modelo Árvore de Decisão
        acuracia_dt = accuracy_score(y_test, y_pred_dt)
        st.write(f"Acurácia do modelo Árvore de Decisão: {acuracia_dt:.2f}")
        st.write("Relatório de Classificação (Árvore de Decisão):")
        st.text(classification_report(y_test, y_pred_dt))
        
        # Exibir a matriz de confusão para Árvore de Decisão
        st.write("Matriz de Confusão (Árvore de Decisão):")
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        fig_dt, ax_dt = plt.subplots()
        sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", ax=ax_dt)
        ax_dt.set_title("Matriz de Confusão - Árvore de Decisão")
        ax_dt.set_xlabel("Classe Prevista")
        ax_dt.set_ylabel("Classe Real")
        st.pyplot(fig_dt)

        # 6. Comparar os resultados
        st.write("## Comparação de Acurácia")
        st.write(f"Acurácia Naive Bayes: {acuracia_nb:.2f}")
        st.write(f"Acurácia Árvore de Decisão: {acuracia_dt:.2f}")
        
        # Exibir o modelo com melhor desempenho
        if acuracia_nb > acuracia_dt:
            st.success("O modelo Naive Bayes teve melhor desempenho!")
        elif acuracia_nb < acuracia_dt:
            st.success("O modelo Árvore de Decisão teve melhor desempenho!")
        else:
            st.info("Ambos os modelos tiveram o mesmo desempenho!")
        
        # Gerar e exibir o gráfico da árvore de decisão
        st.subheader("Visualização da Árvore de Decisão")
        fig_tree = plt.figure(figsize=(20, 10))
        
        max_depth_visualizacao = st.slider(
        "Escolha a Profundidade Máxima da Árvore para Visualização", 
        1, 
        4, 
        10
    )
        
        plot_tree(modelo_dt, max_depth=max_depth_visualizacao, filled=True, feature_names=X.columns, class_names=['MP_LP', 'BP'], rounded=True)
        
        
        st.pyplot(fig_tree)       
        
        
        
        texto = """
Essa imagem é uma representação visual de uma **árvore de decisão** gerada por um modelo de machine learning. As árvores de decisão são usadas para tomar decisões baseadas em uma série de perguntas, cada uma das quais divide os dados em subconjuntos menores até que uma decisão final seja tomada.

### Explicando a Árvore de Decisão:


1. **Condições dos Nós**:
   - **Raiz da Árvore (Nó Superior)**: A árvore começa com uma condição baseada na variável "MP". Se "MP" for menor ou igual a 0.5, a árvore se ramifica à esquerda; caso contrário, à direita.
   - **Segundo Nível de Nós**:
     - **Esquerda (LP <= 0.5)**: A condição verifica se "LP" é menor ou igual a 0.5 para as amostras que vieram do nó superior.
     - **Direita (Gini = 0.0)**: Este nó é puro, com todas as amostras pertencendo à classe "MP_LP".
     
2. **Interpretação dos Caminhos**:
   - **Caminho Esquerdo**: Se "MP <= 0.5" e "LP <= 0.5", as amostras são classificadas como "BP". Este caminho cobre a maior parte dos dados (166 amostras).
   - **Caminho Direito**: Se "MP > 0.5" ou "LP > 0.5", as amostras são classificadas como "MP_LP". 
        - **Esses caminhos levam a nós com "gini = 0.0", indicando uma classificação pura, onde todas as amostras são da mesma classe.

3. **Significado das Cores**:
   - **Azul**: Representa nós onde a classe majoritária é "BP".
   - **Laranja**: Representa nós onde a classe majoritária é "MP_LP".
       
        """
        
        #st.markdown(texto)
        
    st.subheader("5.4. Comparativo")
    with st.expander("Conclusões e informação/interpretação dos dados ", expanded=True):
        

        # Exibir o resumo comparativo
        st.subheader("Resumo Comparativo dos Modelos")

        st.write(f"Acurácia do Modelo Naive Bayes: **{acuracia_nb:.2f}**")
        st.write(f"Acurácia do Modelo Árvore de Decisão: **{acuracia_dt:.2f}**")

        if acuracia_nb > acuracia_dt:
            st.write("**O modelo Naive Bayes apresentou melhor desempenho em termos de acurácia.**")
        elif acuracia_dt > acuracia_nb:
            st.write("**O modelo Árvore de Decisão apresentou melhor desempenho em termos de acurácia.**")
        else:
            st.write("**Ambos os modelos tiveram o mesmo desempenho em termos de acurácia.**")
            
        st.info("Interpretação")
        texto = """

1   - As variáveis "MP" e "LP" desempenham um papel crucial na decisão de classificação dos candidatos. A árvore de decisão depende exclusivamente dessas variáveis para distinguir entre os candidatos "muito bons" e os outros.
   - Isso pode sugerir que essas duas variáveis são altamente indicativas do sucesso ou das qualificações dos candidatos, e podem ser usadas para orientar futuras decisões de contratação ou triagem de candidatos.
2. **Interpretação para a Gestão de Talentos**:
   - A partir da análise, podemos inferir que candidatos com baixos valores em "MP" e "LP" têm uma maior probabilidade de serem considerados "muito bons". Isso pode guiar os recrutadores a focarem nesses atributos durante o processo de seleção para identificar talentos excepcionais.
   - Por outro lado, candidatos com valores mais altos em "MP" e "LP" podem precisar de desenvolvimento adicional para serem considerados na mesma categoria.           
3. - **Aplicação na Prática**: Com base nesse modelo, a equipe de recrutamento pode desenvolver estratégias mais eficazes para identificar e priorizar candidatos altamente qualificados, melhorando o processo de seleção e potencialmente aumentando a qualidade das contratações.
        """
        st.markdown(texto)

        st.write("### Conclusão:")
        st.write("O modelo com a maior acurácia é geralmente preferível, "
                "mas é importante considerar outros fatores como interpretabilidade, "
                "tempo de treinamento, e a natureza dos dados ao escolher o modelo mais adequado para uma aplicação específica.")

    
        
    # Espaçamento para empurrar o conteúdo para o rodapé
    st.sidebar.markdown(
        f"""
        <div style="height: calc(100vh - 930px);"></div>
        """,
        unsafe_allow_html=True
    )    

    st.sidebar.markdown("**Rodrigo Mendes Peixoto**")
    st.sidebar.write("agosto de 2024")
    st.sidebar.markdown("*Disciplina: Mineração de Dados*")
    st.sidebar.markdown("*Professores*: **LUCIANA CONCEICAO DIAS CAMPOS, Heder Soares Bernardino**")
    st.sidebar.write("")

# Executar a aplicação Streamlit
if __name__ == '__main__':
    main()

