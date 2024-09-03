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
from sklearn.inspection import permutation_importance


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
    
    def describe_column(column_name):
        descriptions = {
            "UNNAMED:0": "Índice gerado automaticamente que representa a posição da linha no dataset.",
            "CANDIDATE_ID": "Um identificador exclusivo atribuído a cada candidato para diferenciá-los dentro do dataset.",
            "NAME": "Nome do candidato, geralmente abreviado para manter a privacidade ou facilitar a análise.",
            "NUMBER_OF_CHARACTERS_IN_ORIGINAL_NAME": "Número de caracteres que compõem o nome original completo do candidato.",
            "MONTH_OF_BIRTH": "Mês em que o candidato nasceu, representado em formato abreviado (por exemplo, JAN para janeiro).",
            "YEAR_OF_BIRTH": "Ano de nascimento do candidato, onde a notação 'Y7' e 'Y8' pode indicar um ano específico codificado ou um intervalo de anos.",
            "GENDER": "Gênero do candidato, representado por uma letra (por exemplo, A, B), que pode estar codificada para indicar masculino, feminino ou outro.",
            "STATE_LOCATION": "Estado ou região onde o candidato reside ou onde foi realizada a coleta de dados.",
            "10TH_PERCENTAGE": "Percentual de notas que o candidato obteve no 10º ano escolar, possivelmente referente ao ensino médio.",
            "12TH_PERCENTAGE": "Percentual de notas que o candidato obteve no 12º ano escolar, possivelmente referindo-se ao final do ensino médio ou equivalente.",
            "10TH_COMPLETION_YEAR": "Ano em que o candidato concluiu o 10º ano escolar.",
            "12TH_COMPLETION_YEAR": "Ano em que o candidato concluiu o 12º ano escolar.",
            "DEGREE_OF_STUDY": "O grau de estudo alcançado pelo candidato, como bacharelado, licenciatura, etc.",
            "SPECIALIZATION_IN_STUDY": "Especialização do candidato dentro de seu curso de graduação, como Engenharia, Ciências da Computação, etc.",
            "COLLEGE_PERCENTAGE": "Percentual de notas obtidas pelo candidato durante seu curso universitário ou de graduação.",
            "YEAR_OF_COMPLETION_OF_COLLEGE": "Ano em que o candidato concluiu sua graduação ou curso universitário.",
            "ENGLISH_1": "Primeira avaliação das habilidades do candidato em Inglês, que pode medir compreensão, gramática ou outras competências.",
            "ENGLISH_2": "Segunda avaliação das habilidades do candidato em Inglês.",
            "ENGLISH_3": "Terceira avaliação das habilidades do candidato em Inglês.",
            "ENGLISH_4": "Quarta avaliação das habilidades do candidato em Inglês.",
            "QUANTITATIVE_ABILITY_1": "Primeira avaliação das habilidades quantitativas do candidato, que pode incluir matemática, raciocínio lógico, etc.",
            "QUANTITATIVE_ABILITY_2": "Segunda avaliação das habilidades quantitativas do candidato.",
            "QUANTITATIVE_ABILITY_3": "Terceira avaliação das habilidades quantitativas do candidato.",
            "QUANTITATIVE_ABILITY_4": "Quarta avaliação das habilidades quantitativas do candidato.",
            "DOMAIN_SKILLS_1": "Primeira avaliação das habilidades específicas do candidato em um determinado domínio ou área de conhecimento.",
            "DOMAIN_SKILLS_2": "Segunda avaliação das habilidades específicas do candidato.",
            "DOMAIN_TEST_3": "Terceira avaliação ou teste em um domínio específico, possivelmente uma área técnica ou de especialização.",
            "DOMAIN_TEST_4": "Quarta avaliação ou teste em um domínio específico.",
            "ANALYTICAL_SKILLS_1": "Primeira avaliação das habilidades analíticas do candidato, que pode incluir resolução de problemas, análise de dados, etc.",
            "ANALYTICAL_SKILLS_2": "Segunda avaliação das habilidades analíticas do candidato.",
            "ANALYTICAL_SKILLS_3": "Terceira avaliação das habilidades analíticas do candidato.",
            "PERFORMANCE": "Indicador de desempenho geral do candidato, categorizado como 'BP', que pode representar 'Best Performance' ou um outro critério de desempenho específico."
        }
        
        return descriptions.get(column_name, "Descrição não encontrada para esta coluna.")

        

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
        from sklearn.cluster import MiniBatchKMeans
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
        agglomerative = AgglomerativeClustering(n_clusters=4, linkage=linkageAgglomerativeClustering, metric=affinityAgglomerativeClustering)
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
        st.pyplot(plt)
           
    st.subheader('5.1. Montar modelos que separem as instâncias com BP das instâncias com MP e LP.')
    with st.expander("Separando as instâncias", expanded=True):
        
        st.write(set(dados["PERFORMANCE"].unique()))
        st.info("Separação dos Grupos BP e não BP")
        st.info("Deseja-se montar modelos que separem as instâncias com BP das instâncias com MP e LP.")
        
        # Criar uma nova coluna 'is_BP' que será 1 para BP e 0 para MP ou LP
        dados["BP"] = dados['PERFORMANCE'].apply(lambda x: 1 if x == 'BP' else 0)
        # Codificar 'AP'
        dados["LP"] = dados['PERFORMANCE'].apply(lambda x: 1 if x == 'LP' else 0)
        # Codificar 'CP'
        dados["MP"] = dados['PERFORMANCE'].apply(lambda x: 1 if x == 'MP' else 0)
        
        # Verificar as mudanças]
        y = dados['BP']        
        
        with st.sidebar.expander("SMOTE", expanded=True):
            
            chkSmote = st.checkbox("SMOTE HABILITADO")
            
        x = dados_normalizados_final
        
        
        

        texto = """


**Aplicação do SMOTE no Código**

1. **Redução de Viés do Modelo**: Ao balancear as classes, o SMOTE ajuda a evitar que o modelo aprenda a priorizar a classe majoritária, o que poderia resultar em um desempenho fraco na identificação da classe minoritária. Isso é especialmente importante em problemas onde a classe minoritária tem uma importância crítica, como em diagnósticos médicos ou detecção de fraudes.

2. **Melhoria na Performance**: Com um conjunto de dados mais balanceado, o modelo tem a oportunidade de aprender melhor as características das classes minoritárias, o que pode melhorar métricas de avaliação como a precisão, recall e F1-score.
---

    
   
        """
        st.markdown(texto)
        
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
    
    knn_codigo = """
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
        X_train, X_test,y_train, y_test = train_test_split(x, y, test_size=valor_selecionado2, random_state=42, stratify=y)

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
        knn_model.fit(X_train, y_train)

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
        raio_modelo.fit(X_train, y_train)

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

        ###############################################################
        dados_normalizados_final.dropna(inplace=True)
        dados_knn_r = dados_normalizados_final.copy()

        # Separar as features e o rótulo
        X = dados_knn_r.drop(columns=['BP'])  # Features, sem a coluna is_BP
        y = dados_knn_r['BP']  # Rótulo, que queremos prever

        # Dividir os dados em conjunto de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valor_selecionado2, random_state=42, stratify=y)

        # Aplicar o SMOTE para lidar com o desbalanceamento no conjunto de treinamento
        if chkSmote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Aplicar o PCA para reduzir a dimensionalidade, mantendo 95% da variância
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        valor_selecionado5 = st.slider(
            'Gráfico train_test_split: % test_size ',
            min_value=1.2,
            max_value=10.0,
            value=1.2,  # Valor inicial do slider
            step=0.1    # Incremento de 0.1
        )

        # Instanciar o modelo RadiusNeighborsClassifier com raio 1.2
        radius_model = RadiusNeighborsClassifier(radius=valor_selecionado5)
        radius_model.fit(X_train_pca, y_train)

        # Avaliar o modelo com os dados de teste
        y_pred = radius_model.predict(X_test_pca)
        
        # Calcular a acurácia
        acuracia = accuracy_score(y_test, y_pred)

        # Gerar o relatório de classificação
        report = classification_report(y_test, y_pred, output_dict=True)

        # Converter o relatório de classificação em um DataFrame
        df_report = pd.DataFrame(report).transpose()

        # Remover linhas desnecessárias e ajustar a exibição para duas casas decimais
        df_report = df_report.drop(['accuracy'], axis=0).round(2)

        # Excluindo linhas que não são classes
        df_report = df_report.drop([ 'macro avg', 'weighted avg'])

        # Plotando a acurácia e o relatório de classificação como barplot
        ax = df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
        plt.title(f"Relatório de Classificação para RadiusNeighbors - Acurácia: {acuracia:.2%}")
        plt.ylabel('Score')
        plt.xlabel('Classes')
        plt.ylim(0, 1)
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')

        # Mostrar os valores no topo de cada barra
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

        plt.tight_layout()
        plt.show()

        # Criar um novo ponto nas features originais
        novo_ponto = X.mean().values.reshape(1, -1)
        novo_ponto[0, 0] = 40  # Configurando a primeira feature (x)
        novo_ponto[0, 1] = -35  # Configurando a segunda feature (y)

        # Transformar o novo ponto para o espaço PCA
        novo_ponto_pca = pca.transform(novo_ponto)

        # Prever a classe do novo ponto
        yt = radius_model.predict(novo_ponto_pca)

        # Encontrar os vizinhos dentro do raio de 1.2
        distances, indices = radius_model.radius_neighbors(novo_ponto_pca)

        # Mostrar o resultado da previsão
        print(f"Novo ponto classificado como: {yt[0]}")

        # Visualizar os componentes principais
        plt.figure(figsize=(10, 6))

        # Plotar os dados originais com marcadores específicos para cada classe
        plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], c='blue', marker='o', label='Classe 0')
        plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c='green', marker='s', label='Classe 1')

        # Adicionar os vizinhos mais próximos ao gráfico
        if len(indices[0]) > 0:
            plt.scatter(X_train_pca[indices[0], 0], X_train_pca[indices[0], 1], c='black', marker='x', s=100, label='Vizinhos no raio')

        # Adicionar o novo ponto ao gráfico
        if yt[0] == 0:
            plt.scatter(novo_ponto_pca[0, 0], novo_ponto_pca[0, 1], c='red', marker='o', s=150, label='Novo ponto (Classe 0)')
        else:
            plt.scatter(novo_ponto_pca[0, 0], novo_ponto_pca[0, 1], c='red', marker='s', s=150, label='Novo ponto (Classe 1)')

        plt.title('Visualização PCA com Novo Ponto e Vizinhos no Raio')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend()
        st.pyplot(plt)
        
        
        # Utilizar permutation importance para avaliar a importância das variáveis
        # Utilizar permutation importance para avaliar a importância das variáveis
        result = permutation_importance(knn_model, X_test, y_test, n_repeats=10, random_state=42)

        # Obter as importâncias e os nomes das variáveis
        importances = result.importances_mean
        indices = importances.argsort()[::-1]

        # Filtrar variáveis com importâncias negativas ou próximas de zero
        positive_importance_indices = indices[importances[indices] > 0]
        sorted_feature_names = [X_train.columns[i] for i in positive_importance_indices]

        # Criar o gráfico de barras apenas com variáveis de importância positiva
        plt.figure(figsize=(10, 8))
        plt.barh(sorted_feature_names, importances[positive_importance_indices], align='center')
        plt.xlabel('Importância relativa')
        plt.title('Importância das Variáveis - KNN ')
        plt.gca().invert_yaxis()  # Variáveis mais importantes no topo
        plt.tight_layout()

        # Exibir o gráfico no Streamlit
        st.pyplot(plt)

        
        
        ################################################################
"""
     

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

    st.subheader("5.3. Comparativo Naive Bayes, Árvore de Decisão e Random Florest")
    with st.expander("", expanded=True):
        
        # 2. Preparar as features (X) e a variável alvo (y)
        X = dados_normalizados_final.drop(columns=['BP'])  # Supondo que 'alvo' já foi criado anteriormente
        #y = dados_normalizados_final['BP']
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
        
        if chkSmote:                        
            ######################################
            # Supondo que X e y já foram definidos antes no código
            # Aplicar SMOTE para balancear as classes
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            # Atualizando as variáveis X e y para os dados balanceados
            X_train = X_resampled
            y_train = y_resampled

                
        # Contar a quantidade de 0s e 1s no conjunto de treino e teste
        contagem_treino = y_train.value_counts()
        contagem_teste = y_test.value_counts()

        # Preparar os dados para o gráfico de barras
        df_comparativo = pd.DataFrame({
            'Conjunto': ['Treino', 'Treino', 'Teste', 'Teste'],
            'Classe': ['0', '1', '0', '1'],
            'Quantidade': [
                contagem_treino.get(0, 0),  # Contagem de 0s no treino
                contagem_treino.get(1, 0),  # Contagem de 1s no treino
                contagem_teste.get(0, 0),   # Contagem de 0s no teste
                contagem_teste.get(1, 0)    # Contagem de 1s no teste
            ]
        })

        # Criar o gráfico de barras
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(df_comparativo['Conjunto'] + ' - ' + df_comparativo['Classe'], df_comparativo['Quantidade'], color=['blue', 'orange', 'blue', 'orange'])
        ax.set_title('Comparação da Quantidade de 0s e 1s nos Conjuntos de Treino e Teste')
        ax.set_xlabel('Conjunto e Classe')
        ax.set_ylabel('Quantidade')
        plt.xticks(rotation=45)

        # Adicionar os números em cima das barras
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')

        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
                

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
        
        # Calcular a importância das variáveis usando permutation importance
        result = permutation_importance(modelo_nb, X_test, y_test, n_repeats=10, random_state=42)

        # Obter as importâncias e os nomes das variáveis
        importances = result.importances_mean

        # Filtrar para manter apenas importâncias positivas
        positive_indices = np.where(importances > 0)[0]
        positive_importances = importances[positive_indices]
        sorted_indices = positive_importances.argsort()[::-1]
        sorted_feature_names = [X_train.columns[i] for i in positive_indices[sorted_indices]]

        # Criar o gráfico de importância das variáveis
        plt.figure(figsize=(10, 8))
        plt.barh(sorted_feature_names, positive_importances[sorted_indices], align='center')
        plt.xlabel('Importância Relativa')
        plt.title('Importância das Variáveis - GaussianNB')
        plt.gca().invert_yaxis()  # Variáveis mais importantes no topo
        plt.tight_layout()

        # Exibir o gráfico no Streamlit
        st.pyplot(plt)

        st.info("Parametros de poda")

        max_depth = st.slider(
            "Profundidade Máxima da Árvore (max_depth)", 
            min_value=1, 
            max_value=20, 
            value=8, 
            step=1
        )

        st.write("Define o número mínimo de amostras que um nó deve ter antes de ser dividido em subnós. Se um nó tem menos amostras do que o valor de min_samples_split, ele não será dividido.")
        min_samples_split = st.slider(
            "Número Mínimo de Amostras para Dividir um Nó (min_samples_split)", 
            min_value=2, 
            max_value=20, 
            value=5, 
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
        
              
        
        importances = modelo_dt.feature_importances_
        feature_names = X_train.columns  # Nomes das variáveis do conjunto de treino

        # Filtrar apenas as variáveis com importância positiva
        positive_indices = np.where(importances > 0)[0]
        positive_importances = importances[positive_indices]
        positive_feature_names = feature_names[positive_indices]

        # Ordenar as importâncias e os nomes das variáveis com importância positiva
        sorted_indices = positive_importances.argsort()[::-1]
        sorted_positive_importances = positive_importances[sorted_indices]
        sorted_feature_names = [positive_feature_names[i] for i in sorted_indices]

        # Criar o gráfico
        plt.figure(figsize=(10, 10))
        plt.title('Importância das Variáveis - Árvore de Decisão (Sem Importância Negativa)')
        plt.barh(range(len(sorted_positive_importances)), sorted_positive_importances, align='center')
        plt.yticks(range(len(sorted_positive_importances)), sorted_feature_names)
        plt.xlabel('Importância Relativa')
        plt.ylabel('Variáveis')
        plt.gca().invert_yaxis()  # Variáveis mais importantes no topo
        st.pyplot(plt)
        
        from sklearn.ensemble import RandomForestClassifier
        
        st.info("Random Forest")
        
        max_depth2 = st.slider(
            "Profundidade Máxima da Árvore (max_depth) ", 
            min_value=1, 
            max_value=20, 
            value=8, 
            step=1
        )

        st.write("Define o número mínimo de amostras que um nó deve ter antes de ser dividido em subnós. Se um nó tem menos amostras do que o valor de min_samples_split, ele não será dividido.")
        min_samples_split2 = st.slider(
            "Número Mínimo de Amostras para Dividir um Nó (min_samples_split) ", 
            min_value=2, 
            max_value=20, 
            value=5, 
            step=1
        )

        st.write("Define o número mínimo de amostras que um nó folha (nó final da árvore) deve conter. Se um nó folha tem menos amostras do que esse valor, ele não é considerado uma folha e a divisão não é realizada.")
        min_samples_leaf2 = st.slider(
            "Número Mínimo de Amostras em uma Folha (min_samples_leaf) ", 
            min_value=1, 
            max_value=20, 
            value=5, 
            step=1
        )
        
        
        # 7. Treinar o terceiro modelo: Random Forest
        st.write("## Modelo Random Forest")
        modelo_rf = RandomForestClassifier(
            random_state=42,
            max_depth=max_depth,  # Profundidade máxima das árvores na floresta
            min_samples_split=min_samples_split,  # Min amostras para dividir ajustado pelo slider
            min_samples_leaf=min_samples_leaf,  # Min amostras em folha ajustado pelo slider
            n_estimators=100  # Número de árvores na floresta
        )
        modelo_rf.fit(X_train, y_train)
        y_pred_rf = modelo_rf.predict(X_test)

        # Avaliar o modelo Random Forest
        acuracia_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"Acurácia do modelo Random Forest: {acuracia_rf:.2f}")
        st.write("Relatório de Classificação (Random Forest):")
        st.text(classification_report(y_test, y_pred_rf))

        # Exibir a matriz de confusão para Random Forest
        st.write("Matriz de Confusão (Random Forest):")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        fig_rf, ax_rf = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax_rf)
        ax_rf.set_title("Matriz de Confusão - Random Forest")
        ax_rf.set_xlabel("Classe Prevista")
        ax_rf.set_ylabel("Classe Real")
        st.pyplot(fig_rf)

       
        
        
        
        #st.markdown(texto)
        
    st.subheader("5.4. Comparativo")
    with st.expander("Conclusões e informação/interpretação dos dados ", expanded=True):
        

         # 8. Comparar os resultados dos três modelos
        st.write("## Comparação de Acurácia")
        st.write(f"Acurácia Naive Bayes: {acuracia_nb:.2f}")
        st.write(f"Acurácia Árvore de Decisão: {acuracia_dt:.2f}")
        st.write(f"Acurácia Random Forest: {acuracia_rf:.2f}")

        # Exibir o modelo com melhor desempenho
        if acuracia_nb > acuracia_dt and acuracia_nb > acuracia_rf:
            st.success("O modelo Naive Bayes teve melhor desempenho!")
        elif acuracia_dt > acuracia_nb and acuracia_dt > acuracia_rf:
            st.success("O modelo Árvore de Decisão teve melhor desempenho!")
        elif acuracia_rf > acuracia_nb and acuracia_rf > acuracia_dt:
            st.success("O modelo Random Forest teve melhor desempenho!")
        else:
            st.info("Dois ou mais modelos tiveram o mesmo desempenho!")
            
        from sklearn.metrics import roc_curve, auc
        
       


        
        # 2. Obter as probabilidades previstas para a classe positiva
        y_pred_proba_rf = modelo_rf.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva

        # 3. Calcular a curva ROC e a AUC para o Random Forest
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)

        # 4. Repetir para Naive Bayes e Árvore de Decisão (já está no seu código)
        y_pred_proba_nb = modelo_nb.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva
        fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_proba_nb)
        roc_auc_nb = auc(fpr_nb, tpr_nb)

        y_pred_proba_dt = modelo_dt.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva
        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
        roc_auc_dt = auc(fpr_dt, tpr_dt)

        # 5. Plotar as curvas ROC para todos os modelos
        plt.figure()

        # Curva ROC Naive Bayes
        plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')

        # Curva ROC Árvore de Decisão
        plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label=f'Árvore de Decisão (AUC = {roc_auc_dt:.2f})')

        # Curva ROC Random Forest
        plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
        
        
        # Linha diagonal (referência para classificação aleatória)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Comparação das Curvas ROC')
        plt.legend(loc='lower right')
        st.pyplot(plt)
        
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
    st.sidebar.markdown("*Professores*: **Luciana Conceição Dias Campos, Heder Soares Bernardino**")
    st.sidebar.write("")
    
    

# Executar a aplicação Streamlit
if __name__ == '__main__':
    main()

