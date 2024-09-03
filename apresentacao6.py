import streamlit as st
import pandas as pd

# Configuração inicial
st.set_page_config(page_title="Trabalho 6 - Definição do Problema - Mineração de Dados", layout="wide")

# Título
st.title("Trabalho 6 - Definição do Problema")
st.write("**Autor:** Rodrigo Mendes Peixoto | **Disciplina:** Mineração de Dados | **Data:** 03/09/2024")
st.markdown(""""
            <img src="https://www2.ufjf.br/pgcc/wp-content/uploads/sites/181/2020/10/8200x1417-300x300.png" alt="Imagem da UFJF" width="5%">
            [Programa de Pós-Graduação em Ciências da COmputação - UFJF](https://www2.ufjf.br/pgcc/)            
            """, unsafe_allow_html=True)
st.info("**Professores**: Luciana Conceição Dias Campos, Heder Soares Bernardino")


dados = pd.read_csv('data/iot_device_train.csv')

st.header("Base de Dados")
st.write("""
- **Descrição:** Conjunto de dados de tráfego de rede coletado de dispositivos IoT.
- **Link:** [Acessar Conjunto de Dados no Kaggle](https://www.kaggle.com/datasets/fanbyprinciple/iot-device-identification/data)
""")

st.write(dados.head(), expander=True)
infoRownsColumns = str(dados.shape[0]) + " x " + str(dados.columns.shape[0])
st.info(infoRownsColumns)

# Slide 1: Introdução
st.header("Introdução")
st.write("""
A crescente adoção de dispositivos IoT apresenta desafios significativos para a segurança das redes. 
Esta apresentação explora como a análise de tráfego de rede pode ser usada para identificar dispositivos IoT, 
garantindo uma proteção mais eficaz contra ameaças como ataques DDoS.
""")

# Slide 2: Motivação
st.header("Motivação")
st.write("""
- **Segurança:** A detecção precisa de dispositivos IoT é crucial para prevenir ataques, como DDoS, causados por dispositivos comprometidos.
- **Disponibilidade:** Em setores críticos, como o de saúde, a disponibilidade dos dispositivos IoT é vital. Dispositivos ligados a saúde precisam estar continuamente operacionais e protegidos contra falhas e invasões.
- **Problema Atual:** Redes com uma grande variedade de dispositivos, incluindo os de saúde, exigem técnicas robustas de identificação para garantir tanto a segurança quanto a disponibilidade desses dispositivos, minimizando o risco de interrupções que poderiam comprometer a segurança do paciente.
""")

# Slide 3: Definição do Problema
st.header("Definição do Problema")
st.write("""
- **Tipo de Problema:** Classificação dos dispositivos IoT com base em seus padrões de tráfego de rede.
- **Atributos Disponíveis:** Tamanho dos pacotes, tempos de chegada entre pacotes, entre outros.
- **Objetivo:** Desenvolver um modelo que consiga reconhecer dispositivos com alta precisão.
""")

# Slide 4: Metodologia
st.header("Metodologia")
st.write("""
Foram testados vários algoritmos de machine learning, incluindo Random Forest, Decision Tree, e SVM. 
O melhor desempenho foi alcançado pelo Random Forest, com uma acurácia de 99,9%.
""")

# Slide 5: Artigo de Referência
st.header("Artigo de Referência")
st.write("""
- **Título:** IoT Devices Recognition Through Network Traffic Analysis
- **Autores:** Mustafizur R. Shahid, Gregory Blanc, Zonghua Zhang, Hervé Debar
- **Fonte:** IEEE International Conference on Big Data, 2018
""")
st.markdown("[Link do artigo](https://ieeexplore-ieee-org.ez25.periodicos.capes.gov.br/abstract/document/8622243/authors#authors)")

# Slide: Dispositivos IoT Utilizados
st.subheader("Dispositivos IoT Utilizados")
st.write("""
O estudo foi conduzido em uma rede experimental composta por cinco dispositivos IoT, incluindo uma câmera de segurança Nest, um sensor de movimento D-Link, uma lâmpada inteligente TP-Link e um plugue inteligente TP-Link, representando uma variedade de comportamentos de rede.
""")

# Slide: Conjunto de Dados
st.subheader("Conjunto de Dados")
st.write("""
Os dados foram coletados ao longo de sete dias, resultando em um conjunto de treino com 3.222 amostras e um conjunto de teste com 805 amostras, garantindo uma base robusta para a análise e validação dos modelos de classificação.
""")

# Slide: Análise de Dados
st.subheader("Análise de Dados")
st.write("""
A análise dos dados foi conduzida utilizando as bibliotecas scikit-learn e TensorFlow, permitindo a implementação eficiente de diversas técnicas de machine learning para a classificação dos dispositivos IoT com base nos padrões de tráfego de rede.
""")

# Slide: Dimensionalidade dos Dados
st.subheader("Dimensionalidade dos Dados")
st.write("""
Cada fluxo de rede foi descrito por um vetor de 38 dimensões, incluindo características como o tamanho dos primeiros pacotes enviados e recebidos e os tempos de chegada entre esses pacotes, oferecendo uma visão detalhada dos comportamentos de rede dos dispositivos IoT.
""")

# Slide: Redução de Dimensionalidade com T-SNE
st.subheader("Redução de Dimensionalidade com T-SNE")
st.write("""
Para facilitar a visualização dos dados e explorar o poder discriminativo das características selecionadas, foi aplicada a técnica de redução de dimensionalidade t-SNE, que destacou a capacidade dos dados em diferenciar os diversos dispositivos IoT.
""")

# Slide: Modelos de Classificação Utilizados
st.subheader("Modelos de Classificação Utilizados")
st.write("""
Foram testados seis algoritmos de classificação distintos: Random Forest, Decision Tree, SVM, k-Nearest Neighbors (KNN), Artificial Neural Network (ANN) e Naïve Bayes. O Random Forest apresentou o melhor desempenho, alcançando uma acurácia de 99,9% na classificação dos dispositivos IoT.
""")

# Slide 6: Base de Dados
st.header("Base de Dados")
st.write("""
- **Descrição:** Conjunto de dados de tráfego de rede coletado de dispositivos IoT em uma rede doméstica.
""")

# Slide 7: Conclusão
st.header("Conclusão")
st.write("""
A identificação de dispositivos IoT é crítica para a segurança das redes, e a análise de tráfego oferece uma abordagem eficaz. 
Próximos passos incluem testar o modelo em redes maiores com mais dispositivos IoT.
""")

# Slide 8: Agradecimentos e Perguntas
st.header("Agradecimentos e Perguntas")
st.write("Obrigado pela atenção! Estou aberto para perguntas.")

