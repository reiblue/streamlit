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

# Slide 5: Artigo de Referência
st.header("Artigo de Referência")
st.write("""
- **Título:** IoT Devices Recognition Through Network Traffic Analysis
- **Autores:** Mustafizur R. Shahid, Gregory Blanc, Zonghua Zhang, Hervé Debar
- **Fonte:** IEEE International Conference on Big Data, 2018
""")
st.markdown("[Link do artigo](https://ieeexplore-ieee-org.ez25.periodicos.capes.gov.br/abstract/document/8622243)")

# Título
st.title("Metodologia Proposta")

# Introdução e Definição do Problema
st.header("Introdução e Definição do Problema")
st.write("""
- **Objetivo:** Desenvolver e validar um modelo de machine learning capaz de identificar dispositivos IoT com base em seus padrões de tráfego de rede, como proposto no artigo de referência.
- **Problema:** O foco será na classificação de dispositivos IoT utilizando técnicas de análise de tráfego, com o objetivo de melhorar a segurança e a gestão de redes IoT.
""")

# Coleta de Dados
st.header("Coleta de Dados")
st.write("""
- **Fonte dos Dados:** Serão utilizados dados de tráfego de rede coletados Kaggle.
- **Descrição dos Dados:** Os dados incluirão atributos como tamanho dos pacotes, tempos de chegada entre pacotes, e características específicas de cada dispositivo, conforme descrito no artigo de referência.
""")

# Pré-processamento dos Dados
st.header("Pré-processamento dos Dados")
st.write("""
- **Limpeza dos Dados:** Serão realizadas etapas de limpeza dos dados para remover inconsistências e tratar valores ausentes.
- **Engenharia de Atributos:** Atributos relevantes serão extraídos e transformados, como proposto no artigo, para melhorar a capacidade discriminativa do modelo.
""")

# Redução de Dimensionalidade
st.header("Redução de Dimensionalidade")
st.write("""
- **Aplicação do t-SNE:** A técnica t-SNE será utilizada para visualizar e reduzir a dimensionalidade dos dados, conforme descrito no artigo, permitindo uma melhor compreensão dos padrões de tráfego associados a cada dispositivo.
""")

# Modelagem
st.header("Modelagem")
st.write("""
- **Seleção de Modelos:** Serão testados vários algoritmos de machine learning, incluindo Random Forest, Decision Tree, SVM, k-Nearest Neighbors (KNN), Artificial Neural Networks (ANN), e Naïve Bayes.
""")

# Implementação de Segurança Avançada
st.header("Calssificação de anomalias")
st.write("""
- **Detecção de Anomalias:** Além da classificação, será explorada a implementação de mecanismos de detecção de anomalias para identificar comportamentos suspeitos nos dispositivos IoT em tempo real.
""")

# Resultados e Discussão
st.header("Resultados e Discussão")
st.write("""
- **Análise dos Resultados:** Os resultados serão analisados em comparação com os achados do artigo de referência, destacando as melhorias e os desafios encontrados.
- **Escalabilidade:** A viabilidade do modelo em redes maiores e mais complexas será discutida, propondo futuras extensões e ajustes necessários para sua aplicação em ambientes reais.
""")

# Conclusão
st.header("Conclusão")
st.write("""
- **Resumo dos Achados:** A metodologia permitirá a criação de um modelo robusto para a identificação de dispositivos IoT, contribuindo para a segurança e a gestão de redes IoT.
- **Trabalhos Futuros:** Sugestões serão feitas para melhorias e futuras pesquisas, incluindo a adaptação do modelo a diferentes tipos de redes e dispositivos IoT, além da integração com outras soluções de segurança.
""")

# Slide 7: Conclusão
st.header("Conclusão")
st.write("""
A identificação de dispositivos IoT é crítica para a segurança das redes, e a análise de tráfego oferece uma abordagem eficaz. 
Próximos passos incluem testar o modelo em redes maiores com mais dispositivos IoT.
""")

# Slide 8: Agradecimentos e Perguntas
st.header("Agradecimentos e Perguntas")
st.write("Obrigado pela atenção!")

