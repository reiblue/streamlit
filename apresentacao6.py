import streamlit as st
import pandas as pd

# Configuração inicial
st.set_page_config(page_title="Reconhecimento de Dispositivos IoT", layout="wide")

# Título
st.title("Reconhecimento de Dispositivos IoT Através da Análise de Tráfego de Rede")
st.write("**Autor:** Rodrigo Mendes Peixoto | **Disciplina:** Mineração de Dados | **Data:** 03/09/2024")
st.info("**Professores**: Luciana Conceição Dias Campos, Heder Soares Bernardino")

dados = pd.read_csv('data/iot_device_train.csv')

st.write(dados.head(), expander=True)
st.info(dados.shape[0] + " x " + dados.columns.shape[0])

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
- **Disponibilidade:** Em setores críticos, como o de saúde, a disponibilidade dos dispositivos IoT é vital. Dispositivos como monitores cardíacos, bombas de insulina e outros equipamentos de suporte à vida precisam estar continuamente operacionais e protegidos contra falhas e invasões.
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

# Slide 6: Base de Dados
st.header("Base de Dados")
st.write("""
- **Descrição:** Conjunto de dados de tráfego de rede coletado de dispositivos IoT em uma rede doméstica.
- **Link (opcional):** Incluir se aplicável
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

