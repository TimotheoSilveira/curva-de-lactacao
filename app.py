
import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Modelo da Curva de Lactação (Modelo de Wood) ---
def woods_model(t, a, b, c):
    epsilon = 1e-9
    return a * (t + epsilon)**b * np.exp(-c * (t + epsilon))

# --- 2. Interface do Streamlit ---
st.set_page_config(layout="wide")

# --- Logomarca da Alta + Título ---
logo = Image.open("Logo Alta_azul.png")
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image(logo, width=150)
with col_title:
    st.title("Análise da Curva de Lactação – Alta Genetics")

st.markdown("""
Este aplicativo analisa dados de produção de leite utilizando o **Modelo de Lactação de Wood** para calcular indicadores-chave de desempenho ao longo do ciclo de lactação.
""")

# --- 3. Barra Lateral ---
with st.sidebar:
    st.header("⚙️ Configurações")
    lactation_length = st.number_input("Duração Padrão da Lactação (dias)", min_value=100, max_value=500, value=305, step=5)
    
    st.header("📋 Insira seus Dados")
    st.markdown("Cole abaixo os dados no formato `Dia,Produção`. Cada linha deve conter um par Dia/Produção.")

    dados_exemplo = """15,25.5
30,35.1
45,40.2
60,42.5
75,41.8
90,40.1
120,38.5
150,36.2
180,34.0
210,31.5
240,29.1
270,26.8
300,24.5"""

    data_input = st.text_area("Dados de Produção de Leite (Dia, Produção)", value=dados_exemplo, height=300)

# --- 4. Painel Principal ---
if st.button("📈 Analisar Curva de Lactação"):
    try:
        linhas = data_input.strip().split('\n')
        if len(linhas) < 5:
            st.error("Por favor, insira ao menos 5 pontos para uma análise confiável.")
        else:
            dados = []
            for linha in linhas:
                dia, producao = linha.split(',')
                dados.append([int(dia), float(producao)])
            
            df = pd.DataFrame(dados, columns=['DIM', 'Produção'])
            dim = df['DIM'].values
            producao_leite = df['Produção'].values

            chutes_iniciais = [15, 0.2, 0.003]
            popt, pcov = curve_fit(woods_model, dim, producao_leite, p0=chutes_iniciais, maxfev=10000)
            a, b, c = popt

            tempo_pico = b / c
            pico_producao = woods_model(tempo_pico, a, b, c)
            producao_total, _ = quad(woods_model, 1, lactation_length, args=(a, b, c))
            producao_dia_250 = woods_model(250, a, b, c)
            persistencia = (producao_dia_250 / pico_producao) * 100 if pico_producao > 0 else 0

            st.success("✅ Análise Concluída!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Indicadores de Desempenho")
                st.metric(label="Pico de Produção Diária", value=f"{pico_producao:.2f} kg/dia")
                st.metric(label="Tempo até o Pico", value=f"{tempo_pico:.1f} dias")
                st.metric(label=f"Produção Total em {lactation_length} dias", value=f"{producao_total:.0f} kg")
                st.metric(label="Persistência da Lactação", value=f"{persistencia:.1f} %", help="Mede a capacidade de manter a produção após o pico. Quanto maior, melhor.")

            with col2:
                st.subheader("🔬 Parâmetros do Modelo")
                st.info(f"""
                Modelo utilizado: **Y(t) = a * t^b * e^(-ct)**

                - **Parâmetro 'a'**: {a:.4f} (Fator de escala inicial)
                - **Parâmetro 'b'**: {b:.4f} (Taxa de inclinação pré-pico)
                - **Parâmetro 'c'**: {c:.4f} (Taxa de queda pós-pico)
                """)

            st.subheader("📈 Visualização da Curva de Lactação")
            t_suave = np.linspace(1, lactation_length, 305)
            y_suave = woods_model(t_suave, a, b, c)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(dim, producao_leite, label='Dados Originais', color='blue', zorder=5)
            ax.plot(t_suave, y_suave, label="Curva Ajustada (Wood)", color='red', linewidth=2)
            ax.axvline(tempo_pico, color='green', linestyle='--', label=f'Pico aos {tempo_pico:.1f} dias')
            ax.axhline(pico_producao, color='orange', linestyle='--', label=f'Pico: {pico_producao:.2f} kg')

            ax.set_title('Análise da Curva de Lactação', fontsize=16)
            ax.set_xlabel('Dias em Lactação (DIM)', fontsize=12)
            ax.set_ylabel('Produção Diária (kg)', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        st.warning("Verifique se os dados estão corretamente formatados (ex: `30,25.5`) com uma linha por ponto e ao menos 5 registros.")
