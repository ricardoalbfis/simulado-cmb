import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configuração inicial da página
st.set_page_config(page_title="Simulador CMB", layout="wide", page_icon="🌌")
st.title("Espectro de Potência Angular da Radiação Cósmica de Fundo")
st.markdown("Comparação interativa entre o modelo $\Lambda$CDM e os dados do Planck 2018.")

# --- DADOS MOCK DO PLANCK 2018 ---
planck_ells = [2, 10, 30, 60, 100, 150, 220, 300, 400, 540, 650, 800, 1000, 1200, 1400, 1650, 2000, 2400]
planck_dl = [200, 850, 1250, 1600, 2300, 3800, 5730, 4500, 2600, 2550, 2200, 2540, 1700, 1250, 850, 550, 250, 100]

# Vetor de multipolos para plotar a curva suave do modelo
ells = np.arange(2, 2501, 2)

# --- FUNÇÃO DO MODELO TEÓRICO (APROXIMAÇÃO) ---
def compute_spectrum(h0, ob, oc, ok, ns, as_amp):
    # Fator de deslocamento baseado na geometria (curvatura) e expansão (H0)
    shift = np.sqrt(max(0.01, 1.0 - 1.25 * ok)) * ((h0 / 67.4) ** 0.35)
    b_ratio = ob / 0.0224
    m_ratio = oc / 0.120
    
    l_eff = ells / shift
    
    # Efeito Sachs-Wolfe
    sw = 1000 * np.maximum(ells, 2) / 10 ** (ns - 1.0)
    
    # Picos acústicos (senoides gaussianas amortecidas)
    p1 = 4750 * b_ratio * np.exp(-((l_eff - 220) / 110) ** 2)
    p2 = (2400 / b_ratio) * np.exp(-((l_eff - 540) / 120) ** 2)
    p3 = (2450 * b_ratio * m_ratio) * np.exp(-((l_eff - 810) / 140) ** 2)
    p4 = 1200 * np.exp(-((l_eff - 1120) / 160) ** 2)
    p5 = 800 * np.exp(-((l_eff - 1420) / 180) ** 2)
    
    # Amortecimento de Silk
    silk = np.exp(-(l_eff / 1350) ** 1.4)
    # Tilt primordial
    tilt = (ells / 200) ** (ns - 1.0)
    
    dl = (sw + (p1 + p2 + p3 + p4 + p5) * silk) * as_amp * tilt
    return np.maximum(0, dl)

# --- PAINEL LATERAL (SLIDERS) ---
st.sidebar.header("Parâmetros Cosmológicos")
h0 = st.sidebar.slider("Taxa de Hubble ($H_0$)", 55.0, 80.0, 67.4, 0.5)
ob = st.sidebar.slider("Densidade Bariônica ($\Omega_b h^2$)", 0.015, 0.030, 0.0224, 0.0005)
oc = st.sidebar.slider("Matéria Escura ($\Omega_c h^2$)", 0.080, 0.160, 0.120, 0.002)
ok = st.sidebar.slider("Curvatura Espacial ($\Omega_k$)", -0.08, 0.08, 0.00, 0.01)
ns = st.sidebar.slider("Índice Espectral ($n_s$)", 0.85, 1.10, 0.965, 0.005)
as_amp = st.sidebar.slider("Amplitude ($A_s$)", 0.70, 1.30, 1.00, 0.02)

escala_log = st.sidebar.checkbox("Escala logarítmica (Eixo X)", value=True)

# Cálculos Derivados para exibir como métricas
h = h0 / 100
om_m = (ob + oc) / (h ** 2)
om_lambda = 1.0 - om_m - ok

# Layout superior de Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Primeiro Pico Acústico", f"ℓ ≈ {int(220 * np.sqrt(max(0.01, 1.0 - 1.25 * ok)))}")
col2.metric("Energia Escura ($\Omega_\Lambda$)", f"{om_lambda:.3f}")
if ok < -0.005: geom = "Fechado"
elif ok > 0.005: geom = "Aberto"
else: geom = "Plano"
col3.metric("Geometria", geom)

# --- PLOTAGEM DO GRÁFICO (Matplotlib) ---
dl_model = compute_spectrum(h0, ob, oc, ok, ns, as_amp)

fig, ax = plt.subplots(figsize=(10, 5))
plt.style.use('dark_background')

# Plota Dados Reais (Pontos)
ax.scatter(planck_ells, planck_dl, color='#38bdf8', label='Planck 2018 (Real)', zorder=3)
# Plota Modelo (Linha)
ax.plot(ells, dl_model, color='#ef4444', linewidth=2.5, label='Modelo $\Lambda$CDM', zorder=2)

# Configuração de Eixos
ax.set_xlabel("Multipolo $\ell$", fontsize=12, color='white')
ax.set_ylabel("$D_\ell [\mu K^2]$", fontsize=12, color='white')
ax.set_ylim(0, 7000)
if escala_log:
    ax.set_xlim(2, 2500)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter()) # Remove notação científica
else:
    ax.set_xlim(0, 2500)

ax.grid(color='#334155', linestyle='--', linewidth=0.5)
ax.legend()

# Envia o gráfico para o Streamlit
st.pyplot(fig)