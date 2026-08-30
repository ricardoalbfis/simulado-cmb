import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import camb

# Configuração da página
st.set_page_config(page_title="Simulador CMB - Motor CAMB", layout="wide")
st.title("Espectro de Potência da CMB (Resolvido via CAMB)")
st.markdown("Este painel resolve numericamente as equações de Boltzmann usando o pacote oficial `camb`.")

# --- 1. DEFINIÇÃO DO MODELO DE REFERÊNCIA (PLANCK 2018) ---
# Usamos o st.cache_data para rodar essa integração pesada apenas uma vez!
@st.cache_data
def get_planck_baseline():
    pars = camb.CAMBparams()
    # Parâmetros cosmológicos de melhor ajuste (ΛCDM - Planck 2018)
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, omk=0.0)
    pars.InitPower.set_params(As=2.1e-9, ns=0.965)
    
    # Calculando os espectros até l=2500 (desligamos lenteamento para agilizar o cálculo online)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    
    # Extrai o vetor de multipolos (ls) e o espectro de temperatura (TT)
    totCL = powers['total']
    ls = np.arange(totCL.shape[0])
    return ls, totCL[:, 0]

# Carrega o baseline
ls_base, cl_base = get_planck_baseline()

# --- 2. FUNÇÃO PARA O MODELO DINÂMICO DO USUÁRIO ---
def get_custom_spectrum(H0, ombh2, omch2, omk, ns, As_mult):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk)
    # A amplitude As é multiplicada pela base 2.1e-9
    pars.InitPower.set_params(As=2.1e-9 * As_mult, ns=ns)
    
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    return powers['total'][:, 0]

# --- 3. PAINEL LATERAL (SLIDERS) ---
st.sidebar.header("Variáveis do Modelo")
st.sidebar.markdown("*O tempo de cálculo é de ~1 segundo por ajuste.*")

h0 = st.sidebar.slider("Taxa de Hubble ($H_0$)", 50.0, 85.0, 67.4, 0.5)
ob = st.sidebar.slider("Dens. Bariônica ($\Omega_b h^2$)", 0.010, 0.040, 0.0224, 0.001)
oc = st.sidebar.slider("Matéria Escura ($\Omega_c h^2$)", 0.050, 0.250, 0.120, 0.005)
ok = st.sidebar.slider("Curvatura ($\Omega_k$)", -0.05, 0.05, 0.00, 0.01)
ns = st.sidebar.slider("Índice Espectral ($n_s$)", 0.85, 1.10, 0.965, 0.01)
as_amp = st.sidebar.slider("Amplitude ($A_s$) [Fator]", 0.50, 1.50, 1.00, 0.05)

escala_log = st.sidebar.checkbox("Escala logarítmica (Eixo X)", value=True)

# Calcula as frações de energia atuais para exibir como métricas
h = h0 / 100
om_m = (ob + oc) / (h**2)
om_lambda = 1.0 - om_m - ok

col1, col2 = st.columns(2)
col1.metric("Matéria Total ($\Omega_m$)", f"{om_m:.3f}")
col2.metric("Energia Escura ($\Omega_\Lambda$)", f"{om_lambda:.3f}")

# --- 4. PLOTAGEM DO GRÁFICO ---
with st.spinner('Resolvendo equações de Boltzmann via CAMB...'):
    cl_custom = get_custom_spectrum(h0, ob, oc, ok, ns, as_amp)

fig, ax = plt.subplots(figsize=(10, 5))
plt.style.use('dark_background')

# Plota o Baseline Fixo (Tracejado Cinza/Branco)
ax.plot(ls_base[2:2501], cl_base[2:2501], color='#94a3b8', linestyle='--', linewidth=2, label='Fixo: ΛCDM Planck 2018', zorder=1)

# Plota o Modelo Modificado (Vermelho Vibrante)
ax.plot(ls_base[2:2501], cl_custom[2:2501], color='#ef4444', linewidth=2.5, label='Modelo Modificado', zorder=2)

# Configuração de Eixos
ax.set_xlabel("Multipolo $\ell$", fontsize=12)
ax.set_ylabel("$\ell(\ell+1)C_\ell / 2\pi \quad [\mu K^2]$", fontsize=12)

if escala_log:
    ax.set_xlim(2, 2500)
    ax.set_xscale('log')
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
else:
    ax.set_xlim(2, 2500)

ax.set_ylim(0, 7500)
ax.grid(color='#334155', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right')

st.pyplot(fig)
