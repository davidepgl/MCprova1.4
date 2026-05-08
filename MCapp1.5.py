import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Monte Carlo PRO 2026", layout="wide")

# --- MOTORE DI DOWNLOAD A DOPPIA SORGENTE (ANTI-BLOCCO) ---
@st.cache_data(ttl=86400)
def get_live_data():
    # TENTATIVO 1: YAHOO FINANCE
    try:
        data = yf.download(["SPY", "TLT"], start="2002-01-01", interval="1mo", auto_adjust=True, threads=False)
        if not data.empty and len(data) > 10:
            if isinstance(data.columns, pd.MultiIndex): data = data['Close']
            returns = data.pct_change().dropna()
            returns.columns = ["SPY", "TLT"]
            return returns
    except:
        pass

    # TENTATIVO 2: STOOQ (Fallback per Cloud)
    try:
        df_spy = web.DataReader('^SPX', 'stooq', start='2002-01-01').resample('M').last()
        df_tlt = web.DataReader('TLT.US', 'stooq', start='2002-01-01').resample('M').last()
        data = pd.concat([df_spy['Close'], df_tlt['Close']], axis=1).sort_index()
        data.columns = ["SPY", "TLT"]
        return data.pct_change().dropna()
    except:
        return None

live_returns = get_live_data()

# --- MOTORE MONTE CARLO ---
def run_simulation(capitale, prelievo_pct, equity_pct, anni, ter, n_sim, extra_expenses, mode, params=None):
    mesi = int(anni * 12)
    prelievo_mensile = (capitale * (prelievo_pct / 100)) / 12
    costi_mensili = (ter / 100) / 12
    
    spese_pianificate = np.zeros(mesi + 1)
    for _, row in extra_expenses.iterrows():
        m = int(row['Anno'] * 12)
        if 0 <= m <= mesi: spese_pianificate[m] += row['Importo (€)']

    if mode == "Bootstrap (LIVE DATA)" and live_returns is not None:
        idx = np.random.randint(0, len(live_returns), size=(mesi, n_sim))
        h_spy, h_tlt = live_returns['SPY'].values[idx], live_returns['TLT'].values[idx]
    else:
        m_s, s_s = params['m_spy']/12, params['s_spy']/np.sqrt(12)
        m_t, s_t = params['m_tlt']/12, params['s_tlt']/np.sqrt(12)
        cov = params['corr'] * s_s * s_t
        rets = np.random.multivariate_normal([m_s, m_t], [[s_s**2, cov], [cov, s_t**2]], size=(mesi, n_sim))
        h_spy, h_tlt = rets[:,:,0], rets[:,:,1]

    port_returns = (h_spy * equity_pct) + (h_tlt * (1 - equity_pct))
    percorsi = np.zeros((mesi + 1, n_sim))
    percorsi[0] = capitale
    
    for t in range(mesi):
        val = percorsi[t] * (1 + port_returns[t] - costi_mensili) - prelievo_mensile - spese_pianificate[t+1]
        val[val < 0] = 0
        percorsi[t+1] = val
    return percorsi

# --- INTERFACCIA ---
st.title("🏛️ Diagnosi Finanziaria Avanzata Monte Carlo")

with st.sidebar:
    st.header("📡 Connessione Dati")
    if live_returns is not None:
        st.success(f"Dati Live Collegati: {live_returns.index[-1].strftime('%m/%Y')}")
        with st.expander("Vedi Storico"): st.line_chart((1 + live_returns).cumprod()*100)
    else:
        st.error("Dati Live Offline. Solo Parametrica disponibile.")

    st.divider()
    sim_mode = st.radio("Sorgente:", ["Bootstrap (LIVE DATA)", "Parametrica (Gaussiana Custom)"], 
                        index=0 if live_returns is not None else 1)
    
    param_dict = {}
    if sim_mode == "Parametrica (Gaussiana Custom)":
        st.subheader("⚙️ Ipotesi Asset")
        m_spy = st.number_input("Equity Rend %", value=8.5) / 100
        s_spy = st.number_input("Equity Vol %", value=16.0) / 100
        m_tlt = st.number_input("Bond Rend %", value=3.5) / 100
        s_tlt = st.number_input("Bond Vol %", value=8.0) / 100
        corr = st.slider("Correlazione", -1.0, 1.0, -0.1)
        param_dict = {'m_spy': m_spy, 's_spy': s_spy, 'm_tlt': m_tlt, 's_tlt': s_tlt, 'corr': corr}

    st.divider()
    cap = st.number_input("Capitale (€)", value=1000000)
    prel = st.slider("Prelievo Annuo %", 0.0, 15.0, 4.0)
    eq = st.slider("Equity %", 0.0, 1.0, 0.6)
    yrs = st.slider("Anni", 1, 50, 30)
    ter = st.slider("Costi TER %", 0.0, 5.0, 1.5)
    sim = st.selectbox("Simulazioni", [10000, 50000, 100000], index=1)
    
    st.header("Uscite Straordinarie")
    df_extra = pd.DataFrame([{"Anno": 10, "Importo (€)": 0.0}])
    edited_df = st.data_editor(df_extra, num_rows="dynamic", use_container_width=True)
    
    btn = st.button("ESEGUI ANALISI", type="primary", use_container_width=True)

if btn:
    with st.spinner('Simulazione di 100.000 scenari in corso...'):
        dati = run_simulation(cap, prel, eq, yrs, ter, sim, edited_df, sim_mode, param_dict)
    
    p_levels = [5, 10, 25, 50, 75, 90, 95]
    pct = {p: np.percentile(dati, p, axis=1) for p in p_levels}
    successo = np.mean(dati[-1, :] > 0) * 100

    # KPI TOP
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilità Successo", f"{successo:.1f}%")
    c2.metric("Netto Annuo (Est.)", f"€ {(cap*(prel/100)*0.74):,.0f}")
    c3.metric("Netto Mensile", f"€ {(cap*(prel/100)*0.74)/12:,.0f}")
    c4.metric("Scenari", f"{sim:,}")

    # SCENARI CHIAVE
    st.subheader("📊 Analisi degli Scenari")
    s1, s2, s3 = st.columns(3)
    for col, p, lab in zip([s1, s2, s3], [10, 50, 90], ["P10 (Stress Test)", "P50 (Scenario Base)", "P90 (Ottimista)"]):
        with col:
            st.markdown(f"### {lab}")
            st.write(f"Capitale Finale: **€ {pct[p][-1]:,.0f}**")
            if pct[p][-1] == 0:
                st.error(f"Esaurimento: Anno {np.where(pct[p] == 0)[0][0] // 12}")
            else: st.success("Esaurimento: Mai")
            st.write(f"Minimo toccato: € {np.min(pct[p]):,.0f}")

    # FAN CHART
    st.subheader("📈 Proiezione Probabilistica del Capitale")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t_range = np.arange(yrs * 12 + 1)
    ax.fill_between(t_range, pct[5], pct[95], color='royalblue', alpha=0.1, label='Range 90% (P5-P95)')
    ax.fill_between(t_range, pct[25], pct[75], color='royalblue', alpha=0.3, label='Range 50% (P25-P75)')
    ax.plot(t_range, pct[50], color='navy', linewidth=2, label='Mediana (P50)')
    ax.plot(t_range, pct[10], color='red', linestyle='--', label='Stress Test (P10)')
    ax.set_ylabel("Capitale (€)")
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # TABELLA TEMPORALE
    st.subheader("📅 Tabella Evoluzione Temporale (Percentili)")
    step = 5 if yrs > 15 else 1
    idx_annuali = np.arange(0, (yrs * 12) + 1, step * 12)
    df_tab = pd.DataFrame({f"P{p}": pct[p][idx_annuali] for p in p_levels}, 
                          index=[f"Anno {i//12}" for i in idx_annuali])
    st.dataframe(df_tab.style.format("{:,.0f}"), use_container_width=True)
    
    # AVVISO FINALE
    if successo < 80:
        st.warning(f"Il piano ha un rischio elevato. Solo {successo:.1f} scenari su 100 finiscono con successo.")