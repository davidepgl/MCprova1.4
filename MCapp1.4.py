import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Monte Carlo PRO - Live Data", layout="wide")

# --- FUNZIONE DOWNLOAD LIVE (STILE "CACCIA AL TESORO") ---
@st.cache_data(ttl=3600)
def get_truly_live_data():
    tickers = ["SPY", "TLT"]
    try:
        # Metodo 1: Download Standard ma con parametri anti-blocco
        data = yf.download(
            tickers=tickers,
            start="2002-01-01",
            interval="1mo",
            auto_adjust=True,
            threads=False # Fondamentale per non essere bannati dai server
        )
        
        # Se il download è vuoto, proviamo il Metodo 2 (Singoli Ticker)
        if data.empty or len(data) < 5:
            df_list = []
            for t in tickers:
                tmp = yf.download(t, start="2002-01-01", interval="1mo", auto_adjust=True, threads=False)
                if not tmp.empty:
                    df_list.append(tmp[['Close']].rename(columns={'Close': t}))
            data = pd.concat(df_list, axis=1)

        # Pulizia Multi-Index (se presente)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns:
                data = data['Close']
            else:
                data = data.iloc[:, :len(tickers)]

        data = data.ffill().dropna()
        
        # Rinominia e ordine colonne
        returns = data.pct_change().dropna()
        returns.columns = ["SPY", "TLT"]
        return returns
        
    except Exception as e:
        return None

# Caricamento iniziale
live_returns = get_truly_live_data()

# --- MOTORE MONTE CARLO ---
def run_simulation(capitale, prelievo_pct, equity_pct, anni, ter, n_sim, extra_expenses, mode, params=None):
    mesi = int(anni * 12)
    prelievo_mensile = (capitale * (prelievo_pct / 100)) / 12
    costi_mensili = (ter / 100) / 12
    
    # Gestione uscite straordinarie
    spese_pianificate = np.zeros(mesi + 1)
    for _, row in extra_expenses.iterrows():
        m = int(row['Anno'] * 12)
        if 0 <= m <= mesi:
            spese_pianificate[m] += row['Importo (€)']

    # Generazione rendimenti mensili
    if mode == "Bootstrap (LIVE DATA)" and live_returns is not None:
        idx = np.random.randint(0, len(live_returns), size=(mesi, n_sim))
        h_spy = live_returns['SPY'].values[idx]
        h_tlt = live_returns['TLT'].values[idx]
    else:
        # Parametrica (Gaussiana)
        m_spy = params.get('m_spy', 0.08) / 12
        s_spy = params.get('s_spy', 0.18) / np.sqrt(12)
        m_tlt = params.get('m_tlt', 0.03) / 12
        s_tlt = params.get('s_tlt', 0.08) / np.sqrt(12)
        corr = params.get('corr', 0.0)
        
        cov_val = corr * s_spy * s_tlt
        cov_matrix = [[s_spy**2, cov_val], [cov_val, s_tlt**2]]
        rets = np.random.multivariate_normal([m_spy, m_tlt], cov_matrix, size=(mesi, n_sim))
        h_spy, h_tlt = rets[:,:,0], rets[:,:,1]

    # Portafoglio e Simulazione
    port_returns = (h_spy * equity_pct) + (h_tlt * (1 - equity_pct))
    percorsi = np.zeros((mesi + 1, n_sim))
    percorsi[0] = capitale
    
    for t in range(mesi):
        val = percorsi[t] * (1 + port_returns[t] - costi_mensili)
        val = val - prelievo_mensile - spese_pianificate[t+1]
        val[val < 0] = 0
        percorsi[t+1] = val
    return percorsi

# --- INTERFACCIA STREAMLIT ---
st.title("🛡️ Diagnosi Portafoglio Monte Carlo (Dati Live)")

with st.sidebar:
    st.header("📡 Stato Connessione")
    
    # Verifica se i dati sono disponibili per mostrare la data corretta
    if live_returns is not None and not live_returns.empty:
        ultimo_agg = live_returns.index[-1].strftime('%d %B %Y')
        st.success(f"Dati Live: {ultimo_agg}")
        with st.expander("Vedi Storico Reale"):
            st.line_chart((1 + live_returns).cumprod() * 100)
    else:
        st.error("Dati Live non disponibili: Yahoo Finance Offline.")

    st.divider()
    st.header("1. Motore di Calcolo")
    options = ["Bootstrap (LIVE DATA)", "Parametrica (Gaussiana Custom)"]
    sim_mode = st.radio("Seleziona Sorgente:", options, index=0 if live_returns is not None else 1)
    
    param_dict = {}
    if sim_mode == "Parametrica (Gaussiana Custom)":
        st.subheader("⚙️ Ipotesi Custom")
        m_spy = st.number_input("Rendimento Azionario %", value=8.0) / 100
        s_spy = st.number_input("Volatilità Azionaria %", value=18.0) / 100
        m_tlt = st.number_input("Rendimento Bond %", value=3.0) / 100
        s_tlt = st.number_input("Volatilità Bond %", value=8.0) / 100
        corr = st.slider("Correlazione (Eq/Bond)", -1.0, 1.0, 0.0)
        param_dict = {'m_spy': m_spy, 's_spy': s_spy, 'm_tlt': m_tlt, 's_tlt': s_tlt, 'corr': corr}

    st.divider()
    st.header("2. Input Portafoglio")
    cap = st.number_input("Capitale Iniziale (€)", value=1000000, step=50000)
    prel = st.slider("Prelievo Annuo Lordo (%)", 0.0, 15.0, 4.0)
    eq = st.slider("Esposizione Azionaria (%)", 0.0, 1.0, 0.6)
    yrs = st.slider("Orizzonte (Anni)", 1, 50, 30)
    ter = st.slider("Costi (TER) %", 0.0, 5.0, 1.5)
    sim = st.selectbox("Precisione (N. Simulazioni)", [10000, 50000, 100000], index=1)
    
    st.header("3. Uscite Straordinarie")
    df_extra = pd.DataFrame([{"Anno": 15, "Importo (€)": 0.0}])
    edited_df = st.data_editor(df_extra, num_rows="dynamic", use_container_width=True)
    
    btn = st.button("ESEGUI SIMULAZIONE", type="primary", use_container_width=True)

if btn:
    with st.spinner('Calcolo in corso...'):
        dati = run_simulation(cap, prel, eq, yrs, ter, sim, edited_df, sim_mode, param_dict)
    
    p_levels = [5, 10, 25, 50, 75, 90, 95]
    pct = {p: np.percentile(dati, p, axis=1) for p in p_levels}
    successo = np.mean(dati[-1, :] > 0) * 100

    # --- KPI ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilità Successo", f"{successo:.1f}%")
    c2.metric("Annuo Netto (Est.)", f"€ {(cap*(prel/100)*0.74):,.0f}")
    c3.metric("Mensile Netto", f"€ {(cap*(prel/100)*0.74)/12:,.0f}")
    c4.metric("Campione Dati", f"{len(live_returns)} mesi" if live_returns is not None else "N/A")

    # --- GRAFICO FAN CHART ---
    st.subheader("📈 Proiezione Evolutiva Portafoglio")
    fig, ax = plt.subplots(figsize=(10, 4))
    t_range = np.arange(yrs * 12 + 1)
    ax.fill_between(t_range, pct[5], pct[95], color='royalblue', alpha=0.1, label='Range P5-P95')
    ax.fill_between(t_range, pct[25], pct[75], color='royalblue', alpha=0.3, label='Range P25-P75')
    ax.plot(t_range, pct[50], color='navy', label='P50 (Mediana)')
    ax.plot(t_range, pct[10], color='red', linestyle='--', label='P10 (Stress Test)')
    ax.set_ylabel("Capitale (€)")
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # --- TABELLA RIASSUNTIVA ---
    st.subheader("📑 Dettaglio Scenari al termine")
    df_tab = pd.DataFrame({f"P{p}": [pct[p][-1]] for p in p_levels}, index=["Capitale Finale (€)"])
    st.dataframe(df_tab.style.format("{:,.0f}"), use_container_width=True)