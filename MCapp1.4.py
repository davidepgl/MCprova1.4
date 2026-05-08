import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Monte Carlo Financial Advisor 2026", layout="wide")

# --- FUNZIONE DOWNLOAD DATI (VERSIONE ULTRA-ROBUSTA) ---
@st.cache_data(ttl=86400)
def get_live_market_data():
    try:
        tickers = ["SPY", "TLT"]
        # Proviamo a scaricare i dati. Usiamo threads=False per maggiore stabilità su server cloud
        df = yf.download(tickers, start="2002-01-01", interval="1mo", auto_adjust=True, threads=False)
        
        if df.empty:
            return None
            
        # Gestione Multi-Index
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns:
                data = df['Close']
            else:
                # Se Close non c'è, prendiamo il primo livello disponibile
                data = df.iloc[:, :len(tickers)] 
        else:
            data = df
        
        data = data.ffill().dropna()
        
        if len(data) < 2: # Se abbiamo troppi pochi dati
            return None
            
        returns = data.pct_change().dropna()
        returns.columns = ["SPY", "TLT"]
        return returns
    except Exception:
        return None

# Caricamento iniziale
live_returns = get_live_market_data()

# --- MOTORE MONTE CARLO ---
def run_simulation(capitale, prelievo_pct, equity_pct, anni, ter, n_sim, extra_expenses, mode, params=None):
    mesi = int(anni * 12)
    prelievo_mensile = (capitale * (prelievo_pct / 100)) / 12
    costi_mensili = (ter / 100) / 12
    
    spese_pianificate = np.zeros(mesi + 1)
    for _, row in extra_expenses.iterrows():
        m = int(row['Anno'] * 12)
        if 0 <= m <= mesi:
            spese_pianificate[m] += row['Importo (€)']

    # Scelta del dataset di rendimenti
    if mode == "Bootstrap (Dati Reali Live)" and live_returns is not None:
        idx = np.random.randint(0, len(live_returns), size=(mesi, n_sim))
        h_spy = live_returns['SPY'].values[idx]
        h_tlt = live_returns['TLT'].values[idx]
    else:
        # Fallback o Modalità Parametrica
        m_spy = params.get('m_spy', 0.07) / 12
        s_spy = params.get('s_spy', 0.18) / np.sqrt(12)
        m_tlt = params.get('m_tlt', 0.02) / 12
        s_tlt = params.get('s_tlt', 0.07) / np.sqrt(12)
        corr = params.get('corr', 0.0)
        
        cov_val = corr * s_spy * s_tlt
        cov_matrix = [[s_spy**2, cov_val], [cov_val, s_tlt**2]]
        rets = np.random.multivariate_normal([m_spy, m_tlt], cov_matrix, size=(mesi, n_sim))
        h_spy, h_tlt = rets[:,:,0], rets[:,:,1]

    port_returns = (h_spy * equity_pct) + (h_tlt * (1 - equity_pct))
    percorsi = np.zeros((mesi + 1, n_sim))
    percorsi[0] = capitale
    
    for t in range(mesi):
        val = percorsi[t] * (1 + port_returns[t] - costi_mensili)
        val = val - prelievo_mensile - spese_pianificate[t+1]
        val[val < 0] = 0
        percorsi[t+1] = val
    return percorsi

# --- INTERFACCIA ---
st.title("🛡️ Simulatore Monte Carlo: Diagnosi Real-Time")

with st.sidebar:
    st.header("1. Controllo Dati Live")
    
    # --- FIX SICUREZZA PER L'ERRORE INDEX[-1] ---
    if live_returns is not None and not live_returns.empty:
        ultimo_agg = live_returns.index[-1].strftime('%B %Y')
        st.success(f"✅ Dati Mercato: **{ultimo_agg}**")
        with st.expander("Vedi Grafico Prezzi Reali"):
            prezzi_storici = (1 + live_returns).cumprod() * 100
            st.line_chart(prezzi_storici)
    else:
        st.warning("⚠️ Dati Live non disponibili. Utilizzo parametri statistici (Parametrica).")
        sim_mode_default = 1 # Forza la selezione su Parametrica
    
    st.divider()
    st.header("2. Motore di Calcolo")
    # Se i dati live mancano, disabilitiamo l'opzione Bootstrap o avvisiamo l'utente
    options = ["Bootstrap (Dati Reali Live)", "Parametrica (Gaussiana Custom)"]
    sim_mode = st.radio("Metodo:", options, index=1 if live_returns is None else 0)
    
    param_dict = {}
    # Mostriamo sempre i parametri se siamo in parametrica o se il bootstrap è fallito
    if sim_mode == "Parametrica (Gaussiana Custom)" or live_returns is None:
        st.subheader("⚙️ Ipotesi Asset Class")
        m_spy = st.number_input("Azionario: Rend. atteso %", value=7.0) / 100
        s_spy = st.number_input("Azionario: Volatilità %", value=18.0) / 100
        m_tlt = st.number_input("Obbligazionario: Rend. atteso %", value=2.0) / 100
        s_tlt = st.number_input("Obbligazionario: Volatilità %", value=7.0) / 100
        corr = st.slider("Correlazione (Equity/Bond)", -1.0, 1.0, 0.0)
        param_dict = {'m_spy': m_spy, 's_spy': s_spy, 'm_tlt': m_tlt, 's_tlt': s_tlt, 'corr': corr}

    st.divider()
    st.header("3. Parametri Portafoglio")
    cap = st.number_input("Capitale Iniziale (€)", value=1000000, step=50000)
    prel = st.slider("Prelievo Annuo Lordo (%)", 0.0, 15.0, 4.0)
    eq = st.slider("Esposizione Azionaria (%)", 0.0, 1.0, 0.6)
    yrs = st.slider("Anni di Proiezione", 1, 50, 30)
    ter = st.slider("Costi (TER) %", 0.0, 5.0, 1.5)
    sim = st.selectbox("N. Simulazioni", [10000, 50000, 100000], index=1)
    
    st.header("4. Uscite Extra")
    df_extra = pd.DataFrame([{"Anno": 10, "Importo (€)": 0.0}])
    edited_df = st.data_editor(df_extra, num_rows="dynamic", use_container_width=True)
    
    btn = st.button("ANALIZZA PORTAFOGLIO", type="primary", use_container_width=True)

if btn:
    with st.spinner('Elaborazione scenari in corso...'):
        dati = run_simulation(cap, prel, eq, yrs, ter, sim, edited_df, sim_mode, param_dict)
    
    p_levels = [5, 10, 25, 50, 75, 90, 95]
    pct = {p: np.percentile(dati, p, axis=1) for p in p_levels}
    successo = np.mean(dati[-1, :] > 0) * 100

    # Riquadri KPI
    prel_ann_N = (cap * (prel / 100)) * 0.74
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Successo", f"{successo:.1f}%")
    c2.metric("Annuo Netto (26%)", f"€ {prel_ann_N:,.0f}")
    c3.metric("Mensile Netto", f"€ {prel_ann_N/12:,.0f}")
    c4.metric("Scenari", f"{sim:,}")

    # Grafico
    st.subheader("📈 Evoluzione Fan Chart")
    fig, ax = plt.subplots(figsize=(12, 5))
    t_range = np.arange(yrs * 12 + 1)
    ax.fill_between(t_range, pct[5], pct[95], color='royalblue', alpha=0.1, label='Range P5-P95')
    ax.fill_between(t_range, pct[25], pct[75], color='royalblue', alpha=0.3, label='Range P25-P75')
    ax.plot(t_range, pct[50], color='navy', linewidth=2, label='Mediana (P50)')
    ax.plot(t_range, pct[10], color='red', linestyle='--', label='Stress Test (P10)')
    ax.legend(loc='upper left')
    st.pyplot(fig)