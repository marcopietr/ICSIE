import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from nltk.text import Text
from collections import Counter

st.set_page_config(page_title="Analisi Testi Italiani", layout="wide")

@st.cache_data
def load_data():
    df_risp = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="Risposte + NLP")
    df_tfidf = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="TFIDF_top50")
    return df_risp, df_tfidf

df_risposte, df_tfidf = load_data()

st.title("📊 Analisi delle Risposte: Italiano nel Mondo")

# --- Sidebar Filtri dinamici ---
st.sidebar.header("🔍 Filtri dinamici")

# Step 1: Filtra per Paese
paesi_disp = df_risposte["Paese"].dropna().unique()
sel_paesi = st.sidebar.multiselect("Paesi", sorted(paesi_disp), default=sorted(paesi_disp))
df_f1 = df_risposte[df_risposte["Paese"].isin(sel_paesi)]

# Step 2: Filtra per Tipologia istituzione
tipi_disp = df_f1["Tipologia istituzione"].dropna().unique()
sel_tipi = st.sidebar.multiselect("Tipologia istituzione", sorted(tipi_disp), default=sorted(tipi_disp))
df_f2 = df_f1[df_f1["Tipologia istituzione"].isin(sel_tipi)]

# Step 3: Filtra per Istituzione
ist_disp = df_f2["Nome istituzione/rappresentanza"].dropna().unique()
sel_ist = st.sidebar.multiselect("Istituzione", sorted(ist_disp), default=sorted(ist_disp))
df_filtrato = df_f2[df_f2["Nome istituzione/rappresentanza"].isin(sel_ist)]

# --- Mappa domande e colonne LEMMI_NORM ---
col_map = {
    'Cosa significa «crescere in italiano»?': 'LEMMI_NORM_Cosa significa «crescere in italiano»?',
    "Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?": "LEMMI_NORM_Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?",
    'Quali sono i fattori di attrattività nelle rispettive aree geografiche?': 'LEMMI_NORM_Quali sono i fattori di attrattività nelle rispettive aree geografiche?',
    'Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?': 'LEMMI_NORM_Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?'
}

domanda_sel_label = st.selectbox("❓ Seleziona una domanda per l'analisi", ["Tutte"] + list(col_map.keys()))
col_sel = col_map.get(domanda_sel_label)

# --- Word Cloud TF-IDF ---
st.subheader("☁️ Nuvola di Parole (TF-IDF)")
if domanda_sel_label != "Tutte":
    df_tfidf_filtrato = df_tfidf[df_tfidf["Domanda"] == col_sel]
else:
    df_tfidf_filtrato = df_tfidf.copy()

tfidf_freq = df_tfidf_filtrato.groupby("Lemma")["TF-IDF"].sum().sort_values(ascending=False)
if not tfidf_freq.empty:
    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate_from_frequencies(tfidf_freq)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("Nessun dato disponibile per la word cloud.")

# --- Frequenze parole (da testo completo) ---
st.subheader("📋 Frequenze parole (testo originale tokenizzato)")
colonne_testuali = [col for col in df_risposte.columns if not col.startswith("LEMMI_") and col.startswith(("Cosa ", "Come ", "Quali "))]

if domanda_sel_label != "Tutte":
    testi = df_filtrato[domanda_sel_label].dropna().astype(str).tolist()
else:
    testi = df_filtrato[colonne_testuali].astype(str).apply(lambda r: " ".join(r), axis=1).tolist()

tokens = re.findall(r'\b\w+\b', " ".join(testi).lower())
frequenze_parole = Counter(tokens).most_common(50)

if frequenze_parole:
    st.dataframe(pd.DataFrame(frequenze_parole, columns=["Parola", "Frequenza"]))
else:
    st.info("Nessuna parola trovata.")

# --- Concordanze ---
st.subheader("🔎 Concordanze")
parola = st.text_input("Scrivi una parola da cercare nei testi:")

if parola:
    corpus = " ".join(testi).lower()
    tokens_all = re.findall(r'\b\w+\b', corpus)
    concordanza = Text(tokens_all).concordance_list(parola.lower(), width=70, lines=10)

    if concordanza:
        for c in concordanza:
            st.text("... " + c.line + " ...")
    else:
        st.warning("Nessuna occorrenza trovata.")
