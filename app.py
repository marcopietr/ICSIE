import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.text import Text

nltk.download('punkt')

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

# --- Selezione domanda ---
domande = [col for col in df_risposte.columns if col.startswith("LEMMI_NORM")]
domanda_sel = st.selectbox("❓ Seleziona una domanda per l'analisi", ["Tutte"] + domande)

# --- Word Cloud TF-IDF ---
st.subheader("☁️ Nuvola di Parole (TF-IDF)")
if domanda_sel != "Tutte":
    df_tfidf_filtrato = df_tfidf[df_tfidf["Domanda"] == domanda_sel]
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

# --- Frequenze parole ---
st.subheader("📋 Frequenze parole")
if domanda_sel != "Tutte" and domanda_sel in df_filtrato.columns:
    lemmi = df_filtrato[domanda_sel].dropna().explode()
else:
    lemmi_cols = [col for col in df_filtrato.columns if col.startswith("LEMMI_NORM")]
    lemmi = pd.Series([lemma for sublist in df_filtrato[lemmi_cols].dropna(how='all').values.flatten() if isinstance(sublist, list) for lemma in sublist])

frequenze = lemmi.value_counts().head(50)
if not frequenze.empty:
    st.dataframe(frequenze.rename_axis("Lemma").reset_index(name="Frequenza"))
else:
    st.info("Nessuna frequenza disponibile con i filtri attuali.")

# --- Concordanze ---
st.subheader("🔎 Concordanze")
parola = st.text_input("Scrivi una parola da cercare nei testi:")
if parola:
    domande_testuali = [col for col in df_risposte.columns if col.startswith("Cosa significa") or col.startswith("Come ") or col.startswith("Quali ")]
    testo_completo = df_filtrato[domande_testuali].astype(str).apply(lambda row: " ".join(row), axis=1).str.cat(sep=" ")
    tokens = nltk.word_tokenize(testo_completo.lower())
    concordanza = Text(tokens).concordance_list(parola.lower(), width=70, lines=10)

    if concordanza:
        for c in concordanza:
            st.text("... " + c.line + " ...")
    else:
        st.warning("Nessuna occorrenza trovata.")
