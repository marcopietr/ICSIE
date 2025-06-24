import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.text import Text

nltk.download('punkt')

st.set_page_config(page_title="Analisi Testi Italiani", layout="wide")

# Carica dati
@st.cache_data
def load_data():
    df_risp = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="Risposte + NLP")
    df_tfidf = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="TFIDF_top50")
    return df_risp, df_tfidf

df_risposte, df_tfidf = load_data()

# --- Prepara interfaccia ---
st.title("📊 Analisi delle Risposte: Italiano nel Mondo")

# --- Filtri dinamici ---
st.sidebar.header("🔍 Filtri dinamici")

# Opzioni uniche
paesi = df_risposte["Paese"].dropna().unique()
tipologie = df_risposte["Tipologia istituzione"].dropna().unique()
istituzioni = df_risposte["Nome istituzione/rappresentanza"].dropna().unique()

# Filtri interattivi dinamici
sel_paesi = st.sidebar.multiselect("Paesi", sorted(paesi), default=sorted(paesi))
df_filtrato = df_risposte[df_risposte["Paese"].isin(sel_paesi)]

sel_tipi = st.sidebar.multiselect("Tipologia istituzione", sorted(df_filtrato["Tipologia istituzione"].unique()),
                                   default=sorted(df_filtrato["Tipologia istituzione"].unique()))
df_filtrato = df_filtrato[df_filtrato["Tipologia istituzione"].isin(sel_tipi)]

sel_ist = st.sidebar.multiselect("Istituzione", sorted(df_filtrato["Nome istituzione/rappresentanza"].unique()),
                                  default=sorted(df_filtrato["Nome istituzione/rappresentanza"].unique()))
df_filtrato = df_filtrato[df_filtrato["Nome istituzione/rappresentanza"].isin(sel_ist)]

# --- Selezione domanda per analisi ---
domande = [col for col in df_risposte.columns if col.startswith("LEMMI_NORM")]
domanda_sel = st.selectbox("❓ Seleziona una domanda per l'analisi", ["Tutte"] + domande)

# --- Word Cloud da TF-IDF ---
st.subheader("☁️ Nuvola di Parole (TF-IDF)")
if domanda_sel != "Tutte":
    df_tfidf_filtrato = df_tfidf[df_tfidf["Domanda"] == domanda_sel]
else:
    df_tfidf_filtrato = df_tfidf.copy()

tfidf_freq = df_tfidf_filtrato.groupby("Lemma")["TF-IDF"].sum().sort_values(ascending=False)
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate_from_frequencies(tfidf_freq)

fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# --- Frequenze parole ---
st.subheader("📋 Frequenze parole (grezze)")
if domanda_sel != "Tutte":
    lemmi = df_filtrato[domanda_sel].dropna().explode()
else:
    lemmi_cols = [col for col in df_filtrato.columns if col.startswith("LEMMI_NORM")]
    lemmi = pd.Series([lemma for sublist in df_filtrato[lemmi_cols].dropna(how='all').values.flatten() if isinstance(sublist, list) for lemma in sublist])

frequenze = lemmi.value_counts().head(50)
st.dataframe(frequenze.rename_axis("Lemma").reset_index(name="Frequenza"))

# --- Concordanze ---
st.subheader("🔎 Concordanze")
parola = st.text_input("Scrivi una parola da cercare nei testi:")

if parola:
    st.markdown("**Contesti trovati (max 10):**")
    # Prendi il testo completo da tutte le domande
    domande_testuali = [col for col in df_risposte.columns if col.startswith("Cosa significa") or col.startswith("Come ") or col.startswith("Quali ")]

    testo_completo = df_filtrato[domande_testuali].astype(str).apply(lambda row: " ".join(row), axis=1).str.cat(sep=" ")
    tokens = nltk.word_tokenize(testo_completo.lower())
    concordanza = Text(tokens).concordance_list(parola.lower(), width=70, lines=10)

    if concordanza:
        for c in concordanza:
            st.text("... " + c.line + " ...")
    else:
        st.warning("Nessuna occorrenza trovata.")
