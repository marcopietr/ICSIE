import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.text import Text
from collections import defaultdict

st.set_page_config(page_title="Analisi Testi Tavolo Scuole Italiane", layout="wide")

@st.cache_data
def load_data():
    df_risp = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="Risposte + NLP")
    df_tfidf = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="TFIDF_top50")
    return df_risp, df_tfidf

df_risposte, df_tfidf = load_data()

st.title("📚 Analisi dei seguiti del tavolo di discussione della Prima Conferenza delle scuole italiane all'estero")

# Sidebar con filtri leggibili
with st.sidebar:
    st.header("🔍 Filtri")
    st.markdown("Filtra le risposte per Paese, Tipologia e Istituzione.")

    paesi_disp = df_risposte["Paese"].dropna().unique()
    sel_paesi = st.multiselect("🌍 Paesi", sorted(paesi_disp), default=sorted(paesi_disp), label_visibility="visible")
    df_f1 = df_risposte[df_risposte["Paese"].isin(sel_paesi)]

    tipi_disp = df_f1["Tipologia istituzione"].dropna().unique()
    sel_tipi = st.multiselect("🏫 Tipologia istituzione", sorted(tipi_disp), default=sorted(tipi_disp))
    df_f2 = df_f1[df_f1["Tipologia istituzione"].isin(sel_tipi)]

    ist_disp = df_f2["Nome istituzione/rappresentanza"].dropna().unique()
    sel_ist = st.multiselect("🏛️ Istituzione", sorted(ist_disp), default=sorted(ist_disp))
    df_filtrato = df_f2[df_f2["Nome istituzione/rappresentanza"].isin(sel_ist)]

# Domande leggibili
col_map = {
    'Cosa significa «crescere in italiano»?': 'LEMMI_NORM_Cosa significa «crescere in italiano»?',
    "Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?": "LEMMI_NORM_Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?",
    'Quali sono i fattori di attrattività nelle rispettive aree geografiche?': 'LEMMI_NORM_Quali sono i fattori di attrattività nelle rispettive aree geografiche?',
    'Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?': 'LEMMI_NORM_Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?'
}
domanda_sel_label = st.selectbox("❓ Seleziona una domanda", ["Tutte"] + list(col_map.keys()))
col_sel = col_map.get(domanda_sel_label)

# Word cloud e frequenze lemmi TF-IDF
st.subheader("☁️ Nuvola di Parole (TF-IDF, senza stopwords)")
if domanda_sel_label != "Tutte":
    df_tfidf_filtrato = df_tfidf[df_tfidf["Domanda"] == col_sel]
else:
    df_tfidf_filtrato = df_tfidf.copy()

stopwords = set(STOPWORDS)
df_tfidf_filtrato = df_tfidf_filtrato[~df_tfidf_filtrato["Lemma"].isin(stopwords)]
tfidf_freq = df_tfidf_filtrato.groupby("Lemma")["TF-IDF"].sum().sort_values(ascending=False)

if not tfidf_freq.empty:
    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate_from_frequencies(tfidf_freq)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("Nessun dato disponibile per la word cloud.")

# Frequenze TF-IDF
st.subheader("📋 Frequenze (TF-IDF)")
freq_df = tfidf_freq.reset_index()
freq_df.columns = ["Parola", "TF-IDF"]
st.dataframe(freq_df)

# Concordanze
st.subheader("🔎 Concordanze con evidenziazione")
query = st.text_input("Cerca una parola nei testi originali:")

if query:
    colonne_testuali = [col for col in df_risposte.columns if not col.startswith("LEMMI_") and col.startswith(("Cosa ", "Come ", "Quali "))]
    testo_completo = df_filtrato[colonne_testuali].astype(str).apply(lambda r: " ".join(r), axis=1).str.cat(sep=" ")
    tokens = re.findall(r'\b\w+\b', testo_completo.lower())
    text_obj = Text(tokens)
    results = text_obj.concordance_list(query.lower(), width=120, lines=10)  # LARGHEZZA AUMENTATA

    if results:
        for r in results:
            # Evidenziazione in grassetto
            pattern = rf"\b{query.lower()}\b"
            line = re.sub(pattern, f"**{query.lower()}**", r.line)
            st.markdown(f"... {line} ...")
    else:
        st.warning("Nessuna occorrenza trovata.")
