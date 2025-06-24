import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.text import Text
from collections import defaultdict
import random

st.set_page_config(page_title="Analisi Testi Tavolo Scuole Italiane", layout="wide")

@st.cache_data
def load_data():
    df_risp = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="Risposte + NLP")
    df_tfidf = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="TFIDF_top50")
    return df_risp, df_tfidf

df_risposte, df_tfidf = load_data()

st.title("📚 Analisi delle seguiti del tavolo di discussione della Prima Conferenza delle scuole italiane all'estero")

# Sidebar con filtri integrati
with st.sidebar:
    st.header("🔍 Filtri")

    def multiselect_with_all(label, options, key):
        all_label = f"Tutti i {label.lower()}"
        options_with_all = [all_label] + options
        default = [all_label]
        selection = st.multiselect(label, options_with_all, default=default, key=key)
        if all_label in selection or len(selection) == 0:
            return options
        else:
            return selection

    paesi_disp = sorted(df_risposte["Paese"].dropna().unique())
    sel_paesi = multiselect_with_all("Paesi", paesi_disp, "paesi")
    df_f1 = df_risposte[df_risposte["Paese"].isin(sel_paesi)]

    tipi_disp = sorted(df_f1["Tipologia istituzione"].dropna().unique())
    sel_tipi = multiselect_with_all("Tipologie istituzione", tipi_disp, "tipi")
    df_f2 = df_f1[df_f1["Tipologia istituzione"].isin(sel_tipi)]

    ist_disp = sorted(df_f2["Nome istituzione/rappresentanza"].dropna().unique())
    sel_ist = multiselect_with_all("Istituzioni", ist_disp, "istituzioni")
    df_filtrato = df_f2[df_f2["Nome istituzione/rappresentanza"].isin(sel_ist)]

# Mappatura domande
col_map = {
    'Cosa significa «crescere in italiano»?': 'LEMMI_NORM_Cosa significa «crescere in italiano»?',
    "Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?": "LEMMI_NORM_Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?",
    'Quali sono i fattori di attrattività nelle rispettive aree geografiche?': 'LEMMI_NORM_Quali sono i fattori di attrattività nelle rispettive aree geografiche?',
    'Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?': 'LEMMI_NORM_Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?'
}
domanda_sel_label = st.selectbox("❓ Seleziona una domanda", ["Tutte"] + list(col_map.keys()))
col_sel = col_map.get(domanda_sel_label)

# Categorie semantiche
categorie = {
    "italiano": "green", "lingua": "green", "italia": "green",
    "musica": "red", "arte": "red", "letteratura": "red", "cultura": "red", "culturale": "red",
    "scuola": "blue", "studente": "blue", "educativo": "blue", "formazione": "blue",
    "famiglia": "orange", "comunità": "orange", "territorio": "orange"
}

def color_func(word, **kwargs):
    return categorie.get(word, f"hsl({random.randint(0, 360)}, 60%, 40%)")

# Wordcloud
st.subheader("☁️ Nuvola di Parole (TF-IDF, categorie colorate)")
if domanda_sel_label != "Tutte":
    df_tfidf_filtrato = df_tfidf[df_tfidf["Domanda"] == col_sel]
else:
    df_tfidf_filtrato = df_tfidf.copy()

stopwords = set(STOPWORDS)
df_tfidf_filtrato = df_tfidf_filtrato[~df_tfidf_filtrato["Lemma"].isin(stopwords)]
tfidf_freq = df_tfidf_filtrato.groupby("Lemma")["TF-IDF"].sum().sort_values(ascending=False)

if not tfidf_freq.empty:
    wordcloud = WordCloud(width=1000, height=500, background_color='white', color_func=color_func)        .generate_from_frequencies(tfidf_freq)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("Nessun dato disponibile per la word cloud.")

# Frequenze
st.subheader("📋 Frequenze (TF-IDF)")
freq_df = tfidf_freq.reset_index()
freq_df.columns = ["Parola", "TF-IDF"]
st.dataframe(freq_df)

# Concordanze
st.subheader("🔎 Concordanze con evidenziazione (testo esteso)")
query = st.text_input("Cerca una parola nei testi originali:")

if query:
    colonne_testuali = [col for col in df_risposte.columns if not col.startswith("LEMMI_") and col.startswith(("Cosa ", "Come ", "Quali "))]
    testo_completo = df_filtrato[colonne_testuali].astype(str).apply(lambda r: " ".join(r), axis=1).str.cat(sep=" ")
    tokens = re.findall(r'\b\w+\b', testo_completo.lower())
    text_obj = Text(tokens)
    results = text_obj.concordance_list(query.lower(), width=240, lines=10)

    if results:
        for r in results:
            pattern = rf"\b{query.lower()}\b"
            line = re.sub(pattern, f"**{query.lower()}**", r.line)
            st.markdown(f"... {line} ...")
    else:
        st.warning("Nessuna occorrenza trovata.")

# Testi originali ordinati per Paese
st.subheader("📝 Visualizza risposte originali per Paese")
col_testuali = [col for col in df_risposte.columns if col.startswith(("Cosa ", "Come ", "Quali "))]
df_filtrato_sorted = df_filtrato.sort_values(by=["Paese", "Cognome", "Nome"])

for i, row in df_filtrato_sorted.iterrows():
    nome = row.get("Nome", "")
    cognome = row.get("Cognome", "")
    paese = row.get("Paese", "")
    istituzione = row.get("Nome istituzione/rappresentanza", "")
    with st.expander(f"🇮🇹 {paese} – {nome} {cognome} ({istituzione})"):
        for domanda in col_testuali:
            st.markdown(f"**{domanda}**")
            st.markdown(row[domanda])
