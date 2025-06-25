import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import random

st.set_page_config(page_title="Analisi Testi Tavolo Scuole Italiane", layout="wide")

@st.cache_data
def load_data():
    df_risp = pd.read_excel("analisi_risposte_con_tfidf.xlsx", sheet_name="Risposte + NLP")
    return df_risp

df_risposte = load_data()

st.title("📚 Analisi delle seguiti del tavolo di discussione della Prima Conferenza delle scuole italiane all'estero")

# Sidebar con filtri integrati
with st.sidebar:
    st.header("🔍 Filtri")

    def multiselect_with_all(label, options, key):
        all_label = f"Tutti i {label.lower()}"
        options_with_all = [all_label] + list(options)
        selection = st.multiselect(label, options_with_all, default=[all_label], key=key)
        return options if all_label in selection or len(selection) == 0 else selection

    paesi_disp = sorted(df_risposte["Paese"].dropna().unique())
    sel_paesi = multiselect_with_all("Paesi", paesi_disp, "paesi")
    df1 = df_risposte[df_risposte["Paese"].isin(sel_paesi)]

    tipi_disp = sorted(df1["Tipologia istituzione"].dropna().unique())
    sel_tipi = multiselect_with_all("Tipologie istituzione", tipi_disp, "tipi")
    df2 = df1[df1["Tipologia istituzione"].isin(sel_tipi)]

    ist_disp = sorted(df2["Nome istituzione/rappresentanza"].dropna().unique())
    sel_ist = multiselect_with_all("Istituzioni", ist_disp, "istituzioni")
    df_filtrato = df2[df2["Nome istituzione/rappresentanza"].isin(sel_ist)]

# Mappa domande leggibili -> nomi colonne
col_map = {
    'Cosa significa «crescere in italiano»?': 'LEMMI_NORM_Cosa significa «crescere in italiano»?',
    "Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?": "LEMMI_NORM_Come contribuiscono le scuole all’estero alla promozione culturale dell'Italia?",
    'Quali sono i fattori di attrattività nelle rispettive aree geografiche?': 'LEMMI_NORM_Quali sono i fattori di attrattività nelle rispettive aree geografiche?',
    'Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?': 'LEMMI_NORM_Come possono contribuire le scuole all’estero a una comunità globale dell’italofonia?'
}
domanda_label = st.selectbox("❓ Seleziona una domanda", ["Tutte"] + list(col_map.keys()))
domanda_col = col_map.get(domanda_label)

# Filtro TF-IDF solo su testi presenti nel filtro
if domanda_label != "Tutte" and domanda_col in df_filtrato.columns:
    lemmi = df_filtrato[domanda_col].dropna().explode()
else:
    lemm_cols = [col for col in df_filtrato.columns if col.startswith("LEMMI_NORM_")]
    lemmi = pd.Series([lemma for sublist in df_filtrato[lemm_cols].dropna(how='all').values.flatten() if isinstance(sublist, list) for lemma in sublist])

stopwords = set(STOPWORDS)
lemmi = lemmi[~lemmi.isin(stopwords)]
tfidf_freq = lemmi.value_counts().head(100)

# Categorie semantiche colorate
categorie = {
    "italiano": "green", "lingua": "green", "italia": "green",
    "musica": "red", "arte": "red", "letteratura": "red", "cultura": "red", "culturale": "red",
    "scuola": "blue", "studente": "blue", "educativo": "blue", "formazione": "blue",
    "famiglia": "orange", "comunità": "orange", "territorio": "orange"
}
def color_func(word, **kwargs):
    return categorie.get(word, f"hsl({random.randint(0, 360)}, 60%, 40%)")

# Wordcloud
st.subheader("☁️ Nuvola di Parole (TF-IDF simulato su testi filtrati)")
if not tfidf_freq.empty:
    wordcloud = WordCloud(width=1000, height=500, background_color='white', color_func=color_func)        .generate_from_frequencies(tfidf_freq.to_dict())
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("Nessun dato disponibile per la word cloud.")

# Frequenze
st.subheader("📋 Frequenze basate sui testi selezionati")
if not tfidf_freq.empty:
    freq_df = tfidf_freq.reset_index()
    freq_df.columns = ["Parola", "Frequenza"]
    st.dataframe(freq_df)
else:
    st.info("Nessuna frequenza disponibile con i filtri attuali.")
