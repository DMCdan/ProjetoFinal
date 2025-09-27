
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Análise de Filmes - TMDb", layout="wide")

# === Carregar os dados ===
@st.cache_data
def load_data():
    movies_df = pd.read_csv('tmdb_5000_movies.csv')
    credits_df = pd.read_csv('tmdb_5000_credits.csv', engine='python', on_bad_lines='skip')
    df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')
    df = df.drop('movie_id', axis=1)
    return df

# === Limpeza e Parsing JSON ===
def parse_json_column(df, column):
    for index, row in df.iterrows():
        try:
            df.at[index, column] = json.loads(row[column])
        except (TypeError, json.JSONDecodeError):
            df.at[index, column] = []
    return df

df = load_data()
for col in ['genres', 'keywords', 'cast', 'crew']:
    df = parse_json_column(df, col)

# === Funções auxiliares ===
def main_genre_in_list(genre_list):
    if genre_list:
        return genre_list[0]['name']
    return None

df['main_genre'] = df['genres'].apply(main_genre_in_list)

# === Modelo de Regressão Linear ===
X = df[['budget', 'revenue', 'popularity', 'vote_count']]
y = df['vote_average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# === Layout com abas ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral", 
    "🎬 Gêneros vs Notas", 
    "🌍 Idiomas", 
    "📈 Modelo Preditivo", 
    "🧮 Métricas"
])

with tab1:
    st.title("Análise de Filmes - TMDb")
    st.markdown("Este dashboard analisa dados de mais de 5.000 filmes disponíveis no banco de dados do TMDb.")
    
    st.subheader("Distribuição das Notas dos Filmes")
    fig, ax = plt.subplots()
    sns.histplot(df['vote_average'], kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

    st.subheader("Relação entre Orçamento e Receita")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='budget', y='revenue', ax=ax2, color="coral")
    st.pyplot(fig2)

    st.markdown("Fonte: [TMDb Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-dataset)")

with tab2:
    st.subheader("Distribuição das Notas por Gênero Principal")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='main_genre', y='vote_average', ax=ax, palette="Set2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

with tab3:
    st.subheader("Contagem de Filmes por Idioma Original")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(data=df, y='original_language', order=df['original_language'].value_counts().index, palette="pastel", ax=ax)
    st.pyplot(fig)

with tab4:
    st.subheader("Modelo de Regressão Linear")
    st.markdown("O modelo usa as variáveis:")
    st.code("['budget', 'revenue', 'popularity', 'vote_count']", language='python')

    st.write("### Gráfico: Notas Reais vs Notas Previstas")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6, color="green")
    ax.set_xlabel("Nota Real")
    ax.set_ylabel("Nota Prevista")
    ax.set_title("Real vs Previsto")
    st.pyplot(fig)

with tab5:
    st.subheader("Métricas do Modelo")
    st.metric("MSE", f"{mse:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R²", f"{r2:.2f}")
    st.markdown("Essas métricas indicam a qualidade da previsão feita com regressão linear.")
