import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

st.set_page_config(page_title="t-SNE 3D Dashboard", layout="wide")

st.title("Интерактивный t-SNE 3D Дашборд")
st.write("Загрузите CSV-файл, настройте параметры t-SNE в боковой панели и исследуйте трехмерный эмбеддинг.")

# --- 1. БОКОВАЯ ПАНЕЛЬ: ЗАГРУЗКА ДАННЫХ ---
st.sidebar.header("1. Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Файл загружен!")
else:
    # Демо-данные (3 пространственных кластера в 10-мерном пространстве)
    st.sidebar.info("Используются демонстрационные данные.")
    np.random.seed(42)
    n_samples = 600
    c1 = np.random.normal(loc=0, scale=1.0, size=(200, 10))
    c2 = np.random.normal(loc=4, scale=1.2, size=(200, 10))
    c3 = np.random.normal(loc=-4, scale=1.5, size=(200, 10))
    X_demo = np.vstack([c1, c2, c3])
    y_demo = np.array(["Цифра 0"] * 200 + ["Цифра 1"] * 200 + ["Цифра 2"] * 200)
    
    df = pd.DataFrame(X_demo, columns=[f"feat_{i}" for i in range(10)])
    df["label"] = y_demo

# Выбор колонок
all_columns = df.columns.tolist()
default_label = "label" if "label" in all_columns else all_columns[-1]

st.sidebar.markdown("---")
st.sidebar.header("2. Настройка признаков")
label_col = st.sidebar.selectbox("Колонка с метками классов", all_columns, index=all_columns.index(default_label))
feature_cols = [col for col in all_columns if col != label_col]
selected_features = st.sidebar.multiselect("Признаки для t-SNE", feature_cols, default=feature_cols)

# --- 2. БОКОВАЯ ПАНЕЛЬ: НАСТРОЙКА ПАРАМЕТРОВ ---
st.sidebar.markdown("---")
st.sidebar.header("3. Параметры t-SNE")
p_perplexity = st.sidebar.slider("Perplexity", min_value=5.0, max_value=50.0, value=30.0, step=1.0)
p_early_exagg = st.sidebar.slider("Early Exaggeration", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
p_lr = st.sidebar.slider("Learning Rate", min_value=10.0, max_value=1000.0, value=200.0, step=50.0)
p_max_iter = st.sidebar.slider("Max Iterations (max_iter)", min_value=250, max_value=2000, value=1000, step=250)


# --- 3. ОСНОВНОЙ БЛОК: ВЫЧИСЛЕНИЯ И ПРОСМОТР ---
if len(selected_features) < 3:
    st.warning("Для построения корректной 3D-проекции выберите как минимум 3 признака.")
else:
    X = df[selected_features].values
    labels = df[label_col].astype(str).values

    st.subheader("Предпросмотр таблицы")
    st.dataframe(df[[label_col] + selected_features].head(5), use_container_width=True)

    # Кнопка запуска
    if st.button("Рассчитать t-SNE 3D", type="primary"):
        with st.spinner("Вычисление 3D проекции t-SNE..."):
            tsne = TSNE(
                n_components=3,
                perplexity=p_perplexity,
                early_exaggeration=p_early_exagg,
                learning_rate=p_lr,
                max_iter=p_max_iter,
                random_state=42
            )
            Y_tsne = tsne.fit_transform(X)
            
            # Собираем DataFrame для Plotly
            df_plot = pd.DataFrame(Y_tsne, columns=["t-SNE 1", "t-SNE 2", "t-SNE 3"])
            df_plot["Class"] = labels
            
            # Строим интерактивный 3D график
            fig = px.scatter_3d(
                df_plot, 
                x="t-SNE 1", y="t-SNE 2", z="t-SNE 3",
                color="Class",
                title=f"Разбиение t-SNE в 3D (KL-дивергенция: {tsne.kl_divergence_:.4f})",
                labels={"color": label_col},
                opacity=0.75,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # ИСПРАВЛЕНО: используем правильный формат line для 3D маркеров вместо edgecolor
            fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
            
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)"),
                    yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)"),
                    zaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)")
                ),
                legend=dict(orientation="h", y=-0.05)
            )
            
            # Вывод графика на экран
            st.plotly_chart(fig, use_container_width=True)
            st.success("Вычисления завершены! Сцену можно свободно крутить мышкой.")