import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
from PIL import Image

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------
st.set_page_config(
    page_title="Análise Preditiva de Obesidade",
    page_icon="🧬",
    layout="wide",
)

# -------------------------------------------------
# PÁGINA: PAINEL ANALÍTICO
# -------------------------------------------------
def show_dashboard():

    st.markdown(
        "<h1 style='text-align:left; color:#2F7E79;'>Análise dos Fatores Associados à Obesidade</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
    Esta seção apresenta uma análise exploratória do conjunto de dados utilizado no modelo,
    com foco em apoiar a interpretação clínica e a tomada de decisão em saúde.
    Os resultados permitem identificar padrões relevantes e subsidiar ações estratégicas
    relacionadas à gestão da saúde.
    """)

    max_width = 900

    
    def show_img(path, caption):
        try:
            img = Image.open(path)
            w, h = img.size

            if w > max_width:
                scale = max_width / w
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.LANCZOS)

            st.image(img, caption=caption)
        except FileNotFoundError:
            st.warning(f"Imagem '{path}' não encontrada.")

    # 0. Distribuição geral
    st.subheader("1. Distribuição dos Níveis de Obesidade")
    show_img(
        "1_distribuicao_obesidade.png",
        "Distribuição dos níveis de obesidade na amostra"
    )
    st.markdown("""**Síntese analítica:** A distribuição dos níveis de obesidade na amostra mostra presença relevante de indivíduos em diferentes estágios, 
    incluindo graus mais avançados. Esse cenário indica a necessidade de abordagens diferenciadas, considerando que a população analisada apresenta desde 
    situações iniciais até quadros mais complexos de obesidade.""")

    # 1. Histórico familiar
    st.subheader("2. Histórico Familiar de Sobrepeso")
    show_img(
        "3b_hist_familiar_barras_lado_a_lado.png",
        "Relação entre histórico familiar de sobrepeso e nível de obesidade"
    )
    st.markdown("""
    **Síntese analítica:** A presença de histórico familiar de sobrepeso é mais frequente nos níveis mais elevados de obesidade,
    indicando possível influência genética e ambiental. Esse achado aponta para a relevância de considerar o contexto familiar 
    como um elemento de atenção no acompanhamento e na definição de estratégias preventivas.
    """)

    # 2. Atividade física
    st.subheader("3. Atividade Física")
    show_img(
        "4_atividade_fisica_vs_obesidade_barras_verde.png",
        "Frequência de atividade física por nível de obesidade"
    )
    st.markdown("""
    **Síntese analítica:** Observa-se redução da prática de atividade física conforme o nível de obesidade aumenta, 
    com maior presença de sedentarismo nos estágios mais avançados. Esse comportamento destaca o papel da atividade física 
    regular como elemento associado à manutenção do peso e à redução do risco de progressão da obesidade.
    """)

    # 3. Transporte
    st.subheader("4.Transporte Diário")
    show_img(
        "5_transporte_vs_obesidade_empilhado_verde_cinza.png",
        "Meio de transporte por nível de obesidade"
    )
    st.markdown("""
    **Síntese analítica:** O uso de transporte motorizado se torna mais comum nos níveis mais elevados de obesidade,
    enquanto formas de deslocamento ativo aparecem com menor frequência. Esse padrão sugere que a mobilidade diária pode 
    estar associada aos níveis de obesidade, reforçando a importância de escolhas de deslocamento mais ativas no cotidiano.
    """)

    # 4. Idade
    st.subheader("5. Perfil Etário")
    show_img(
        "2_idade_vs_obesidade.png",
        "Distribuição de idade por nível de obesidade"
    )
    st.markdown("""
    **Síntese analítica:** À medida que os níveis de obesidade aumentam, observa-se que a idade média dos indivíduos também tende a ser maior.
     Esse padrão sugere que o excesso de peso pode se acumular ao longo do tempo, reforçando a importância de cuidados contínuos e de ações que 
    acompanhem o indivíduo ao longo das diferentes fases da vida.
     """)


TARGET = "Nivel_obesidade"

NUM_ESPERADAS = [
    "Idade", "Altura", "Peso",
    "Frequencia_Consumo_Vegetais", "Numero_Refeicoes_Principais",
    "Consumo_Agua_Litros", "Frequencia_Atividade_Fisica",
    "Tempo_Uso_Dispositivos_Tecnologicos",
]


def parse_int(text: str):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    try:
        return int(float(t.replace(",", ".")))
    except ValueError:
        return None

def parse_float(text: str):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return None
# =================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0
if "errors" not in st.session_state:
    st.session_state.errors = {}
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False


# =================================================
# LOAD: MODELO + EXCEL + PIPELINE (PREPROCESSOR DO EXCEL)
# =================================================
@st.cache_resource
def load_models_and_data():
    # 1) Carrega o modelo salvo (best_model.pkl)
    try:
        base_model = joblib.load("best_model.pkl")
    except FileNotFoundError:
        st.error("Arquivo 'best_model.pkl' não encontrado na pasta do app.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar 'best_model.pkl': {e}")
        st.stop()

    # 2) Carrega o Excel tratado
    try:
        df = pd.read_excel("arquivo_obesidade_tratado.xlsx")
    except FileNotFoundError:
        st.error("Arquivo 'arquivo_obesidade_tratado.xlsx' não encontrado na pasta do app.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar 'arquivo_obesidade_tratado.xlsx': {e}")
        st.stop()

    if TARGET not in df.columns:
        st.error(f"Coluna alvo '{TARGET}' não existe no Excel.")
        st.stop()

    df = df.copy()

    # coerção numérica igual ao treino 
    for col in NUM_ESPERADAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    # LabelEncoder no target 
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)

    # detecta num/cat baseado no Excel tratado
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    # Fit do preprocessor no seu Excel (apenas para definir as colunas codificadas)
    preprocessor.fit(X)

    # Pipeline final (preprocess -> modelo salvo)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", base_model),
    ])

    models = {"LightGBM + XGBoost": pipe}

    meta = {
        "X_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_encoder": le_y,
        "y_encoded": y_enc,
        "num_esperadas": NUM_ESPERADAS,
    }

    return models, df, meta

def calculate_accuracy(pipe_model, df, meta):
    try:
        X = df.drop(columns=[TARGET]).copy()
        for col in meta["num_esperadas"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        y_true_enc = meta["y_encoded"]
        y_pred_enc = pipe_model.predict(X)
        return accuracy_score(y_true_enc, y_pred_enc)
    except Exception as e:
        st.warning(f"Não foi possível calcular a acurácia do modelo: {e}")
        return None

@st.cache_data
def get_model_insights_chart(model_name, _models):
    pipe = _models[model_name]
    if not hasattr(pipe, "named_steps"):
        return None
    if "preprocessor" not in pipe.named_steps or "classifier" not in pipe.named_steps:
        return None

    preprocessor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    df_importance = None
    chart_title = ""
    x_axis_title = ""

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        chart_title = "Principais Fatores por Importância"
        x_axis_title = "Nível de Importância"
        df_importance = pd.DataFrame({"feature": feature_names, "importance": importances})

    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        importances = np.abs(coef[0]) if len(coef.shape) > 1 else np.abs(coef)
        chart_title = "Principais Fatores por Impacto"
        x_axis_title = "Impacto (Coeficiente Absoluto)"
        df_importance = pd.DataFrame({"feature": feature_names, "importance": importances})

    if df_importance is None:
        return None

    df_importance = df_importance.sort_values("importance", ascending=False).head(10)

    df_importance["feature_translated"] = (
        df_importance["feature"].astype(str)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    chart = alt.Chart(df_importance).mark_bar(opacity=0.85).encode(
        x=alt.X("importance:Q", title=x_axis_title),
        y=alt.Y("feature_translated:N", sort="-x", title="Característica"),
        tooltip=[
            alt.Tooltip("feature_translated", title="Característica"),
            alt.Tooltip("importance", title="Importância", format=".4f"),
        ],
    ).properties(title=chart_title)

    return chart

# =================================================
# CONTROLE DE ESTADO
# =================================================
def reset_app():
    st.session_state.prediction_result = None
    st.session_state.edit_mode = False
    st.session_state.errors = {}
    st.session_state.reset_counter += 1

def enable_edit_mode():
    st.session_state.edit_mode = True
    st.session_state.prediction_result = None

def render_input(widget_type, label, options, key, **kwargs):
    dynamic_key = f"{key}_{st.session_state.reset_counter}"

    if st.session_state.get(dynamic_key) is not None and dynamic_key in st.session_state.errors:
        del st.session_state.errors[dynamic_key]

    if widget_type == "selectbox":
        input_widget = st.selectbox(label, options, key=dynamic_key, **kwargs)
    elif widget_type == "radio":
        input_widget = st.radio(label, options, key=dynamic_key, **kwargs)
    else:
        input_widget = None

    if dynamic_key in st.session_state.errors:
        st.markdown(
            "<p style='font-size: 12px; color: red; margin-top: -15px; margin-bottom: 5px;'>Campo obrigatório</p>",
            unsafe_allow_html=True
        )

    return input_widget

# =================================================
# NAVEGAÇÃO
# =================================================
with st.sidebar.container(border=True):
    st.subheader("Menu")
    app_mode = st.radio(
        "Escolha a funcionalidade:",
        ["Painel Analítico", "Sistema Preditivo"],
        horizontal=True,
        label_visibility="collapsed",
    )

# =================================================
# SISTEMA PREDITIVO
# =================================================
if app_mode == "Sistema Preditivo":

    models, df, meta = load_models_and_data()

    model_selection = list(models.keys())[0]
    active_model = models[model_selection]  # PIPELINE
    FEATURES = meta["X_columns"]
    le_y = meta["label_encoder"]

    # ---------------- SIDEBAR: ENTRADAS ----------------
    with st.sidebar:
        is_disabled = (st.session_state.prediction_result is not None) and not st.session_state.edit_mode
        reset_key = st.session_state.reset_counter

        st.header("Configurações da Predição")
        st.markdown(f"Modelo em uso: **{model_selection}**")

        st.divider()
        st.header("Insira os Dados para Análise")

        st.subheader("Seu perfil")

        # Opção A: começa vazio e converte
        age_txt = st.text_input("Idade", placeholder="Ex: 25", disabled=is_disabled, key=f"age_txt_{reset_key}")
        height_txt = st.text_input("Altura (m)", placeholder="Ex: 1,70", disabled=is_disabled, key=f"height_txt_{reset_key}")
        weight_txt = st.text_input("Peso (kg)", placeholder="Ex: 70,0", disabled=is_disabled, key=f"weight_txt_{reset_key}")

        age = parse_int(age_txt)
        height = parse_float(height_txt)
        weight = parse_float(weight_txt)

        # mensagens de erro
        if st.session_state.errors.get(f"age_txt_{reset_key}"):
            st.error("Informe uma idade válida.")
        if st.session_state.errors.get(f"height_txt_{reset_key}"):
            st.error("Informe uma altura válida.")
        if st.session_state.errors.get(f"weight_txt_{reset_key}"):
            st.error("Informe um peso válido.")

        gender_label = render_input(
            "radio", "Gênero",
            ["Feminino", "Masculino"],
            key="gender_input",
            index=None,
            horizontal=True,
            disabled=is_disabled,
        )

        st.subheader("Estilo de Vida e Histórico")
        family_history_label = render_input(
            "radio", "Histórico Familiar de Sobrepeso?",
            ["Sim", "Não"],
            key="family_history_input",
            index=None, horizontal=True, disabled=is_disabled
        )
        favc_label = render_input(
            "radio", "Consumo de Alimentos Calóricos?",
            ["Sim", "Não"],
            key="favc_input",
            index=None, horizontal=True, disabled=is_disabled
        )
        scc_label = render_input(
            "radio", "Monitoramento de Calorias?",
            ["Sim", "Não"],
            key="scc_input",
            index=None, horizontal=True, disabled=is_disabled
        )
        smoke_label = render_input(
            "radio", "Fumante?",
            ["Sim", "Não"],
            key="smoke_input",
            index=None, horizontal=True, disabled=is_disabled
        )

        caec_label = render_input(
            "selectbox", "Consumo de Alimentos Entre Refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"],
            key="caec_input",
            index=None, placeholder="Selecione...", disabled=is_disabled
        )
        calc_label = render_input(
            "selectbox", "Consumo de Álcool?",
            ["Não", "Às vezes", "Frequentemente"],
            key="calc_input",
            index=None, placeholder="Selecione...", disabled=is_disabled
        )
        mtrans_label = render_input(
            "selectbox", "Meio de Transporte Principal?",
            ["Transporte Público", "Automóvel", "Caminhando", "Moto", "Bicicleta"],
            key="mtrans_input",
            index=None, placeholder="Selecione...", disabled=is_disabled
        )

        st.subheader("Hábitos Diários")
        fcvc = st.slider("Frequência de consumo de vegetais(1-3)?", 1, 3, 1, help="1 = Raramente • 2 = Às vezes • 3 = Sempre", disabled=is_disabled, key=f"fcvc_{reset_key}")
        ncp = st.slider("Nº de refeições principais (1–4)?", 1, 4, 1, disabled=is_disabled, key=f"ncp_{reset_key}")
        ch2o = st.slider("Consumo de água - litros/dia (1–3)?", 1, 3, 1, disabled=is_disabled, key=f"ch2o_{reset_key}")
        faf = st.slider("Atividade física - dias/semana 0–3)?", 0, 3, 0, help="0 = nenhuma, 1 = 1–2×/sem, 2 = 3–4×/sem, 3 = 5×/sem ou mais", disabled=is_disabled, key=f"faf_{reset_key}")
        tue = st.slider("Tempo de uso de telas- horas/dia (0–2)?", 0, 2, 0, disabled=is_disabled, key=f"tue_{reset_key}")

    # ---------------- ÁREA PRINCIPAL ----------------
    accuracy = calculate_accuracy(active_model, df.copy(), meta)
    model_insights_chart = get_model_insights_chart(model_selection, models)

    st.markdown(
        "<h1 style='text-align: left; color: #2F7E79;'>Análise Personalizada do Risco de Obesidade</h1>",
        unsafe_allow_html=True,
    )

    sub_header_col, metric_col = st.columns([4, 1])
    with sub_header_col:
        st.markdown("Preencha os dados na barra lateral à esquerda e clique no botão abaixo para realizar a predição.")
    with metric_col:
        if accuracy is not None:
            st.metric(label=f"Acurácia ({model_selection})", value=f"{accuracy*100:.2f}%")

    st.markdown("---")

    if st.session_state.errors:
        st.warning("⚠️ Por favor, revise os campos destacados.")

    button_placeholder = st.empty()

    # ------------- BOTÃO DE PREDIÇÃO -------------
    if st.session_state.prediction_result is None or st.session_state.edit_mode:
        if button_placeholder.button("**Realizar Predição**", type="primary", use_container_width=True):

            st.session_state.errors = {}
            current_reset_key = st.session_state.reset_counter

            # valida numéricos (opção A)
            errors = {}
            if age is None or not (1 <= age <= 100):
                errors[f"age_txt_{current_reset_key}"] = True
            if height is None or not (1.0 <= height <= 2.5):
                errors[f"height_txt_{current_reset_key}"] = True
            if weight is None or not (30.0 <= weight <= 200.0):
                errors[f"weight_txt_{current_reset_key}"] = True

            # valida categóricos
            inputs_to_validate = {
                f"gender_input_{current_reset_key}": gender_label,
                f"family_history_input_{current_reset_key}": family_history_label,
                f"favc_input_{current_reset_key}": favc_label,
                f"scc_input_{current_reset_key}": scc_label,
                f"smoke_input_{current_reset_key}": smoke_label,
                f"caec_input_{current_reset_key}": caec_label,
                f"calc_input_{current_reset_key}": calc_label,
                f"mtrans_input_{current_reset_key}": mtrans_label,
            }

            for k, v in inputs_to_validate.items():
                if v is None:
                    errors[k] = True

            if errors:
                st.session_state.errors = errors
                st.rerun()

            st.session_state.edit_mode = False

            #  nomes EXATOS do seu Excel tratado
            input_values_excel = {
                "Idade": age,
                "Altura": height,
                "Peso": weight,
                "Genero": gender_label,
                "Historico_Familiar_Sobrepeso": family_history_label,
                "Consumo_Alimentos_Caloricos": favc_label,
                "Monitoramento_Calorias": scc_label,
                "Fumante": smoke_label,
                "Consumo_Alimentos_Entre_Refeicoes": caec_label,
                "Consumo_Alcool": calc_label,
                "Meio_Transporte": mtrans_label,
                "Frequencia_Consumo_Vegetais": float(fcvc),
                "Numero_Refeicoes_Principais": float(ncp),
                "Consumo_Agua_Litros": float(ch2o),
                "Frequencia_Atividade_Fisica": float(faf),
                "Tempo_Uso_Dispositivos_Tecnologicos": float(tue),
            }

            input_data = pd.DataFrame([input_values_excel]).reindex(columns=FEATURES)

            # coerção numérica igual ao treino
            for col in NUM_ESPERADAS:
                if col in input_data.columns:
                    input_data[col] = pd.to_numeric(input_data[col], errors="coerce")

            with st.spinner(f"Analisando os dados com o modelo {model_selection}..."):
                pred_enc = active_model.predict(input_data)[0]
                prediction_label = le_y.inverse_transform([pred_enc])[0]

                prediction_proba = None
                if hasattr(active_model, "predict_proba"):
                    try:
                        prediction_proba = active_model.predict_proba(input_data)
                    except Exception:
                        prediction_proba = None

                report_values = {
                    "Historico_Familiar_Sobrepeso": family_history_label,
                    "Consumo_Alimentos_Caloricos": favc_label,
                    "Frequencia_Consumo_Vegetais": fcvc,
                    "Numero_Refeicoes_Principais": ncp,
                    "Consumo_Alimentos_Entre_Refeicoes": caec_label,
                    "Fumante": smoke_label,
                    "Consumo_Agua_Litros": ch2o,
                    "Monitoramento_Calorias": scc_label,
                    "Frequencia_Atividade_Fisica": faf,
                    "Tempo_Uso_Dispositivos_Tecnologicos": tue,
                    "Meio_Transporte": mtrans_label,
                }

                st.session_state.prediction_result = (
                    prediction_label,
                    prediction_proba,
                    report_values,
                    model_selection,
                )
                st.rerun()

    else:
        col1_btn, col2_btn = button_placeholder.columns(2)
        col1_btn.button("**⬅️ Realizar Nova Predição**", use_container_width=True, on_click=reset_app)
        col2_btn.button("**📝 Editar Dados Informados**", use_container_width=True, on_click=enable_edit_mode)

    # ------------- EXIBIÇÃO DO RESULTADO -------------
    if st.session_state.prediction_result is not None:
        prediction_label, prediction_proba, input_values, used_model = st.session_state.prediction_result

        st.markdown(
            f"<h2 style='text-align: center;'>Resultado da Predição (Modelo: {used_model})</h2>",
            unsafe_allow_html=True,
        )

        # Ajuste as chaves conforme seus rótulos reais do Excel
        color_map = {
            "Peso Normal": "#2ECC71",
            "Sobrepeso I": "#F1C40F",
            "Sobrepeso II": "#E67E22",
            "Obesidade I": "#E74C3C",
            "Obesidade II": "#C0392B",
            "Obesidade III": "#A93226",
            "Abaixo do peso": "#3498DB",
        }

        result_color = color_map.get(prediction_label, "#34495E")

        st.markdown(
            f"<h2 style='text-align: center; color: {result_color};'>{prediction_label}</h2>",
            unsafe_allow_html=True,
        )

        if prediction_proba is not None:
            st.markdown(
                f"<p style='text-align: center;'>Confiança do modelo no resultado: "
                f"<strong>{np.max(prediction_proba)*100:.2f}%</strong>.</p>",
                unsafe_allow_html=True,
            )

        # =========================
        # ✅ JANELA (EXPANDER)
        # =========================
        _, center_col, _ = st.columns([0.5, 3, 0.5])
        with center_col:
            st.markdown(
                """
                <style>
                div[data-testid="stExpander"] summary {
                    position: relative;
                    background-color: #2F7E79;
                    color: white;
                    border-radius: 0.25rem;
                }
                div[data-testid="stExpander"] summary p {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-size: 18px;
                    font-weight: 600;
                    width: 90%;
                    text-align: center;
                }
                div[data-testid="stExpander"] summary svg {
                    fill: white;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("🔎 Clique para ver a análise detalhada dos seus hábitos"):
                st.markdown("<h4 style='text-align: center;'>Análise de Hábitos</h4>", unsafe_allow_html=True)

                risk_factors, protective_factors = [], []

                if input_values["Historico_Familiar_Sobrepeso"] == "Sim":
                    risk_factors.append("Possui histórico familiar de sobrepeso.")
                else:
                    protective_factors.append("Não possui histórico familiar de sobrepeso (segundo o informado).")

                if input_values["Consumo_Alimentos_Caloricos"] == "Sim":
                    risk_factors.append("Consome alimentos calóricos com frequência.")
                else:
                    protective_factors.append("Relata baixo consumo de alimentos calóricos.")

                if input_values["Frequencia_Consumo_Vegetais"] < 2:
                    risk_factors.append("Baixo consumo de vegetais.")
                else:
                    protective_factors.append("Bom consumo de vegetais.")

                if input_values["Numero_Refeicoes_Principais"] < 3:
                    risk_factors.append("Faz menos de 3 refeições principais.")
                else:
                    protective_factors.append("Faz 3 ou mais refeições principais.")

                if input_values["Consumo_Alimentos_Entre_Refeicoes"] in ["Frequentemente", "Sempre"]:
                    risk_factors.append("Lanches entre refeições com alta frequência.")

                if input_values["Fumante"] == "Sim":
                    risk_factors.append("É fumante.")
                else:
                    protective_factors.append("Não é fumante.")

                if input_values["Consumo_Agua_Litros"] < 2:
                    risk_factors.append("Baixo consumo de água.")
                else:
                    protective_factors.append("Bom consumo de água.")

                if input_values["Monitoramento_Calorias"] == "Sim":
                    protective_factors.append("Monitora o consumo de calorias.")
                else:
                    risk_factors.append("Não monitora calorias.")

                if input_values["Frequencia_Atividade_Fisica"] < 2:
                    risk_factors.append("Baixa frequência de atividade física.")
                else:
                    protective_factors.append("Boa frequência de atividade física.")

                if input_values["Tempo_Uso_Dispositivos_Tecnologicos"] > 1:
                    risk_factors.append("Muito tempo em dispositivos/telas.")
                else:
                    protective_factors.append("Tempo de telas moderado.")

                if input_values["Meio_Transporte"] in ["Automóvel", "Transporte Público"]:
                    risk_factors.append("Transporte mais associado a sedentarismo.")
                elif input_values["Meio_Transporte"] in ["Caminhando", "Bicicleta"]:
                    protective_factors.append("Transporte ativo (caminhada/bicicleta).")

                col_risk, col_prot = st.columns(2)
                with col_risk:
                    st.markdown("<h5 style='color:#E74C3C;'>🔴 Fatores de Risco</h5>", unsafe_allow_html=True)
                    if risk_factors:
                        for f in risk_factors:
                            st.markdown(f"- {f}")
                    else:
                        st.markdown("- Nenhum fator de risco óbvio identificado.")

                with col_prot:
                    st.markdown("<h5 style='color:#2ECC71;'>🟢 Fatores Protetivos</h5>", unsafe_allow_html=True)
                    if protective_factors:
                        for f in protective_factors:
                            st.markdown(f"- {f}")
                    else:
                        st.markdown("- Nenhum fator protetivo óbvio identificado.")

                st.markdown("<hr>", unsafe_allow_html=True)

                if prediction_proba is not None:
                    st.markdown("<h4 style='text-align:center;'>Probabilidade por Classe</h4>", unsafe_allow_html=True)
                    proba = prediction_proba[0]
                    encoded_classes = active_model.named_steps["classifier"].classes_
                    decoded_classes = le_y.inverse_transform(encoded_classes)

                    df_proba = pd.DataFrame({
                        "Classe": decoded_classes,
                        "Probabilidade": proba
                    }).sort_values("Probabilidade", ascending=False)

                    df_proba["Probabilidade"] = df_proba["Probabilidade"].apply(lambda p: f"{p*100:.2f}%")
                    st.dataframe(df_proba, hide_index=True, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ✅ “Janela” do modelo (popover)
            if model_insights_chart:
                with st.popover(f"Ver Análise de Fatores do Modelo ({used_model})", use_container_width=True):
                    st.altair_chart(model_insights_chart, use_container_width=True)

# =================================================
# PAINEL ANALÍTICO
# =================================================
elif app_mode == "Painel Analítico":
    show_dashboard()