from pathlib import Path
from dash import Dash, html, dcc, Input, Output

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    from dashboard.utils import load_data, compute_kpis
except ImportError:
    from utils import load_data, compute_kpis


# =========================================================
# CONFIG
# =========================================================
PRIMARY_COLOR = "#811433"
SECONDARY_COLOR = "#d9a5b3"
ACCENT_COLOR = "#f3d7df"
BG_COLOR = "#f8f4f6"
CARD_COLOR = "#ffffff"
TEXT_MUTED = "#6b4d57"
TEXT_DARK = "#1f1f1f"
SUCCESS_COLOR = "#2e8b57"
WARNING_COLOR = "#d97a00"
DANGER_COLOR = "#c0392b"

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "reports" / "rapport_hospitalisation.html"

# =========================================================
# DONNÉES
# =========================================================
df = load_data()

reg_results_df = pd.DataFrame([
    ["Gradient Boosting", 768.56, 974.48, 0.83],
    ["Linear Regression", 792.10, 1016.21, 0.82],
    ["Random Forest", 773.61, 1029.66, 0.81],
], columns=["Modèle", "MAE", "RMSE", "R²"])

clf_results_df = pd.DataFrame([
    ["Logistic Regression", 1.00, 1.00, 1.00, 1.00],
    ["Random Forest", 1.00, 1.00, 1.00, 1.00],
    ["Decision Tree", 1.00, 1.00, 1.00, 1.00],
], columns=["Modèle", "Accuracy", "Precision", "Recall", "F1-score"])

fi_reg = pd.DataFrame([
    ["DureeSejour", 0.8305307614261574],
    ["Age", 0.0513140688925478],
    ["Traitement_Antibiotiques", 0.010255385531352232],
    ["Maladie_Cancer", 0.009272464443786498],
    ["Traitement_Soins spéciaux", 0.00775568208477175],
    ["Departement_Neurologie", 0.007098045783511487],
    ["Departement_Oncologie", 0.007077434019018367],
    ["Traitement_Physiothérapie", 0.005358737802814489],
    ["Traitement_Chirurgie", 0.005352757094231359],
    ["Maladie_Hypertension", 0.005207347329501489],
], columns=["Feature", "Importance"])

fi_clf = pd.DataFrame([
    ["DureeSejour", 0.77439493],
    ["Age", 0.06605217],
    ["Variable_3", 0.00934389],
    ["Variable_4", 0.00818688],
    ["Variable_5", 0.00807206],
    ["Variable_6", 0.00801195],
    ["Variable_7", 0.00797115],
    ["Variable_8", 0.00791145],
    ["Variable_9", 0.00775605],
    ["Variable_10", 0.00764234],
], columns=["Feature", "Importance"])

# =========================================================
# HELPERS
# =========================================================
def get_kpi_color(title, value):
    try:
        value_float = float(value)
    except Exception:
        value_float = None

    if title == "Anomalies":
        if value_float is None:
            return PRIMARY_COLOR
        if value_float == 0:
            return SUCCESS_COLOR
        if value_float <= 10:
            return WARNING_COLOR
        return DANGER_COLOR

    if title == "Durée moyenne":
        if value_float is None:
            return PRIMARY_COLOR
        if value_float <= 7:
            return SUCCESS_COLOR
        if value_float <= 10:
            return WARNING_COLOR
        return DANGER_COLOR

    if title == "Coût moyen":
        if value_float is None:
            return PRIMARY_COLOR
        if value_float <= 4000:
            return SUCCESS_COLOR
        if value_float <= 7000:
            return WARNING_COLOR
        return DANGER_COLOR

    return PRIMARY_COLOR


def kpi_card(card_id, title, value, subtitle):
    accent = get_kpi_color(title, value)

    return html.Div(
        id=card_id,
        style={
            "background": "linear-gradient(180deg, #ffffff 0%, #fff8fa 100%)",
            "borderRadius": "20px",
            "padding": "22px",
            "boxShadow": "0 12px 0 rgba(129,20,51,0.08), 0 18px 28px rgba(0,0,0,0.08)",
            "border": "1px solid rgba(129,20,51,0.08)",
            "minHeight": "145px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
            "position": "relative",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                style={
                    "position": "absolute",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "6px",
                    "backgroundColor": accent,
                }
            ),
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "color": TEXT_MUTED,
                    "fontWeight": "bold",
                    "marginBottom": "10px",
                    "marginTop": "4px",
                },
            ),
            html.Div(
                str(value),
                style={
                    "fontSize": "30px",
                    "fontWeight": "bold",
                    "color": accent,
                    "marginBottom": "6px",
                    "lineHeight": "1.1",
                },
            ),
            html.Div(subtitle, style={"fontSize": "14px", "color": "#555"}),
        ],
    )


def graph_card(graph_id, figure):
    return html.Div(
        style={
            "backgroundColor": CARD_COLOR,
            "borderRadius": "20px",
            "padding": "16px",
            "boxShadow": "0 10px 24px rgba(0,0,0,0.06)",
            "border": "1px solid rgba(129,20,51,0.08)",
        },
        children=[
            dcc.Graph(id=graph_id, figure=figure, config={"displayModeBar": False})
        ],
    )


def section_title(title, subtitle=None):
    children = [
        html.H2(
            title,
            style={
                "color": PRIMARY_COLOR,
                "marginBottom": "6px",
                "fontSize": "28px",
                "fontWeight": "700",
            },
        )
    ]
    if subtitle:
        children.append(
            html.P(
                subtitle,
                style={"color": TEXT_MUTED, "marginTop": "0", "marginBottom": "18px"},
            )
        )
    return html.Div(children)


def info_panel(title, items):
    return html.Div(
        style={
            "backgroundColor": "#fffafc",
            "borderLeft": f"6px solid {PRIMARY_COLOR}",
            "borderRadius": "18px",
            "padding": "20px",
            "boxShadow": "0 10px 24px rgba(0,0,0,0.06)",
            "border": "1px solid rgba(129,20,51,0.08)",
        },
        children=[
            html.H3(title, style={"color": PRIMARY_COLOR, "marginTop": "0"}),
            html.Ul(
                [html.Li(item) for item in items],
                style={"paddingLeft": "20px", "marginBottom": "0"},
            ),
        ],
    )


def filter_dataframe(base_df, dept, maladie, sexe, age):
    filtered_df = base_df.copy()

    if dept:
        filtered_df = filtered_df[filtered_df["Departement"] == dept]
    if maladie:
        filtered_df = filtered_df[filtered_df["Maladie"] == maladie]
    if sexe:
        filtered_df = filtered_df[filtered_df["Sexe"] == sexe]
    if age:
        filtered_df = filtered_df[filtered_df["TrancheAge"].astype(str) == str(age)]

    return filtered_df


# =========================================================
# FIGURES
# =========================================================
def build_age_figure(filtered_df):
    fig = px.histogram(
        filtered_df,
        x="Age",
        nbins=20,
        title="Distribution de l'âge",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(
        template="plotly_white",
        bargap=0.15,
        title_font_color=PRIMARY_COLOR,
        xaxis_title="Âge",
        yaxis_title="Nombre de patients",
    )
    fig.update_traces(marker_line_color="white", marker_line_width=1.2)
    return fig


def build_sex_figure(filtered_df):
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Répartition par sexe", template="plotly_white")
        return fig

    fig = px.pie(
        filtered_df,
        names="Sexe",
        title="Répartition par sexe",
        color="Sexe",
        color_discrete_map={"M": PRIMARY_COLOR, "F": SECONDARY_COLOR},
    )
    fig.update_traces(textinfo="percent+label+value")
    fig.update_layout(template="plotly_white", title_font_color=PRIMARY_COLOR)
    return fig


def build_dept_figure(filtered_df):
    dept_counts = filtered_df["Departement"].value_counts().reset_index()
    dept_counts.columns = ["Departement", "Nombre"]

    if dept_counts.empty:
        dept_counts = pd.DataFrame({"Departement": ["Aucune donnée"], "Nombre": [0]})

    dept_counts = dept_counts.sort_values("Nombre", ascending=True)

    fig = px.bar(
        dept_counts,
        x="Nombre",
        y="Departement",
        orientation="h",
        title="Départements les plus sollicités",
        text="Nombre",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(
        template="plotly_white",
        title_font_color=PRIMARY_COLOR,
        xaxis_title="Nombre de patients",
        yaxis_title="Département",
    )
    fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1.2)
    return fig


def build_cost_duration_figure(filtered_df):
    fig = px.scatter(
        filtered_df,
        x="DureeSejour",
        y="Cout",
        color="Departement",
        title="Relation entre durée de séjour et coût",
        hover_data=["Age", "Sexe", "Maladie", "Traitement"],
    )
    fig.update_layout(
        template="plotly_white",
        title_font_color=PRIMARY_COLOR,
        xaxis_title="Durée de séjour",
        yaxis_title="Coût",
    )
    return fig


def build_cost_by_dept_figure(filtered_df):
    dept_cost = filtered_df.groupby("Departement", as_index=False)["Cout"].mean()
    if dept_cost.empty:
        dept_cost = pd.DataFrame({"Departement": ["Aucune donnée"], "Cout": [0]})
    dept_cost = dept_cost.sort_values("Cout", ascending=True)

    fig = px.bar(
        dept_cost,
        x="Cout",
        y="Departement",
        orientation="h",
        title="Coût moyen par département",
        text=dept_cost["Cout"].round(2),
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(
        template="plotly_white",
        title_font_color=PRIMARY_COLOR,
        xaxis_title="Coût moyen",
        yaxis_title="Département",
    )
    fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1.2)
    return fig


def build_top_maladies_figure(filtered_df):
    mal_counts = filtered_df["Maladie"].value_counts().reset_index()
    mal_counts.columns = ["Maladie", "Nombre"]
    if mal_counts.empty:
        mal_counts = pd.DataFrame({"Maladie": ["Aucune donnée"], "Nombre": [0]})
    mal_counts = mal_counts.sort_values("Nombre", ascending=True)

    fig = px.bar(
        mal_counts,
        x="Nombre",
        y="Maladie",
        orientation="h",
        title="Maladies les plus fréquentes",
        text="Nombre",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(
        template="plotly_white",
        title_font_color=PRIMARY_COLOR,
        xaxis_title="Nombre de cas",
        yaxis_title="Maladie",
    )
    fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1.2)
    return fig


def build_monthly_figure(filtered_df):
    monthly = filtered_df.groupby("MoisAdmission", as_index=False).agg(
        Admissions=("PatientID", "count") if "PatientID" in filtered_df.columns else ("Age", "count"),
        CoutMoyen=("Cout", "mean"),
    )
    if monthly.empty:
        monthly = pd.DataFrame({"MoisAdmission": ["Aucune donnée"], "Admissions": [0], "CoutMoyen": [0]})

    fig = px.bar(
        monthly,
        x="MoisAdmission",
        y="Admissions",
        title="Admissions par mois",
        color_discrete_sequence=[PRIMARY_COLOR],
        text="Admissions",
    )
    fig.update_layout(
        template="plotly_white",
        title_font_color=PRIMARY_COLOR,
        xaxis_title="Mois",
        yaxis_title="Admissions",
    )
    fig.update_traces(textposition="outside")
    return fig


def build_reg_results_figure():
    fig = px.bar(
        reg_results_df,
        x="Modèle",
        y="R²",
        title="Performance des modèles de régression (R²)",
        text="R²",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(template="plotly_white", title_font_color=PRIMARY_COLOR)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig


def build_fi_reg_figure():
    data = fi_reg.sort_values("Importance", ascending=True)
    fig = px.bar(
        data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Variables importantes - prédiction du coût",
        text=data["Importance"].round(3),
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(template="plotly_white", title_font_color=PRIMARY_COLOR)
    fig.update_traces(textposition="outside")
    return fig


def build_fi_clf_figure():
    data = fi_clf.sort_values("Importance", ascending=True)
    fig = px.bar(
        data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Facteurs associés aux séjours longs",
        text=data["Importance"].round(3),
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(template="plotly_white", title_font_color=PRIMARY_COLOR)
    fig.update_traces(textposition="outside")
    return fig


# =========================================================
# INSIGHTS / RECOMMANDATIONS
# =========================================================
def generate_insights_component(filtered_df):
    if filtered_df.empty:
        return html.Div(
            "Aucune donnée disponible pour les filtres sélectionnés.",
            style={"color": PRIMARY_COLOR, "fontWeight": "bold"},
        )

    duree = round(filtered_df["DureeSejour"].mean(), 2)
    cout = round(filtered_df["Cout"].mean(), 2)
    dept_top = filtered_df["Departement"].mode().iloc[0]
    maladie_top = filtered_df["Maladie"].mode().iloc[0]
    proportion_long = round(filtered_df["SejourLong"].mean() * 100, 2)

    return html.Div(
        children=[
            html.H4("Insights automatiques", style={"color": PRIMARY_COLOR, "marginTop": "0"}),
            html.Ul(
                [
                    html.Li(f"La durée moyenne de séjour est de {duree} jours."),
                    html.Li(f"Le coût moyen est de {cout} FCFA."),
                    html.Li(f"Le département dominant est {dept_top}."),
                    html.Li(f"La maladie la plus fréquente est {maladie_top}."),
                    html.Li(f"La proportion de séjours longs est de {proportion_long} %."),
                    html.Li("La durée de séjour reste le principal facteur influençant les coûts."),
                ],
                style={"paddingLeft": "18px", "marginBottom": "0"},
            ),
        ]
    )


def generate_decision_aid(filtered_df):
    if filtered_df.empty:
        return [
            "Aucune recommandation disponible, car aucun enregistrement ne correspond aux filtres sélectionnés."
        ]

    duree = filtered_df["DureeSejour"].mean()
    cout = filtered_df["Cout"].mean()
    anomalies = int((filtered_df["Anomalie"] == -1).sum())

    recommandations = []

    if duree > 10:
        recommandations.append(
            "Prioriser l’optimisation des parcours de soins : la durée moyenne observée est élevée."
        )
    else:
        recommandations.append(
            "La durée moyenne reste maîtrisée, mais un suivi des cas longs reste recommandé."
        )

    if cout > 7000:
        recommandations.append(
            "Le coût moyen est élevé : cibler les services et pathologies générant les dépenses les plus fortes."
        )
    else:
        recommandations.append(
            "Le coût moyen reste globalement modéré : renforcer le pilotage préventif plutôt que correctif."
        )

    if anomalies > 0:
        recommandations.append(
            f"{anomalies} anomalies ont été détectées : vérifier les cas atypiques pour améliorer la qualité de la décision."
        )
    else:
        recommandations.append(
            "Aucune anomalie détectée sur le périmètre filtré : les données paraissent cohérentes."
        )

    recommandations.append(
        "Utiliser la prédiction du coût comme outil d’aide à la planification budgétaire."
    )

    return recommandations


def generate_auto_recommendations(filtered_df):
    if filtered_df.empty:
        return [
            "Aucune recommandation automatique disponible pour cette sélection."
        ]

    dept_top = filtered_df["Departement"].mode().iloc[0]
    maladie_top = filtered_df["Maladie"].mode().iloc[0]
    long_rate = round(filtered_df["SejourLong"].mean() * 100, 2)

    return [
        f"Renforcer la vigilance sur le département {dept_top}, actuellement le plus représenté.",
        f"Mettre en place un suivi plus fin des patients liés à la pathologie {maladie_top}.",
        f"Surveiller les séjours longs : leur proportion actuelle est de {long_rate} %.",
        "Concentrer les efforts d’optimisation sur la réduction des durées de séjour.",
        "Prévoir un reporting régulier à partir des indicateurs du dashboard."
    ]


# =========================================================
# APP
# =========================================================
initial_kpis = compute_kpis(df)

app = Dash(__name__)
app.title = "Dashboard Hospitalier Intelligent"
server = app.server

tabs_style = {
    "height": "52px",
    "borderRadius": "14px",
    "backgroundColor": "#f6ebef",
    "padding": "6px",
    "marginBottom": "20px",
}

tab_style = {
    "backgroundColor": "#f6ebef",
    "border": "none",
    "padding": "12px",
    "fontWeight": "600",
    "color": TEXT_MUTED,
    "borderRadius": "12px",
}

tab_selected_style = {
    "backgroundColor": PRIMARY_COLOR,
    "color": "white",
    "border": "none",
    "padding": "12px",
    "fontWeight": "700",
    "borderRadius": "12px",
}

app.layout = html.Div(
    style={
        "backgroundColor": BG_COLOR,
        "height": "100vh",
        "padding": "20px",
        "fontFamily": "Arial, sans-serif",
        "overflow": "hidden",
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1400px",
                "height": "100%",
                "margin": "0 auto",
                "display": "grid",
                "gridTemplateColumns": "290px 1fr",
                "gap": "22px",
                "alignItems": "stretch",
            },
            children=[
                # SIDEBAR
                html.Div(
                    style={
                        "height": "100%",
                        "background": "linear-gradient(180deg, #ffffff 0%, #fff8fa 100%)",
                        "borderRadius": "24px",
                        "padding": "14px",
                        "boxShadow": "0 12px 30px rgba(0,0,0,0.07)",
                        "border": "1px solid rgba(129,20,51,0.08)",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "flex-start",
                        "overflow": "hidden",
                    },
                    children=[
                        html.Div(
                            style={
                                "flex": "unset",
                                "minHeight": "0",
                                "overflowY": "visible",
                                "paddingRight": "0px",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "14px",
                                        "marginBottom": "10px",
                                    },
                                    children=[
                                        html.Img(
                                            src="/assets/logo.png",
                                            style={
                                                "height": "46px",
                                                "width": "46px",
                                                "objectFit": "contain",
                                                "borderRadius": "12px",
                                                "backgroundColor": "rgba(129,20,51,0.06)",
                                                "padding": "6px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Hospital AI",
                                                    style={"margin": "0", "color": PRIMARY_COLOR, "fontSize": "16px"},
                                                ),
                                                html.P(
                                                    "Pilotage intelligent",
                                                    style={
                                                        "margin": "0",
                                                        "fontSize": "12px",
                                                        "color": TEXT_MUTED,
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                ),

                                html.H4(
                                    "Navigation",
                                    style={"color": PRIMARY_COLOR, "marginBottom": "6px", "marginTop": "0"},
                                ),
                                html.Ul(
                                    [
                                        html.Li("Vue d’ensemble"),
                                        html.Li("Analyse exploratoire"),
                                        html.Li("Aide à la décision"),
                                        html.Li("Machine Learning"),
                                    ],
                                    style={"paddingLeft": "18px", "color": TEXT_DARK, "marginBottom": "10px"},
                                ),

                                html.Hr(style={"borderColor": "#f1dce3", "margin": "10px 0"}),

                                html.H4(
                                    "Filtres",
                                    style={"color": PRIMARY_COLOR, "marginBottom": "8px", "marginTop": "0"},
                                ),
                                dcc.Dropdown(
                                    id="departement-filter",
                                    options=[{"label": d, "value": d} for d in sorted(df["Departement"].dropna().unique())],
                                    placeholder="Département",
                                    clearable=True,
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="maladie-filter",
                                    options=[{"label": m, "value": m} for m in sorted(df["Maladie"].dropna().unique())],
                                    placeholder="Maladie",
                                    clearable=True,
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="sexe-filter",
                                    options=[{"label": s, "value": s} for s in sorted(df["Sexe"].dropna().unique())],
                                    placeholder="Sexe",
                                    clearable=True,
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Dropdown(
                                    id="age-filter",
                                    options=[{"label": str(t), "value": str(t)} for t in df["TrancheAge"].dropna().unique()],
                                    placeholder="Tranche d’âge",
                                    clearable=True,
                                    style={"marginBottom": "8px"},
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                "paddingTop": "10px",
                                "borderTop": "1px solid #f1dce3",
                                "marginTop": "0px",
                            },
                            children=[
                                html.Button(
                                    "Réinitialiser",
                                    id="reset-button",
                                    style={
                                        "backgroundColor": PRIMARY_COLOR,
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px 12px",
                                        "borderRadius": "10px",
                                        "cursor": "pointer",
                                        "fontWeight": "bold",
                                        "width": "100%",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Button(
                                    "Télécharger le rapport",
                                    id="download-report-btn",
                                    style={
                                        "backgroundColor": "#ffffff",
                                        "color": PRIMARY_COLOR,
                                        "border": f"1px solid {PRIMARY_COLOR}",
                                        "padding": "10px 12px",
                                        "borderRadius": "10px",
                                        "cursor": "pointer",
                                        "fontWeight": "bold",
                                        "width": "100%",
                                    },
                                ),
                                dcc.Download(id="download-report"),
                            ],
                        ),
                    ],
                ),

                # CONTENU PRINCIPAL
                html.Div(
                    style={
                        "height": "100%",
                        "overflowY": "auto",
                        "paddingRight": "6px",
                    },
                    children=[
                        html.Div(
                            style={
                                "background": f"linear-gradient(135deg, {PRIMARY_COLOR}, #a51f4c)",
                                "padding": "32px",
                                "borderRadius": "24px",
                                "color": "white",
                                "marginBottom": "26px",
                                "boxShadow": "0 12px 30px rgba(129,20,51,0.18)",
                            },
                            children=[
                                html.H1(
                                    "Dashboard Hospitalier Intelligent",
                                    style={"margin": "0", "fontSize": "36px"},
                                ),
                                html.P(
                                    "Application d’analyse, d’aide à la décision et de pilotage hospitalier basée sur la data et le machine learning.",
                                    style={"marginTop": "10px", "fontSize": "17px", "maxWidth": "900px"},
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                "backgroundColor": CARD_COLOR,
                                "borderRadius": "20px",
                                "padding": "22px",
                                "marginBottom": "24px",
                                "boxShadow": "0 10px 24px rgba(0,0,0,0.06)",
                                "border": "1px solid rgba(129,20,51,0.08)",
                            },
                            children=[
                                html.H3("Introduction", style={"color": PRIMARY_COLOR, "marginTop": "0"}),
                                html.P(
                                    "Ce dashboard a été conçu pour transformer les données hospitalières en informations exploitables. "
                                    "Il permet d’explorer les profils patients, d’identifier les facteurs influençant les coûts et les séjours longs, "
                                    "et de fournir une aide à la décision à travers des indicateurs dynamiques, des visualisations interactives et une lecture métier."
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(3, 1fr)",
                                "gap": "20px",
                                "marginBottom": "25px",
                            },
                            children=[
                                kpi_card("patients-card", "Patients", initial_kpis["patients"], "Observations analysées"),
                                kpi_card("age-card", "Âge moyen", initial_kpis["age_moyen"], "Âge moyen des patients"),
                                kpi_card("duree-card", "Durée moyenne", initial_kpis["duree_moyenne"], "Jours d’hospitalisation"),
                                kpi_card("cout-card", "Coût moyen", initial_kpis["cout_moyen"], "Coût moyen du séjour"),
                                kpi_card("dept-card", "Département dominant", initial_kpis["departement_dominant"], "Le plus sollicité"),
                                kpi_card("anomalie-card", "Anomalies", initial_kpis["anomalies"], "Détectées par Isolation Forest"),
                            ],
                        ),

                        dcc.Tabs(
                            id="main-tabs",
                            value="tab-overview",
                            parent_style=tabs_style,
                            children=[
                                dcc.Tab(label="Vue d’ensemble", value="tab-overview", style=tab_style, selected_style=tab_selected_style),
                                dcc.Tab(label="Analyse", value="tab-analysis", style=tab_style, selected_style=tab_selected_style),
                                dcc.Tab(label="Décision", value="tab-decision", style=tab_style, selected_style=tab_selected_style),
                                dcc.Tab(label="Machine Learning", value="tab-ml", style=tab_style, selected_style=tab_selected_style),
                            ],
                        ),

                        html.Div(id="tab-content"),
                    ],
                ),
            ],
        )
    ],
)


# =========================================================
# CALLBACK RESET
# =========================================================
@app.callback(
    Output("departement-filter", "value"),
    Output("maladie-filter", "value"),
    Output("sexe-filter", "value"),
    Output("age-filter", "value"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    return None, None, None, None


# =========================================================
# DOWNLOAD RAPPORT
# =========================================================
@app.callback(
    Output("download-report", "data"),
    Input("download-report-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_report(n_clicks):
    if REPORT_PATH.exists():
        return dcc.send_file(str(REPORT_PATH))
    return None


# =========================================================
# TAB CONTENT
# =========================================================
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("departement-filter", "value"),
    Input("maladie-filter", "value"),
    Input("sexe-filter", "value"),
    Input("age-filter", "value"),
)
def render_tab_content(active_tab, dept, maladie, sexe, age):
    filtered_df = filter_dataframe(df, dept, maladie, sexe, age)

    if active_tab == "tab-overview":
        return html.Div(
            children=[
                html.Div(
                    id="insights-box",
                    style={
                        "backgroundColor": "#fffafc",
                        "padding": "22px",
                        "borderRadius": "20px",
                        "marginBottom": "24px",
                        "boxShadow": "0 10px 24px rgba(0,0,0,0.06)",
                        "borderLeft": f"6px solid {PRIMARY_COLOR}",
                    },
                    children=generate_insights_component(filtered_df),
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                        "marginBottom": "20px",
                    },
                    children=[
                        graph_card("overview-age-graph", build_age_figure(filtered_df)),
                        graph_card("overview-sex-graph", build_sex_figure(filtered_df)),
                    ],
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                    },
                    children=[
                        graph_card("overview-dept-graph", build_dept_figure(filtered_df)),
                        graph_card("overview-cost-graph", build_cost_duration_figure(filtered_df)),
                    ],
                ),
            ]
        )

    if active_tab == "tab-analysis":
        return html.Div(
            children=[
                section_title("Analyse détaillée", "Lecture approfondie des structures de coûts, maladies et admissions."),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                        "marginBottom": "20px",
                    },
                    children=[
                        graph_card("analysis-cost-by-dept-graph", build_cost_by_dept_figure(filtered_df)),
                        graph_card("analysis-top-maladies-graph", build_top_maladies_figure(filtered_df)),
                    ],
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr",
                        "gap": "20px",
                    },
                    children=[
                        graph_card("analysis-monthly-graph", build_monthly_figure(filtered_df)),
                    ],
                ),
            ]
        )

    if active_tab == "tab-decision":
        return html.Div(
            children=[
                section_title("Aide à la décision", "Traduction des résultats en actions prioritaires."),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                    },
                    children=[
                        info_panel("Panneau d’aide à la décision", generate_decision_aid(filtered_df)),
                        info_panel("Recommandations automatiques", generate_auto_recommendations(filtered_df)),
                    ],
                ),
            ]
        )

    return html.Div(
        children=[
            section_title("Machine Learning", "Intégration des résultats prédictifs validés dans le projet."),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                    "marginBottom": "20px",
                },
                children=[
                    graph_card("ml-reg-results-graph", build_reg_results_figure()),
                    graph_card("ml-fi-reg-graph", build_fi_reg_figure()),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                },
                children=[
                    graph_card("ml-fi-clf-graph", build_fi_clf_figure()),
                    html.Div(
                        style={
                            "backgroundColor": CARD_COLOR,
                            "borderRadius": "20px",
                            "padding": "22px",
                            "boxShadow": "0 10px 24px rgba(0,0,0,0.06)",
                            "border": "1px solid rgba(129,20,51,0.08)",
                        },
                        children=[
                            html.H3("Interprétation ML", style={"color": PRIMARY_COLOR, "marginTop": "0"}),
                            html.P("Le modèle Gradient Boosting est le plus performant pour la prédiction du coût, avec un R² de 0,83."),
                            html.P("La durée de séjour est la variable la plus déterminante dans la prédiction du coût."),
                            html.P("Les scores parfaits en classification doivent être interprétés avec prudence à cause d’un data leakage."),
                            html.P("Le dashboard permet ainsi de visualiser la performance des modèles, leur utilité et leurs limites."),
                        ],
                    ),
                ],
            ),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)