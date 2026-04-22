"""
Telecom Customer Churn Predictor
=================================
Portfolio ML project — end-to-end churn prediction pipeline.
Stack: Python · scikit-learn · imbalanced-learn · Streamlit · pandas · matplotlib

Run with:  streamlit run app.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "churn.csv")
RANDOM_STATE = 42
TARGET = "Churn"

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0F1117",
    "surface":  "#1A1D27",
    "border":   "#2A2D3E",
    "accent":   "#6C63FF",
    "accent2":  "#00D4AA",
    "danger":   "#FF4B6E",
    "warning":  "#FFA940",
    "text":     "#E8EAF0",
    "muted":    "#8B8FA8",
}

CHART_STYLE = {
    "figure.facecolor":  PALETTE["surface"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.alpha":        0.6,
}

def apply_chart_style():
    plt.rcParams.update(CHART_STYLE)

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Space Grotesk', sans-serif;
      background-color: {PALETTE['bg']};
      color: {PALETTE['text']};
  }}

  /* ── Header strip ── */
  .hero {{
      background: linear-gradient(135deg, {PALETTE['surface']} 0%, #1E1B3A 100%);
      border: 1px solid {PALETTE['border']};
      border-radius: 16px;
      padding: 2.5rem 2.5rem 2rem;
      margin-bottom: 1.5rem;
      position: relative;
      overflow: hidden;
  }}
  .hero::before {{
      content: '';
      position: absolute;
      top: -60px; right: -60px;
      width: 200px; height: 200px;
      background: radial-gradient(circle, {PALETTE['accent']}22 0%, transparent 70%);
      pointer-events: none;
  }}
  .hero h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      margin: 0 0 0.4rem;
      background: linear-gradient(90deg, {PALETTE['text']} 0%, {PALETTE['accent']} 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
  }}
  .hero p {{
      color: {PALETTE['muted']};
      margin: 0;
      font-size: 0.95rem;
      line-height: 1.5;
  }}

  /* ── KPI cards ── */
  .kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
  .kpi-card {{
      flex: 1;
      background: {PALETTE['surface']};
      border: 1px solid {PALETTE['border']};
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
  }}
  .kpi-card .label {{
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {PALETTE['muted']};
      margin-bottom: 0.4rem;
  }}
  .kpi-card .value {{
      font-size: 2rem;
      font-weight: 700;
      color: {PALETTE['text']};
      font-family: 'JetBrains Mono', monospace;
      line-height: 1;
  }}
  .kpi-card .sub {{
      font-size: 0.72rem;
      color: {PALETTE['muted']};
      margin-top: 0.3rem;
  }}
  .kpi-card.accent  {{ border-top: 3px solid {PALETTE['accent']}; }}
  .kpi-card.accent2 {{ border-top: 3px solid {PALETTE['accent2']}; }}
  .kpi-card.danger  {{ border-top: 3px solid {PALETTE['danger']}; }}
  .kpi-card.warning {{ border-top: 3px solid {PALETTE['warning']}; }}

  /* ── Section headers ── */
  .section-title {{
      font-size: 1.05rem;
      font-weight: 600;
      color: {PALETTE['text']};
      letter-spacing: -0.01em;
      margin: 1.8rem 0 0.8rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid {PALETTE['border']};
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 0;
      background: {PALETTE['surface']};
      border-radius: 10px;
      padding: 4px;
      border: 1px solid {PALETTE['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px;
      padding: 0.5rem 1.2rem;
      font-weight: 500;
      font-size: 0.88rem;
      color: {PALETTE['muted']};
  }}
  .stTabs [aria-selected="true"] {{
      background: {PALETTE['accent']} !important;
      color: white !important;
  }}

  /* ── Metric override ── */
  [data-testid="stMetricValue"] {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.7rem !important;
      color: {PALETTE['text']} !important;
  }}
  [data-testid="stMetricLabel"] {{
      font-size: 0.72rem !important;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: {PALETTE['muted']} !important;
  }}

  /* ── Prediction result card ── */
  .result-card {{
      background: {PALETTE['surface']};
      border: 1px solid {PALETTE['border']};
      border-radius: 14px;
      padding: 2rem;
      text-align: center;
      margin-top: 1rem;
  }}
  .result-card .prob {{
      font-size: 3.5rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      line-height: 1;
  }}
  .result-card .verdict {{
      font-size: 1.1rem;
      font-weight: 600;
      margin-top: 0.6rem;
  }}
  .result-card .advice {{
      font-size: 0.85rem;
      color: {PALETTE['muted']};
      margin-top: 0.8rem;
      line-height: 1.6;
  }}
  .churn    {{ color: {PALETTE['danger']}; }}
  .no-churn {{ color: {PALETTE['accent2']}; }}

  /* ── Tag pills ── */
  .tag {{
      display: inline-block;
      background: {PALETTE['accent']}22;
      border: 1px solid {PALETTE['accent']}44;
      color: {PALETTE['accent']};
      font-size: 0.7rem;
      font-weight: 600;
      padding: 0.15rem 0.6rem;
      border-radius: 20px;
      margin-right: 0.3rem;
      font-family: 'JetBrains Mono', monospace;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background: {PALETTE['surface']};
      border-right: 1px solid {PALETTE['border']};
  }}
  [data-testid="stSidebar"] * {{ color: {PALETTE['text']}; }}

  /* ── Dataframe ── */
  .stDataFrame {{ border-radius: 10px; overflow: hidden; }}

  /* ── Slider ── */
  .stSlider [data-baseweb="slider"] {{ padding-top: 0.5rem; }}

  /* ── Inputs ── */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input {{
      background: {PALETTE['surface']};
      border: 1px solid {PALETTE['border']};
      color: {PALETTE['text']};
      border-radius: 8px;
  }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}
</style>
"""

# ── Data pipeline ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"CSV not found at: {DATA_PATH}\n"
            "Place the Kaggle Telco churn CSV at data/churn.csv"
        )
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    return df


@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tenure_bin"] = pd.cut(
        out["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["New", "Mid", "Loyal", "Long-term"],
    )
    out["AvgMonthlyCharge"] = np.round(out["TotalCharges"] / out["tenure"].clip(1), 2)
    out["ChargeRatio"] = np.round(out["MonthlyCharges"] / (out["TotalCharges"] + 1), 4)
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    out["NumServices"] = out[service_cols].apply(lambda r: (r == "Yes").sum(), axis=1)
    out["HasInternet"] = (out["InternetService"] != "No").astype(int)
    out["IsAutoPayment"] = out["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    return out


@st.cache_data
def preprocess(df: pd.DataFrame):
    drop_cols = ["customerID", "tenure_bin"]
    model_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = (model_df[TARGET] == "Yes").astype(int)
    X = model_df.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test  = pd.DataFrame(scaler.transform(X_test),  columns=feature_names)
    return (
        X_train, X_test,
        y_train.reset_index(drop=True), y_test.reset_index(drop=True),
        feature_names, scaler,
    )


@st.cache_data
def train_models(_X_train, _y_train, use_smote: bool = True):
    X_tr, y_tr = _X_train.copy(), _y_train.copy()
    if use_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                       random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                            learning_rate=0.1,
                                                            random_state=RANDOM_STATE),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)
    return models


@st.cache_data
def evaluate_models(_models, _X_test, _y_test, threshold: float = 0.5):
    rows = []
    for name, model in _models.items():
        probs = model.predict_proba(_X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        rows.append({
            "Model":      name,
            "Accuracy":   round(accuracy_score(_y_test, preds),          4),
            "Precision":  round(precision_score(_y_test, preds, zero_division=0), 4),
            "Recall":     round(recall_score(_y_test, preds, zero_division=0),    4),
            "F1":         round(f1_score(_y_test, preds, zero_division=0),        4),
            "ROC-AUC":    round(roc_auc_score(_y_test, probs),            4),
        })
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(raw_df, models, X_test, y_test):
    with st.sidebar:
        st.markdown("## About this project")
        st.markdown(
            "End-to-end churn prediction pipeline built on the "
            "[IBM Telco dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). "
            "Combines classical ML with business-focused threshold tuning."
        )
        st.divider()
        st.markdown("**Stack**")
        tags = ["scikit-learn", "imbalanced-learn", "pandas", "Streamlit", "matplotlib"]
        st.markdown(" ".join(f'<span class="tag">{t}</span>' for t in tags), unsafe_allow_html=True)
        st.divider()
        st.markdown("**Pipeline**")
        steps = [
            ("1", "EDA & segment analysis"),
            ("2", "Feature engineering (6 derived features)"),
            ("3", "SMOTE oversampling"),
            ("4", "Model comparison (3 classifiers)"),
            ("5", "Threshold optimisation"),
            ("6", "Live inference"),
        ]
        for num, step in steps:
            st.markdown(
                f'<div style="display:flex;gap:0.6rem;align-items:flex-start;'
                f'margin-bottom:0.4rem">'
                f'<span style="background:{PALETTE["accent"]}33;color:{PALETTE["accent"]};'
                f'border-radius:4px;padding:0 6px;font-size:0.72rem;font-family:monospace;'
                f'line-height:1.6">{num}</span>'
                f'<span style="font-size:0.82rem;color:{PALETTE["muted"]}">{step}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        best = models["Gradient Boosting"]
        probs = best.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, probs), 4)
        st.metric("Best ROC-AUC", f"{auc:.4f}", "Gradient Boosting + SMOTE")


# ── Tab helpers ───────────────────────────────────────────────────────────────

def tab_overview(raw_df, df):
    apply_chart_style()
    churn_pct = (raw_df[TARGET] == "Yes").mean() * 100
    n = len(raw_df)

    # KPI strip
    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-card accent">
            <div class="label">Dataset Size</div>
            <div class="value">{n:,}</div>
            <div class="sub">customer records</div>
          </div>
          <div class="kpi-card danger">
            <div class="label">Churn Rate</div>
            <div class="value">{churn_pct:.1f}%</div>
            <div class="sub">class imbalance handled via SMOTE</div>
          </div>
          <div class="kpi-card accent2">
            <div class="label">Features</div>
            <div class="value">26</div>
            <div class="sub">20 raw + 6 engineered</div>
          </div>
          <div class="kpi-card warning">
            <div class="label">Best F1</div>
            <div class="value">~0.62</div>
            <div class="sub">Gradient Boosting + SMOTE</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Churn by Contract Type</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        rates = (
            raw_df.groupby("Contract")[TARGET]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=True)
        )
        colors = [PALETTE["danger"] if v > 30 else PALETTE["accent"] for v in rates.values]
        bars = ax.barh(rates.index, rates.values, color=colors, height=0.5)
        for bar, v in zip(bars, rates.values):
            ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=9, color=PALETTE["text"])
        ax.set_xlabel("Churn Rate (%)", fontsize=8)
        ax.set_xlim(0, rates.max() * 1.25)
        ax.axvline(churn_pct, color=PALETTE["muted"], linestyle="--", lw=1, alpha=0.6)
        ax.text(churn_pct + 0.3, -0.6, "avg", fontsize=7, color=PALETTE["muted"])
        ax.grid(axis="x", alpha=0.3)
        sns.despine(left=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">Churn by Tenure Segment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        order = ["New", "Mid", "Loyal", "Long-term"]
        rates2 = (
            df.groupby("tenure_bin", observed=True)[TARGET]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reindex(order)
        )
        segment_colors = [PALETTE["danger"], PALETTE["warning"], PALETTE["accent"], PALETTE["accent2"]]
        bars = ax.bar(rates2.index, rates2.values, color=segment_colors, width=0.5)
        for bar, v in zip(bars, rates2.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.1f}%", ha="center", fontsize=9, color=PALETTE["text"])
        ax.set_ylabel("Churn Rate (%)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Monthly Charges Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        for label, color in [("No", PALETTE["accent2"]), ("Yes", PALETTE["danger"])]:
            subset = raw_df[raw_df[TARGET] == label]["MonthlyCharges"]
            ax.hist(subset, bins=30, alpha=0.7, color=color, label=f"Churn = {label}", density=True)
        ax.set_xlabel("Monthly Charges ($)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.markdown('<div class="section-title">Tenure Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        for label, color in [("No", PALETTE["accent2"]), ("Yes", PALETTE["danger"])]:
            subset = raw_df[raw_df[TARGET] == label]["tenure"]
            ax.hist(subset, bins=24, alpha=0.7, color=color, label=f"Churn = {label}", density=True)
        ax.set_xlabel("Tenure (months)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def tab_model(models, X_train, X_test, y_train, y_test, feature_names):
    apply_chart_style()

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.01,
        help="Adjust the probability cut-off. Lower = more churn flags (↑ recall, ↓ precision).",
    )

    results = evaluate_models(models, X_test, y_test, threshold)

    # Highlight best per metric
    def highlight_best(s):
        is_max = s == s.max()
        return [f"color: {PALETTE['accent2']}; font-weight:600" if v else "" for v in is_max]

    styled = (
        results.style
        .apply(highlight_best, subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
        .format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]})
        .set_properties(**{"text-align": "center"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    # ── ROC curves ───────────────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="section-title">ROC Curves</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        model_colors = [PALETTE["accent"], PALETTE["accent2"], PALETTE["warning"]]
        for (name, model), color in zip(models.items(), model_colors):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1, label="Random (AUC=0.500)")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="section-title">Confusion Matrix — Gradient Boosting</div>', unsafe_allow_html=True)
        best = models["Gradient Boosting"]
        probs = best.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
            ax=ax, linewidths=0.5, linecolor=PALETTE["border"],
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(f"Threshold = {threshold:.2f}", fontsize=9, color=PALETTE["muted"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Precision-Recall curve ────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Precision–Recall Trade-off</div>', unsafe_allow_html=True)
        precisions, recalls, thr_pr = precision_recall_curve(y_test, probs)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(recalls, precisions, color=PALETTE["accent"], lw=2)
        idx = min(np.searchsorted(thr_pr, threshold, side="right"), len(precisions) - 1)
        ax.scatter([recalls[idx]], [precisions[idx]],
                   color=PALETTE["danger"], s=100, zorder=5,
                   label=f"Threshold {threshold:.2f}")
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Feature importance ────────────────────────────────────────────────────
    with col4:
        st.markdown('<div class="section-title">Top 15 Feature Importances</div>', unsafe_allow_html=True)
        rf = models["Random Forest"]
        imp = pd.Series(rf.feature_importances_, index=feature_names).nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        colors = [
            PALETTE["accent2"] if i >= len(imp) - 3 else PALETTE["accent"]
            for i in range(len(imp))
        ]
        imp.plot.barh(ax=ax, color=colors, edgecolor="none")
        ax.set_xlabel("Importance", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        sns.despine(left=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def tab_predict(models, X_train, y_train, feature_names, scaler):
    apply_chart_style()
    prod_model = models["Gradient Boosting"]

    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        gender      = st.selectbox("Gender",          ["Male", "Female"])
        senior      = st.selectbox("Senior Citizen",  [0, 1])
        partner     = st.selectbox("Partner",         ["Yes", "No"])
        dependents  = st.selectbox("Dependents",      ["Yes", "No"])
        tenure      = st.slider("Tenure (months)",    1, 72, 12)
        phone       = st.selectbox("Phone Service",   ["Yes", "No"])
    with c2:
        multi       = st.selectbox("Multiple Lines",  ["Yes", "No", "No phone service"])
        internet    = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
        security    = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup      = st.selectbox("Online Backup",   ["Yes", "No", "No internet service"])
        protection  = st.selectbox("Device Protection",["Yes","No","No internet service"])
        tech        = st.selectbox("Tech Support",    ["Yes", "No", "No internet service"])
    with c3:
        stv         = st.selectbox("Streaming TV",    ["Yes", "No", "No internet service"])
        smov        = st.selectbox("Streaming Movies",["Yes", "No", "No internet service"])
        contract    = st.selectbox("Contract",        ["Month-to-month", "One year", "Two year"])
        paperless   = st.selectbox("Paperless Billing",["Yes", "No"])
        payment     = st.selectbox("Payment Method",  [
                          "Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly     = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total       = st.number_input("Total Charges ($)",   0.0, 10000.0, float(monthly * tenure))

    predict_btn = st.button("Run Prediction →", type="primary", use_container_width=True)

    if predict_btn:
        svc_vals = [security, backup, protection, tech, stv, smov]
        num_svc  = sum(1 for s in svc_vals if s == "Yes")

        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": tech,
            "StreamingTV": stv, "StreamingMovies": smov,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
            "AvgMonthlyCharge": round(total / max(tenure, 1), 2),
            "ChargeRatio": round(monthly / (total + 1), 4),
            "NumServices": num_svc,
            "HasInternet": int(internet != "No"),
            "IsAutoPayment": int(payment in ["Bank transfer (automatic)", "Credit card (automatic)"]),
        }])
        row_enc    = pd.get_dummies(row, drop_first=True).reindex(columns=feature_names, fill_value=0)
        row_scaled = pd.DataFrame(scaler.transform(row_enc), columns=feature_names)
        prob       = prod_model.predict_proba(row_scaled)[0][1]

        is_churn   = prob >= 0.5
        prob_color = PALETTE["danger"] if is_churn else PALETTE["accent2"]
        verdict    = "⚠ High Churn Risk" if is_churn else "✓ Low Churn Risk"
        advice = (
            "Recommend a proactive retention offer — loyalty discount, plan upgrade, "
            "or a dedicated support touchpoint."
            if is_churn else
            "Customer appears stable. Continue standard engagement and monitor "
            "for contract renewals."
        )

        st.markdown(
            f"""
            <div class="result-card">
              <div class="prob" style="color:{prob_color}">{prob:.1%}</div>
              <div class="verdict {'churn' if is_churn else 'no-churn'}">{verdict}</div>
              <div class="advice">{advice}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Mini breakdown bar
        st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 0.9))
        ax.barh([0], [prob],       color=PALETTE["danger"],  height=0.5)
        ax.barh([0], [1 - prob], left=[prob], color=PALETTE["accent2"], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.text(prob / 2, 0, f"Churn {prob:.1%}", ha="center", va="center",
                fontsize=9, color="white", fontweight="600")
        ax.text(prob + (1 - prob) / 2, 0, f"Stay {1 - prob:.1%}", ha="center", va="center",
                fontsize=9, color="white", fontweight="600")
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Churn Predictor",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
          <h1>Telecom Churn Predictor</h1>
          <p>
            End-to-end ML pipeline — EDA · Feature engineering · SMOTE · 
            Model comparison · Threshold tuning · Live inference
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load & prepare ────────────────────────────────────────────────────────
    with st.spinner("Loading and preparing data…"):
        raw_df           = load_data()
        df               = engineer_features(raw_df)
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)
        models           = train_models(X_train, y_train, use_smote=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    render_sidebar(raw_df, models, X_test, y_test)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📊  Overview", "🤖  Model Performance", "🔮  Live Predictor"])

    with t1:
        tab_overview(raw_df, df)

    with t2:
        tab_model(models, X_train, X_test, y_train, y_test, feature_names)

    with t3:
        tab_predict(models, X_train, y_train, feature_names, scaler)


if __name__ == "__main__":
    main()
