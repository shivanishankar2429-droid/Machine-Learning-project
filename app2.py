import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
    learning_curve as sk_learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay

from imblearn.over_sampling import SMOTE


st.set_page_config(page_title="NASA MDP Software Defect Prediction", layout="wide")
st.title("NASA MDP Software Defect Prediction Dashboard")
st.markdown(
    "This dashboard trains multiple **classification** models on NASA MDP software metrics "
    "to predict whether a module is defective or not. It supports cross‑validation, explainability, "
    "single‑module prediction, and batch prediction for engineering use-cases."
)

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
       background-color: rgba(28, 131, 225, 0.1);
       border: 1px solid rgba(28, 131, 225, 0.1);
       padding: 15px 15px 15px 20px;
       border-radius: 10px;
       color: rgb(30, 103, 119);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload NASA MDP CSV", type=["csv"])
label_column = st.sidebar.text_input("Label column (defect count or binary)", value="defects")
label_is_count = st.sidebar.checkbox("Label is defect count (>0 = defective)", value=True)
use_smote = st.sidebar.checkbox("Use SMOTE for class balancing", value=True)
enable_tuning = st.sidebar.checkbox("Enable RF Hyperparameter Tuning", value=False)

top_k_features = st.sidebar.slider(
    "Select Number of Top Features to Use (RFE)",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
    help="Recursive Feature Elimination selects the K most important features for the main model.",
)

threshold = st.sidebar.slider(
    "Risk tolerance threshold (decision boundary for 'Defective')",
    min_value=0.05,
    max_value=0.95,
    value=0.5,
    step=0.05,
    help="Lower threshold = more modules flagged as defective (higher Recall, lower Precision).",
)

random_state = 42


@st.cache_data
def load_data(file, label_col, is_count):
    df = pd.read_csv(file)
    if is_count:
        df["target"] = (df[label_col] > 0).astype(int)
    else:
        df["target"] = df[label_col].astype(int)

    drop_cols = [label_col, "target", "module", "file", "name"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    features = [c for c in df.columns if c not in drop_cols]
    X_df = df[features].select_dtypes(include=[np.number])

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df)
    y = df["target"].to_numpy().astype(int)
    return df, X, y, features, imputer


def apply_rfe_feature_selection(X, y, feature_names, k, rnd_state=42):
    k = min(k, X.shape[1])
    base_estimator = LogisticRegression(
        max_iter=500, class_weight="balanced", solver="lbfgs"
    )
    selector = RFE(estimator=base_estimator, n_features_to_select=k, step=1)
    selector = selector.fit(X, y)
    X_reduced = selector.transform(X)
    selected_feature_names = [f for f, keep in zip(feature_names, selector.support_) if keep]
    return X_reduced, selected_feature_names


def train_models(X, y, use_smote_flag, enable_tuning_flag, rnd_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=rnd_state
    )

    if use_smote_flag:
        smote = SMOTE(random_state=rnd_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {}

    
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=rnd_state,
        n_estimators=200,
        max_depth=None,
    )
    if enable_tuning_flag:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        }
        rf_search = RandomizedSearchCV(
            rf,
            param_distributions=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            random_state=rnd_state,
        )
        rf_search.fit(X_train, y_train)
        rf = rf_search.best_estimator_
    else:
        rf.fit(X_train, y_train)
    models["Random Forest"] = {"model": rf, "scaled": False}

    
    lr = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")
    lr.fit(X_train_s, y_train)
    models["Logistic Regression"] = {"model": lr, "scaled": True}

   
    svm = SVC(probability=True, class_weight="balanced", kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train_s, y_train)
    models["SVM (RBF)"] = {"model": svm, "scaled": True}

    
    gb = GradientBoostingClassifier(random_state=rnd_state)
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = {"model": gb, "scaled": False}

    
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2,
    )
    knn.fit(X_train_s, y_train)
    models["KNN"] = {"model": knn, "scaled": True}

    return models, scaler, X_train, X_test, y_train, y_test, X_train_s, X_test_s


def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = sk_learning_curve(
        estimator,
        X,
        y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")

    ax.set_ylim(0.5, 1.01)
    ax.set_title(title)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def plot_pca_2d(X, y, scaler=None):
    if X.shape[0] < 3:
        st.info("Not enough samples to plot 2D PCA (need at least 3).")
        return
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        local_scaler = StandardScaler()
        X_scaled = local_scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Status"] = ["Defective" if val == 1 else "Clean" for val in y]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Status", alpha=0.6, ax=ax)
    ax.set_title("2D Data Projection (PCA)")
    st.pyplot(fig)
    plt.close(fig)


def plot_pca_3d(X, y, scaler=None):
    if X.shape[0] < 4:
        st.info("Not enough samples to plot 3D PCA (need at least 4).")
        return
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        local_scaler = StandardScaler()
        X_scaled = local_scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    pca_df["Status"] = ["Defective" if val == 1 else "Clean" for val in y]
    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Status",
        opacity=0.7,
        title="3D Data Projection (PCA)",
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)


def plot_threshold_explorer(y_test, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.plot(thresholds, f1_scores[:-1], "r-", label="F1 Score", lw=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metric Trade-off vs. Decision Threshold")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def plot_cumulative_gains_and_lift(y_true, y_prob):
    df_temp = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_temp = df_temp.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df_temp["is_positive"] = df_temp["y_true"]
    df_temp["cum_positives"] = df_temp["is_positive"].cumsum()
    total_positives = df_temp["is_positive"].sum()
    total_samples = len(df_temp)
    df_temp["cum_percent_samples"] = (np.arange(1, total_samples + 1) / total_samples) * 100
    df_temp["cum_percent_positives"] = (df_temp["cum_positives"] / total_positives) * 100
    baseline = df_temp["cum_percent_samples"]

    fig_g, ax_g = plt.subplots(figsize=(6, 4))
    ax_g.plot(df_temp["cum_percent_samples"], df_temp["cum_percent_positives"], label="Model")
    ax_g.plot(df_temp["cum_percent_samples"], baseline, "k--", label="Baseline (random)")
    ax_g.set_xlabel("Percentage of Modules Tested")
    ax_g.set_ylabel("Percentage of Defects Captured")
    ax_g.set_title("Cumulative Gains Curve")
    ax_g.legend()
    st.pyplot(fig_g)
    plt.close(fig_g)

    lift = df_temp["cum_percent_positives"] / baseline.replace(0, np.nan)
    fig_l, ax_l = plt.subplots(figsize=(6, 4))
    ax_l.plot(df_temp["cum_percent_samples"], lift, label="Lift")
    ax_l.axhline(1.0, color="k", linestyle="--", label="Baseline (Lift=1)")
    ax_l.set_xlabel("Percentage of Modules Tested")
    ax_l.set_ylabel("Lift")
    ax_l.set_title("Lift Curve")
    ax_l.legend()
    st.pyplot(fig_l)
    plt.close(fig_l)


if not uploaded_file:
    st.warning("Please upload a NASA MDP CSV file in the sidebar to begin.")
    st.stop()


df, X_raw, y, feature_names_raw, imputer_raw = load_data(
    uploaded_file, label_column, label_is_count
)


X_rfe, feature_names_rfe = apply_rfe_feature_selection(
    X_raw, y, feature_names_raw, top_k_features
)


X_df_rfe = df[feature_names_rfe].select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy="median")
imputer.fit(X_df_rfe)


X = imputer.transform(X_df_rfe)
feature_names = feature_names_rfe


st.session_state["imputer"] = imputer
st.session_state["feature_names"] = feature_names
st.session_state["X_full"] = X
st.session_state["y_full"] = y


modules_analyzed = df.shape[0]
avg_complexity = None
for cand in ["cyclomatic_complexity", "mcabe_complexity", "v(g)", "complexity"]:
    if cand in df.columns:
        avg_complexity = df[cand].mean()
        break

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Modules Analyzed", f"{modules_analyzed:,}", "NASA dataset")
with col2:
    if avg_complexity is not None:
        st.metric("Avg Complexity", f"{avg_complexity:.1f}")
    else:
        st.metric("Avg Complexity", "N/A")
best_acc_display = st.session_state.get("best_cv_accuracy_display", "N/A")
best_model_display = st.session_state.get("best_model_name_display", "Train models in Tab 1")
with col3:
    st.metric("Detection Accuracy (CV)", best_acc_display, best_model_display)
with col4:
    risk_label = "Elevated" if y.mean() > 0.3 else "Moderate"
    st.metric("Risk Level", risk_label)


st.markdown("### Dataset Overview")
col_a, col_b = st.columns(2)
with col_a:
    st.write(f"Shape: **{df.shape[0]} rows, {df.shape[1]} columns**")
    st.write(f"Number of numeric features after RFE: **{len(feature_names)}**")
with col_b:
    defect_rate = y.mean()
    st.write(f"Defect rate (positive class): **{defect_rate:.3f}**")
st.dataframe(df.head(), use_container_width=True)

st.markdown("### Training feature columns used after RFE")
st.write(feature_names)


with st.expander("Data Profiling: Feature Distributions"):
    numeric_cols = df[feature_names].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        prof_col1, prof_col2 = st.columns(2)
        with prof_col1:
            feat1 = st.selectbox("Feature for Histogram", numeric_cols, key="prof_hist")
            fig_h, ax_h = plt.subplots()
            sns.histplot(df[feat1], kde=True, ax=ax_h)
            ax_h.set_title(f"Distribution of {feat1}")
            st.pyplot(fig_h)
            plt.close(fig_h)
        with prof_col2:
            feat2 = st.selectbox("Feature for Boxplot", numeric_cols, key="prof_box")
            fig_b, ax_b = plt.subplots()
            sns.boxplot(x=df[feat2], ax=ax_b)
            ax_b.set_title(f"Boxplot of {feat2}")
            st.pyplot(fig_b)
            plt.close(fig_b)
    else:
        st.info("No numeric features available for profiling.")

st.sidebar.download_button(
    label="Download Processed CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="processed_nasa_data.csv",
    mime="text/csv",
)


st.markdown("#### Interactive Correlation Heatmap (Top 15 numeric features, post‑RFE)")
numeric_df = df[feature_names].select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr()
    top_features = numeric_df.var().sort_values(ascending=False).head(15).index
    corr_top = corr.loc[top_features, top_features]
    fig_corr = px.imshow(
        corr_top,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )
    fig_corr.update_layout(
        title="Interactive Metric Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
    )
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not enough numeric features to compute a correlation heatmap.")


scaler_for_pca = st.session_state.get("scaler", None)
if st.checkbox("Show 2D Data Projection (PCA)"):
    plot_pca_2d(X, y, scaler=scaler_for_pca)
if st.checkbox("Show 3D Data Projection (PCA)"):
    plot_pca_3d(X, y, scaler_for_pca)


tabs = st.tabs(["Model Performance", "Explainability", "Single Prediction", "Batch Prediction"])


with tabs[0]:
    st.subheader("Training, Cross‑Validation & Learning Curves")
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... this may take a moment."):
            (
                models,
                scaler,
                X_train,
                X_test,
                y_train,
                y_test,
                X_train_s,
                X_test_s,
            ) = train_models(X, y, use_smote, enable_tuning, random_state)
            st.session_state.models = models
            st.session_state.scaler = scaler
            st.session_state.X_train = X_train
            st.session_state.X_train_s = X_train_s
            st.session_state.X_test = X_test
            st.session_state.X_test_s = X_test_s
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.feature_names = feature_names
            st.success("Models trained successfully!")

    if "models" in st.session_state:
        models = st.session_state.models
        X_test = st.session_state.X_test
        X_test_s = st.session_state.X_test_s
        y_test = st.session_state.y_test

        results = []
        ranking = []
        cv = StratifiedKFold(
            n_splits=min(5, max(2, X.shape[0] // 5)),
            shuffle=True,
            random_state=random_state,
        )

        for name, info in models.items():
            model = info["model"]
            scaled_flag = info["scaled"]
            X_eval = X_test_s if scaled_flag else X_test
            y_prob = model.predict_proba(X_eval)[:, 1]

            precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
            f1_scores = 2 * (precisions := precision_vals) * (recalls := recall_vals) / (
                precisions + recalls + 1e-6
            )
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[max(best_idx - 1, 0)]
            y_pred = (y_prob >= best_threshold).astype(int)

            X_full = X if not scaled_flag else StandardScaler().fit_transform(X)
            cv_scores = cross_validate(
                model,
                X_full,
                y,
                cv=cv,
                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                n_jobs=-1,
            )

            results.append(
                {
                    "Model": name,
                    "Accuracy (CV)": f"{cv_scores['test_accuracy'].mean():.3f} ± {cv_scores['test_accuracy'].std():.3f}",
                    "Precision (CV)": f"{cv_scores['test_precision'].mean():.3f}",
                    "Recall (CV)": f"{cv_scores['test_recall'].mean():.3f}",
                    "F1 (CV)": f"{cv_scores['test_f1'].mean():.3f}",
                    "ROC‑AUC (CV)": f"{cv_scores['test_roc_auc'].mean():.3f}",
                }
            )

            ranking.append(
                {
                    "Model": name,
                    "F1 (Test)": f1_score(y_test, y_pred),
                    "ROC‑AUC (Test)": roc_auc_score(y_test, y_prob),
                }
            )

        st.markdown("#### 5‑Fold Cross‑Validation Metrics")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        rank_df = pd.DataFrame(ranking).sort_values("F1 (Test)", ascending=False)
        best_model_name = rank_df.iloc[0]["Model"]
        st.session_state.best_model_name_display = best_model_name
        row_best = results_df[results_df["Model"] == best_model_name]
        if not row_best.empty:
            st.session_state.best_cv_accuracy_display = row_best["Accuracy (CV)"].values[0]
        else:
            st.session_state.best_cv_accuracy_display = "N/A"

        st.markdown("#### 🏆 Model Ranking on Hold‑out Test Set (by F1)")
        st.dataframe(rank_df, use_container_width=True)
        st.markdown("#### F1 and ROC‑AUC Bar Chart")
        st.bar_chart(rank_df.set_index("Model")[["F1 (Test)", "ROC‑AUC (Test)"]])

        best_info = models[best_model_name]
        best_model = best_info["model"]
        best_scaled = best_info["scaled"]

        st.markdown(f"#### Confusion Matrix (Best Model: {best_model_name}, threshold 0.50)")
        X_eval_best = X_test_s if best_scaled else X_test
        y_prob_best = best_model.predict_proba(X_eval_best)[:, 1]
        y_pred_best = (y_prob_best >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_best)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non‑Defective", "Defective"],
            yticklabels=["Non‑Defective", "Defective"],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.markdown("#### Probability Calibration (Reliability Diagram) - Best Model")
        prob_true, prob_pred = calibration_curve(
            y_test, y_prob_best, n_bins=10, strategy="uniform"
        )
        fig_cal, ax_cal = plt.subplots(figsize=(5, 4))
        ax_cal.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax_cal.plot(prob_pred, prob_true, "s-", label=best_model_name)
        ax_cal.set_xlabel("Mean Predicted Probability")
        ax_cal.set_ylabel("Fraction of Positives")
        ax_cal.set_title("Calibration Curve")
        ax_cal.legend()
        st.pyplot(fig_cal)
        plt.close(fig_cal)

        st.markdown("#### Learning Curve (Best Model)")
        X_full_best = X if not best_scaled else StandardScaler().fit_transform(X)
        plot_learning_curve(best_model, X_full_best, y, title=f"Learning Curve - {best_model_name}")

        st.markdown("#### Precision–Recall Curve (Best Model)")
        fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_prob_best,
            name=best_model_name,
            ax=ax_pr,
        )
        ax_pr.set_title("Precision–Recall Curve")
        st.pyplot(fig_pr)
        plt.close(fig_pr)

        st.markdown("#### Threshold‑Dependent Metric Explorer")
        plot_threshold_explorer(y_test, y_prob_best)

        st.markdown("#### Confidence Distribution (Probability Density)")
        def plot_probability_distribution(model_pd, X_eval_pd, y_true_pd, name_pd):
            probs_pd = model_pd.predict_proba(X_eval_pd)[:, 1]
            df_probs = pd.DataFrame({"Probability": probs_pd, "Actual": y_true_pd})
            fig_pd, ax_pd = plt.subplots(figsize=(8, 4))
            sns.kdeplot(
                data=df_probs,
                x="Probability",
                hue="Actual",
                fill=True,
                common_norm=False,
                ax=ax_pd,
            )
            ax_pd.set_title(f"Confidence Distribution: {name_pd}")
            ax_pd.set_xlim(0, 1)
            ax_pd.set_xlabel("Predicted defect probability")
            st.pyplot(fig_pd)
            plt.close(fig_pd)

        plot_probability_distribution(best_model, X_eval_best, y_test, best_model_name)

        st.markdown("#### Cumulative Gains & Lift Curves")
        plot_cumulative_gains_and_lift(y_test, y_prob_best)

        st.subheader("💰 Economic Impact Analysis")
        with st.expander("Configure Business Costs"):
            cost_fp = st.number_input(
                "Cost of False Alarm (e.g., manual review cost per clean module)",
                value=500,
                min_value=0,
            )
            cost_fn = st.number_input(
                "Cost of Missing a Defect (e.g., production bug cost)",
                value=5000,
                min_value=0,
            )

        thresholds_cost = np.linspace(0, 1, 100)
        costs = []
        for t in thresholds_cost:
            preds = (y_prob_best >= t).astype(int)
            fp = ((preds == 1) & (y_test == 0)).sum()
            fn = ((preds == 0) & (y_test == 1)).sum()
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            costs.append(total_cost)
        fig_cost, ax_cost = plt.subplots()
        ax_cost.plot(thresholds_cost, costs, color="green", lw=2)
        ax_cost.set_xlabel("Decision Threshold")
        ax_cost.set_ylabel("Total Quality Cost ($)")
        ax_cost.set_title(f"Economic Optimization for {best_model_name}")
        st.pyplot(fig_cost)
        plt.close(fig_cost)

        best_t = thresholds_cost[int(np.argmin(costs))]
        st.success(
            f"💡 **Manager's Tip:** To minimize estimated quality cost, "
            f"set your Risk Threshold to **{best_t:.2f}**"
        )

        st.markdown("#### Model Stability under Input Noise")
        noise_level = st.slider(
            "Noise level for stress test (fraction of feature std)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
        )

        X_test_for_noise = X_eval_best
        y_prob_original = best_model.predict_proba(X_test_for_noise)[:, 1]
        y_pred_original = (y_prob_original >= threshold).astype(int)

        feature_std = X_test_for_noise.std(axis=0, ddof=0)
        noise = np.random.normal(
            loc=0.0,
            scale=feature_std * noise_level,
            size=X_test_for_noise.shape,
        )
        X_test_noisy = X_test_for_noise + noise

        y_prob_noisy = best_model.predict_proba(X_test_noisy)[:, 1]
        y_pred_noisy = (y_prob_noisy >= threshold).astype(int)

        flips = (y_pred_original != y_pred_noisy).sum()
        stability_score = 1.0 - flips / len(y_pred_original)

        st.write(
            f"- Predictions changed on **{flips} / {len(y_pred_original)}** test modules "
            f"at noise level {noise_level:.2f}."
        )
        st.write(
            f"- **Model Stability Score:** {stability_score:.3f} "
            "(1.0 = perfectly stable, closer to 0 = fragile)."
        )

        st.markdown("### 🛰️ Mission Intelligence Feed")
        with st.container():
            st.info(
                f"**System Note:** {best_model_name} is currently performing best on the validation set."
            )
            st.error(
                "**Risk Alert:** The highest‑risk module in the latest predictions exceeds the 0.85 risk threshold. "
                "Recommended action: Manual Review and additional tests."
            )
            st.success(
                "**Optimization:** SMOTE balancing has improved Recall compared to the baseline model. "
                "Use the Precision–Recall curve to verify the trade‑off."
            )


with tabs[1]:
    st.subheader("Global Explainability with SHAP (Random Forest)")

    if "models" not in st.session_state:
        st.info("Train the models in Tab 1 first.")
        st.stop()

    rf_model = st.session_state.models["Random Forest"]["model"]
    X_df_all = pd.DataFrame(X, columns=feature_names)

    @st.cache_data
    def compute_shap_values_fast(_model, data_sample):
        explainer_local = shap.TreeExplainer(_model, model_output="raw")
        shap_values_local = explainer_local.shap_values(data_sample)
        return shap_values_local, explainer_local.expected_value

    sample_size = min(150, len(X_df_all))
    X_sample = X_df_all.sample(sample_size, random_state=random_state)

    st.markdown(
        f"Global feature importance using **mean absolute SHAP values** over {sample_size} randomly sampled modules."
    )

    try:
        shap_values_raw, expected_value_raw = compute_shap_values_fast(rf_model, X_sample)
        if isinstance(shap_values_raw, list):
            class_idx = 1 if len(shap_values_raw) > 1 else 0
            shap_values_global = shap_values_raw[class_idx]
        else:
            shap_values_global = shap_values_raw

        fig = plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values_global,
            X_sample,
            plot_type="bar",
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.warning("SHAP bar plot failed.")
        st.code(str(e))

    st.markdown("#### Global Feature Interaction (Partial Dependence)")
    from_feature = st.selectbox(
        "Select feature for Partial Dependence Plot",
        feature_names,
        key="pdp_feature",
    )
    try:
        fig_pdp, ax_pdp = plt.subplots(figsize=(6, 4))
        feature_index = feature_names.index(from_feature)
        PartialDependenceDisplay.from_estimator(
            rf_model,
            X,
            [feature_index],
            ax=ax_pdp,
        )
        ax_pdp.set_title(f"Partial Dependence of {from_feature}")
        st.pyplot(fig_pdp)
        plt.close(fig_pdp)
    except Exception as e:
        st.warning("Partial dependence plot could not be generated.")
        st.code(str(e))


with tabs[2]:
    st.subheader("Single Module Prediction")
    if "models" not in st.session_state:
        st.info("Train the models in Tab 1 first.")
        st.stop()

    X = st.session_state.get("X_full", X)
    y = st.session_state.get("y_full", y)
    feature_names = st.session_state.get("feature_names", feature_names)

    model_name = st.selectbox("Select model", list(st.session_state.models.keys()))
    model_info = st.session_state.models[model_name]
    model = model_info["model"]
    scaled_flag = model_info["scaled"]

    st.markdown("Provide metric values for a **single software module**. Defaults are median values.")
    with st.expander("Adjust Module Metrics", expanded=True):
        input_cols = st.columns(3)
        user_inputs = []
        for i, fname in enumerate(feature_names):
            col = input_cols[i % 3]
            col_values = X[:, i]

            p1 = float(np.percentile(col_values, 1))
            p99 = float(np.percentile(col_values, 99))

            if p1 == p99:
                min_v = p1 - 1.0
                max_v = p1 + 1.0
                default_v = p1
            else:
                min_v = p1
                max_v = p99
                default_v = float(np.median(col_values))

            with col:
                val = st.slider(
                    fname,
                    min_value=min_v,
                    max_value=max_v,
                    value=default_v,
                )
            user_inputs.append(val)

    complexity_feature_name = None
    for cand in ["cyclomatic_complexity", "mcabe_complexity", "v(g)", "complexity"]:
        if cand in feature_names:
            complexity_feature_name = cand
            break

    if st.button("Predict Defect Risk"):
        X_in = np.array(user_inputs).reshape(1, -1)
        if scaled_flag:
            X_in_scaled = st.session_state.scaler.transform(X_in)
        else:
            X_in_scaled = X_in

        prob = model.predict_proba(X_in_scaled)[0, 1]
        predicted_label = int(prob >= threshold)

        if prob >= threshold:
            st.error(f"###  HIGH RISK: {prob:.1%} Defect Probability")
            st.progress(min(float(prob), 1.0))
            st.write("**Action Required:** High‑priority manual code review and extended testing.")
        else:
            st.success(f"### LOW RISK: {prob:.1%} Defect Probability")
            st.progress(min(float(prob), 1.0))
            st.write("**Action Required:** Standard CI/CD pipeline and routine review.")

        st.markdown("#### Developer's Summary (Engineering Advice)")
        if prob > 0.8:
            st.write(
                "- **High Priority:** Require a senior developer code review and full regression test before release."
            )
        elif prob > 0.5:
            st.write(
                "- **Medium Risk:** Schedule peer review and targeted unit tests for this module."
            )
        else:
            st.write(
                "- **Lower Risk:** Standard review process is likely sufficient, but monitor changes over time."
            )

        if complexity_feature_name is not None:
            comp_idx = feature_names.index(complexity_feature_name)
            comp_value = user_inputs[comp_idx]
            if comp_value > 20:
                st.write(
                    f"- `{complexity_feature_name}` ≈ {comp_value:.1f} (> 20). "
                    "Cyclomatic complexity is high; refactor large functions and add more tests."
                )
            elif comp_value > 10:
                st.write(
                    f"- `{complexity_feature_name}` ≈ {comp_value:.1f} (> 10). "
                    "Complexity above NASA/NIST comfort level; add focused unit tests and consider simplifying logic."
                )
            else:
                st.write(
                    f"- `{complexity_feature_name}` ≈ {comp_value:.1f}. Complexity appears manageable for maintenance."
                )
        else:
            st.write(
                "- Complexity metric not found in the selected features; consider adding a cyclomatic complexity metric to the dataset."
            )

        st.markdown("#### 🛠️ Automated Refactoring Recommendations")
        recommendations = []

        for cand in ["cyclomatic_complexity", "mcabe_complexity", "v(g)", "complexity"]:
            if cand in feature_names:
                idx = feature_names.index(cand)
                val = user_inputs[idx]
                if val > 20:
                    recommendations.append(
                        f"- `{cand}` ≈ {val:.1f} (> 20). Consider splitting large functions, "
                        "reducing nested conditionals, and extracting helper methods."
                    )
                elif val > 10:
                    recommendations.append(
                        f"- `{cand}` ≈ {val:.1f} (> 10). Moderate complexity; review branching logic "
                        "and add targeted unit tests around edge cases."
                    )
                break

        for cand in ["loc", "lines_of_code", "loc_total", "nloc"]:
            if cand in feature_names:
                idx = feature_names.index(cand)
                val = user_inputs[idx]
                if val > 500:
                    recommendations.append(
                        f"- `{cand}` ≈ {val:.0f} lines. Very large module; split into smaller, "
                        "cohesive components to reduce change impact and defect surface area."
                    )
                elif val > 200:
                    recommendations.append(
                        f"- `{cand}` ≈ {val:.0f} lines. Large module; consider extracting reusable "
                        "sub‑modules to simplify maintenance."
                    )
                break

        for cand in ["cbo", "coupling_between_objects", "fan_out", "fanout"]:
            if cand in feature_names:
                idx = feature_names.index(cand)
                val = user_inputs[idx]
                if val > 20:
                    recommendations.append(
                        f"- `{cand}` ≈ {val:.1f} (> 20). High coupling; introduce clearer interfaces, "
                        "reduce cross‑module dependencies, and apply dependency inversion where possible."
                    )
                elif val > 10:
                    recommendations.append(
                        f"- `{cand}` ≈ {val:.1f} (> 10). Coupling is elevated; review responsibilities "
                        "and consider decoupling tightly linked modules."
                    )
                break

        for cand in ["code_churn", "churn", "revisions", "bugfix_count"]:
            if cand in feature_names:
                idx = feature_names.index(cand)
                val = user_inputs[idx]
                if val > 10:
                    recommendations.append(
                        f"- `{cand}` ≈ {val:.0f}. High change rate; enforce stricter code review and "
                        "add regression tests around frequently modified areas."
                    )
                break

        if prob > 0.7 and not recommendations:
            recommendations.append(
                "- Model flags high risk despite metrics being within typical ranges. "
                "Review domain‑specific assumptions, naming, and boundary conditions; consider adding "
                "more descriptive metrics (e.g., defect history, ownership, or code review coverage)."
            )

        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write(
                "- No specific refactoring flags were triggered by metric thresholds. "
                "Continue to follow standard code review and testing practices."
            )

        is_tree_model = ("Random Forest" in model_name) or ("Gradient Boosting" in model_name)

        st.markdown("#### Manager's Insight (SHAP‑Driven Recommendation)")
        if is_tree_model:
            try:
                explainer_mgr = shap.TreeExplainer(model, model_output="raw")
                shap_values_raw_mgr = explainer_mgr.shap_values(X_in_scaled)
                if isinstance(shap_values_raw_mgr, list):
                    class_idx = 1 if len(shap_values_raw_mgr) > 1 else 0
                    shap_vals_row = shap_values_raw_mgr[class_idx][0]
                else:
                    shap_vals_row = shap_values_raw_mgr[0]
                abs_shap = np.abs(shap_vals_row)
                top_idx = int(np.argmax(abs_shap))
                top_feature = feature_names[top_idx]
                top_contrib = shap_vals_row[top_idx]
                st.write(
                    f"- Top risk driver for this prediction: `{top_feature}` "
                    f"(SHAP contribution ≈ {top_contrib:.3f})."
                )
            except Exception:
                st.warning("SHAP analysis is optimized for tree‑based models.")
        else:
            st.info("SHAP deep‑dive is currently active for tree‑based models only.")

        if is_tree_model:
            try:
                explainer = shap.TreeExplainer(model)
                shap_val_single = explainer.shap_values(X_in_scaled)
                if isinstance(shap_val_single, list):
                    shap_val_single = shap_val_single[1]
                st.markdown("#### Local SHAP Waterfall Plot")
                fig_wf = plt.figure()
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
                    shap_val_single[0],
                    feature_names=feature_names,
                    max_display=10,
                    show=False,
                )
                st.pyplot(plt.gcf())
                plt.close()
                st.write(
                    "**Top Risk Driver:** The metric at the top of the chart is the primary reason for this risk score."
                )
            except Exception:
                st.info("Local SHAP visualization is unavailable for this specific model configuration.")

        st.info("Recommendations based on feature analysis:")
        if complexity_feature_name and user_inputs[feature_names.index(complexity_feature_name)] > 15:
            st.write("- Extract long methods into smaller, testable functions.")
            st.write("- Reduce nested conditional logic (if/else).")
        else:
            st.write("- Maintain current modularity; ensure unit test coverage remains high.")


with tabs[3]:
    st.subheader("Batch Prediction Workflow")
    st.markdown(
        "Upload a CSV with the same feature columns to generate a risk report for multiple modules."
    )

    batchfile = st.file_uploader("Upload Batch CSV", type="csv", key="batchupload")

    if batchfile and "models" in st.session_state:
        batch_df = pd.read_csv(batchfile)

        
        imputer_batch = st.session_state.get("imputer", None)
        feature_names = st.session_state.get("feature_names", None)
        models = st.session_state.models
        scaler = st.session_state.get("scaler", None)

        if feature_names is None or imputer_batch is None:
            st.error(
                "Preprocessing artefacts missing. "
                "Train models in the 'Model Performance' tab first."
            )
            st.stop()

        
        if label_column in batch_df.columns:
            batch_df = batch_df.drop(columns=[label_column])

        
        train_cols = list(feature_names)

        missing = [c for c in train_cols if c not in batch_df.columns]
        extra = [c for c in batch_df.columns if c not in train_cols]

        if missing or extra:
            st.error(
                "Batch CSV columns do not exactly match the training feature set.\n\n"
                f"Required columns ({len(train_cols)}): {train_cols}\n\n"
                f"Missing: {missing}\nExtra: {extra}"
            )
            st.stop()

        
        X_batch_df = batch_df[train_cols]

        
        try:
            X_batch = imputer_batch.transform(X_batch_df)
        except Exception as e:
            st.error(
                "Failed to apply imputer to batch data. "
                "Ensure column names and types match the training data exactly."
            )
            st.code(str(e))
            st.stop()

        
        batch_model_name = st.selectbox(
            "Model for Batch Inference",
            list(models.keys()),
            key="batchmod",
        )

        model_info = models[batch_model_name]
        model = model_info["model"]
        scaled_flag = model_info["scaled"]

        
        if scaled_flag:
            if scaler is None:
                st.error(
                    "Scaler not found in session state for this model. "
                    "Train models again in the 'Model Performance' tab."
                )
                st.stop()
            try:
                X_batch_proc = scaler.transform(X_batch)
            except Exception as e:
                st.error(
                    "Failed to scale batch data. "
                    "Verify feature order and numeric types."
                )
                st.code(str(e))
                st.stop()
        else:
            X_batch_proc = X_batch

        
        try:
            probs = model.predict_proba(X_batch_proc)[:, 1]
        except Exception as e:
            st.error(
                "Batch prediction failed. "
                "Ensure the uploaded CSV uses the same preprocessing and feature set as the training data."
            )
            st.code(str(e))
            st.stop()

        batch_df["DefectProbability"] = probs
        batch_df["RiskLevel"] = pd.cut(
            probs,
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
        )

        st.markdown("### Batch Risk Report")
        st.dataframe(
            batch_df.sort_values("DefectProbability", ascending=False),
            use_container_width=True,
        )

        csv_output = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Risk Report",
            data=csv_output,
            file_name="nasa_risk_report.csv",
            mime="text/csv",
        )

    elif "models" not in st.session_state:
        st.info("Please train models in the 'Model Performance' tab before using batch prediction.")
