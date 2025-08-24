# streamlit_app.py
# -------------------------------------------------------------
# Stat‑Analyst: interactive stats web app
# Upload CSV/XLSX → preprocess → tests (numeric & categorical) →
# effect sizes → diagnostics (QQ/residuals) → results + downloads → graphs
# -------------------------------------------------------------

import io
import zipfile
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.mosaicplot import mosaic
import scikit_posthocs as sp
import plotly.express as px
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stat‑Analyst", layout="wide")

# ====================== Utilities ======================

def read_data_with_header(upload, first_row_header: bool = True) -> pd.DataFrame:
    """Read CSV/XLS/XLSX. If first_row_header is True → header=0 else header=None."""
    if upload is None:
        return pd.DataFrame()
    suffix = upload.name.lower().split(".")[-1]
    header_arg = 0 if first_row_header else None
    try:
        if suffix in ["csv", "txt"]:
            try:
                return pd.read_csv(upload, header=header_arg)
            except Exception:
                upload.seek(0)
                return pd.read_csv(upload, sep=";", header=header_arg)
        elif suffix in ["xlsx", "xls"]:
            xls = pd.ExcelFile(upload)
            sheet = st.sidebar.selectbox("Select Excel sheet", xls.sheet_names)
            return pd.read_excel(xls, sheet_name=sheet, header=header_arg)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return pd.DataFrame()


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df


def impute(df: pd.DataFrame, num_strategy: str = "drop", cat_strategy: str = "drop") -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = out.select_dtypes(exclude=[np.number]).columns.tolist()
    if num_strategy == "mean" and num_cols:
        out[num_cols] = out[num_cols].apply(lambda s: s.fillna(s.mean()))
    elif num_strategy == "median" and num_cols:
        out[num_cols] = out[num_cols].apply(lambda s: s.fillna(s.median()))
    if cat_strategy == "mode":
        for c in cat_cols:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].mode().iloc[0])
    return out


def is_categorical(series: pd.Series, max_unique_ratio: float = 0.2) -> bool:
    """Heuristic: non-numeric OR numeric with low distinct ratio treated as categorical."""
    if not pd.api.types.is_numeric_dtype(series):
        return True
    n = len(series.dropna())
    if n == 0:
        return True
    return series.dropna().nunique() / n <= max_unique_ratio


# ===================== Assumptions =====================

def shapiro_safe(x: pd.Series) -> Tuple[float, float]:
    x = x.dropna()
    if len(x) > 5000:
        x = x.sample(5000, random_state=42)
    if len(x) < 3:
        return np.nan, np.nan
    return stats.shapiro(x)


def levene_safe(data: List[pd.Series]) -> Tuple[float, float]:
    groups = [g.dropna() for g in data if len(g.dropna()) > 1]
    if len(groups) < 2:
        return np.nan, np.nan
    return stats.levene(*groups, center='median')


# ================= Effect Size Helpers =================

def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().values
    b = b.dropna().values
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = sum((ai > bj) for ai in a for bj in b)
    less = sum((ai < bj) for ai in a for bj in b)
    return (greater - less) / (len(a) * len(b))


def eta_omega_from_anova(anova_tbl: pd.DataFrame, effect_term: str) -> Tuple[float, float]:
    # Requires columns: sum_sq, df and a 'Residual' row
    if not {"sum_sq", "df"}.issubset(set(anova_tbl.columns)):
        return np.nan, np.nan
    if effect_term not in anova_tbl.index or "Residual" not in anova_tbl.index:
        return np.nan, np.nan
    ss_between = float(anova_tbl.loc[effect_term, "sum_sq"])  # type: ignore
    df_between = float(anova_tbl.loc[effect_term, "df"])      # type: ignore
    ss_error = float(anova_tbl.loc["Residual", "sum_sq"])    # type: ignore
    df_error = float(anova_tbl.loc["Residual", "df"])        # type: ignore
    ss_total = ss_between + ss_error
    ms_error = ss_error / df_error if df_error > 0 else np.nan
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    omega2 = (ss_between - df_between * ms_error) / (ss_total + ms_error) if ms_error and ss_total + ms_error > 0 else np.nan
    return eta2, omega2


def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    # Nonparametric effect size for Kruskal–Wallis
    if n <= k or n <= 0 or k <= 1:
        return np.nan
    return (H - k + 1) / (n - k)


# ================= Suggestion Logic ====================

def suggest_test_numeric_vs_group(y: pd.Series, g: pd.Series) -> dict:
    df = pd.DataFrame({"y": y, "g": g}).dropna()
    if df.empty or df["g"].nunique() < 2:
        return {"norm_table": pd.DataFrame(), "levene_p": np.nan,
                "suggestion": "Not enough groups to test", "all_normal": False, "homogeneous": False}
    uniq = df.g.unique()
    groups = [df.loc[df.g == lvl, "y"] for lvl in uniq]
    norm_rows, flags = [], []
    for lvl, arr in zip(uniq, groups):
        _, p = shapiro_safe(arr)
        norm_rows.append({"group": lvl, "shapiro_p": p})
        flags.append(False if pd.isna(p) else p >= 0.05)
    all_normal = all(flags) if flags else False
    _, lev_p = levene_safe(groups)
    homo = (lev_p >= 0.05) if not np.isnan(lev_p) else False
    k = len(groups)
    if k == 2:
        choice = "Independent t-test" if all_normal and homo else ("Welch t-test" if all_normal else "Mann–Whitney U")
    else:
        choice = "One-way ANOVA" if all_normal and homo else ("Welch ANOVA" if all_normal else "Kruskal–Wallis")
    return {"norm_table": pd.DataFrame(norm_rows), "levene_p": lev_p,
            "suggestion": choice, "all_normal": all_normal, "homogeneous": homo}


def suggest_test_correlation(x: pd.Series, y: pd.Series) -> str:
    _, px = shapiro_safe(x)
    _, py = shapiro_safe(y)
    return "Pearson correlation" if (not pd.isna(px) and px >= 0.05 and not pd.isna(py) and py >= 0.05) else "Spearman correlation"


# ================== Statistical Tests ==================
# ---- Numeric vs Group ----

def run_independent_t(df: pd.DataFrame, y: str, g: str):
    groups = [d.dropna() for _, d in df[[y, g]].dropna().groupby(g)]
    if len(groups) != 2:
        return pd.DataFrame(), "Exactly two groups are required for an independent t-test.", {}
    a, b = groups
    t, p = stats.ttest_ind(a[y], b[y], equal_var=True)
    eff = {
        "Cohen_d": cohen_d(a[y], b[y])
    }
    table = pd.DataFrame({"statistic": [t], "p_value": [p], "cohen_d": [eff["Cohen_d"]]})
    interp = f"The independent t-test compared the means of {y} between the two {g} groups. The p-value was {p:.4g}. Cohen's d ≈ {eff['Cohen_d']:.3f}."
    return table, interp, eff


def run_welch_t(df: pd.DataFrame, y: str, g: str):
    groups = [d.dropna() for _, d in df[[y, g]].dropna().groupby(g)]
    if len(groups) != 2:
        return pd.DataFrame(), "Exactly two groups are required for a Welch t-test.", {}
    a, b = groups
    t, p = stats.ttest_ind(a[y], b[y], equal_var=False)
    eff = {
        "Cohen_d": cohen_d(a[y], b[y])  # still informative even under heteroscedasticity
    }
    table = pd.DataFrame({"statistic": [t], "p_value": [p], "cohen_d": [eff["Cohen_d"]]})
    interp = f"The Welch t-test compared unequal-variance means of {y} between the two {g} groups. The p-value was {p:.4g}. Cohen's d ≈ {eff['Cohen_d']:.3f}."
    return table, interp, eff


def run_mannwhitney(df: pd.DataFrame, y: str, g: str):
    groups = [d.dropna() for _, d in df[[y, g]].dropna().groupby(g)]
    if len(groups) != 2:
        return pd.DataFrame(), "Exactly two groups are required for Mann–Whitney U.", {}
    a, b = groups
    u, p = stats.mannwhitneyu(a[y], b[y], alternative='two-sided')
    delta = cliffs_delta(a[y], b[y])
    table = pd.DataFrame({"U": [u], "p_value": [p], "cliffs_delta": [delta]})
    interp = f"The Mann–Whitney U test compared the distributions of {y} between the two {g} groups. The p-value was {p:.4g}. Cliff's δ ≈ {delta:.3f}."
    eff = {"Cliffs_delta": delta}
    return table, interp, eff


def run_oneway_anova(df: pd.DataFrame, y: str, g: str):
    df_sub = df.dropna(subset=[y, g])
    model = smf.ols(f"{y} ~ C({g})", data=df_sub).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    tuk = pairwise_tukeyhsd(endog=df_sub[y], groups=df_sub[g])
    tuk_df = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    term = f"C({g})"
    eta2, omega2 = eta_omega_from_anova(anova_tbl, term)
    interp = (f"One-way ANOVA tested mean differences of {y} across levels of {g}. "
              f"The omnibus p-value was {anova_tbl.loc[term, 'PR(>F)']:.4g}. "
              f"Effect sizes: η² ≈ {eta2:.3f}, ω² ≈ {omega2:.3f}. Significant pairwise differences are in the post‑hoc table.")
    eff = {"eta_squared": eta2, "omega_squared": omega2}
    return anova_tbl.reset_index().rename(columns={"index": "term"}), tuk_df, interp, eff, model


def run_welch_anova(df: pd.DataFrame, y: str, g: str):
    try:
        import pingouin as pg
        df_sub = df.dropna(subset=[y, g])
        welch = pg.welch_anova(dv=y, between=g, data=df_sub)
        gh = pg.pairwise_gameshowell(dv=y, between=g, data=df_sub)
        # Use eta² from classic OLS as an approximation for reporting
        model = smf.ols(f"{y} ~ C({g})", data=df_sub).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        term = f"C({g})"
        eta2, omega2 = eta_omega_from_anova(anova_tbl, term)
        interp = (f"Welch ANOVA tested mean differences of {y} across {g} allowing unequal variances. "
                  f"The omnibus p-value was {welch.loc[0, 'p-unc']:.4g}. "
                  f"Effect sizes (from OLS): η² ≈ {eta2:.3f}, ω² ≈ {omega2:.3f}. Games–Howell post‑hoc adjusts for heteroscedasticity.")
        eff = {"eta_squared": eta2, "omega_squared": omega2}
        return welch.rename(columns={"Source": "term"}), gh, interp, eff, model
    except Exception:
        st.warning("Install 'pingouin' for Welch ANOVA + Games–Howell. Falling back to standard ANOVA.")
        a, b, c, d, model = run_oneway_anova(df, y, g)
        return a, b, c, d, model


def run_kruskal(df: pd.DataFrame, y: str, g: str):
    df_sub = df.dropna(subset=[y, g])
    groups = [d[y] for _, d in df_sub[[y, g]].groupby(g)]
    if len(groups) < 2:
        return pd.DataFrame(), pd.DataFrame(), "At least two groups are required for Kruskal–Wallis.", {}, None
    h, p = stats.kruskal(*groups)
    omni = pd.DataFrame({"statistic": [h], "p_value": [p]})
    dunn = sp.posthoc_dunn(df_sub[[y, g]], val_col=y, group_col=g, p_adjust='bonferroni')
    dunn = dunn.reset_index().rename(columns={"index": "group1"})
    eps2 = epsilon_squared_kruskal(h, len(df_sub), df_sub[g].nunique())
    interp = (f"Kruskal–Wallis tested distributional differences of {y} across {g}. "
              f"The omnibus p-value was {p:.4g}. Epsilon² ≈ {eps2:.3f}. "
              f"Dunn’s post‑hoc (Bonferroni) identifies pairwise differences.")
    eff = {"epsilon_squared": eps2}
    return omni, dunn, interp, eff, None

# ---- Categorical vs Categorical ----

def cramers_v(confusion: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.values.sum()
    r, k = confusion.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1)))) if n > 0 else np.nan


def run_categorical_association(df: pd.DataFrame, a: str, b: str):
    """Chi-square of independence; fallback to Fisher for 2x2 with small expected counts.
    Returns: (contingency_table, result_table, interpretation, test_used, residuals_table, effects)
    """
    df_sub = df.dropna(subset=[a, b]).copy()
    df_sub[a] = df_sub[a].astype(str)
    df_sub[b] = df_sub[b].astype(str)
    table = pd.crosstab(df_sub[a], df_sub[b])
    if table.size == 0:
        return pd.DataFrame(), pd.DataFrame(), "No data available after filtering.", "", None, {}

    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    if table.shape == (2, 2) and (expected < 5).any():
        odds, p_fisher = stats.fisher_exact(table.values)
        result = pd.DataFrame({"test": ["Fisher exact (2x2)"], "odds_ratio": [odds], "p_value": [p_fisher]})
        phi = np.sqrt(chi2 / table.values.sum())
        effects = {"phi": phi}
        interp = (f"Fisher's exact test evaluated the association between {a} and {b} in a 2×2 table. "
                  f"The p-value was {p_fisher:.4g}. Phi ≈ {phi:.3f}.")
        return table, result, interp, "Fisher", None, effects
    else:
        result = pd.DataFrame({"test": ["Chi-square of independence"], "chi2": [chi2], "dof": [dof], "p_value": [p]})
        cv = cramers_v(table)
        resid = (table - expected) / np.sqrt(expected)
        resid = pd.DataFrame(resid, index=table.index, columns=table.columns)
        effects = {"cramers_v": cv}
        interp = (f"The chi-square test assessed independence between {a} and {b}. "
                  f"The p-value was {p:.4g}. Cramer's V ≈ {cv:.3f}. Cells with |standardized residual| > 1.96 contribute notably.")
        return table, result, interp, "Chi-square", resid, effects


# ================= Diagnostics & Plots =================

def qqplot_series(x: pd.Series, title: str):
    fig = plt.figure(figsize=(5, 4), dpi=150)
    stats.probplot(x.dropna(), dist="norm", plot=plt)
    plt.title(title)
    st.pyplot(fig)


def ols_diagnostics(model):
    fitted = model.fittedvalues
    resid = model.resid
    # QQ plot of residuals
    fig1 = plt.figure(figsize=(5, 4), dpi=150)
    sm.qqplot(resid, line='45', fit=True)
    plt.title("QQ plot of residuals")
    st.pyplot(fig1)
    # Residuals vs Fitted
    fig2 = px.scatter(x=fitted, y=resid, labels={"x": "Fitted", "y": "Residuals"}, title="Residuals vs Fitted")
    st.plotly_chart(fig2, use_container_width=True)


# ================== Downloads (ZIP) ====================

def build_zip_export(name_prefix: str, tables: Dict[str, pd.DataFrame], interpretation: str, meta: Dict[str, str]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        # write tables as CSV
        for fname, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                continue
            zf.writestr(f"{name_prefix}_{fname}.csv", df.to_csv(index=False))
        # write interpretation and meta
        zf.writestr(f"{name_prefix}_INTERPRETATION.txt", interpretation)
        meta_lines = "\n".join([f"{k}: {v}" for k, v in meta.items()])
        zf.writestr(f"{name_prefix}_META.txt", meta_lines)
    return bio.getvalue()


# ======================== UI ==========================

st.title("📊 Stat‑Analyst — Interactive Statistical Analysis")
st.caption("Upload data ➜ preprocess ➜ normality ➜ auto‑suggest tests ➜ effect sizes ➜ diagnostics ➜ downloads ➜ graphs")

with st.sidebar:
    st.header("1) Upload data")
    first_row_header = st.checkbox("First row contains column names", value=True)
    up = st.file_uploader("CSV or Excel", type=["csv", "txt", "xlsx", "xls"]) 

    st.markdown("**Or load a demo dataset (quick test):**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Demo: ANOVA (tips)"):
            import seaborn as sns
            df_demo = sns.load_dataset("tips")[["total_bill", "day"]].rename(columns={"total_bill": "value", "day": "group"})
            st.session_state["_demo_df"] = df_demo
    with c2:
        if st.button("Demo: Correlation (iris)"):
            import seaborn as sns
            df_demo = sns.load_dataset("iris")[ ["sepal_length", "sepal_width"] ].rename(columns={"sepal_length": "x", "sepal_width": "y"})
            st.session_state["_demo_df"] = df_demo
    with c3:
        if st.button("Demo: Chi-square (tips)"):
            import seaborn as sns
            df_demo = sns.load_dataset("tips")[ ["sex", "smoker"] ].rename(columns={"sex": "A", "smoker": "B"})
            st.session_state["_demo_df"] = df_demo

    st.header("2) Preprocess")
    do_basic = st.checkbox("Trim text & standardize missing values", value=True)
    num_imp = st.selectbox("Numeric missing values", ["drop", "mean", "median"], index=0)
    cat_imp = st.selectbox("Categorical missing values", ["drop", "mode"], index=0)

    st.header("3) Analysis type")
    analysis = st.radio(
        "Choose an analysis:",
        [
            "Compare groups (numeric vs categorical)",
            "Correlation (two numeric)",
            "Association (two categorical)",
            "Graph builder",
        ],
    )

# Prefer upload; else demo
if up is None and "_demo_df" in st.session_state:
    raw = st.session_state["_demo_df"].copy()
else:
    raw = read_data_with_header(up, first_row_header=first_row_header)

if raw.empty:
    st.info("Upload a CSV/XLSX file or load a demo dataset to begin.")
    st.stop()

if do_basic:
    raw = basic_clean(raw)
raw = impute(raw, num_strategy=num_imp, cat_strategy=cat_imp)

st.subheader("Data preview")
st.dataframe(raw.head(30), use_container_width=True)

num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in raw.columns if is_categorical(raw[c])]
all_cols = raw.columns.tolist()

st.divider()

# --------------------- Analyses ------------------------
if analysis == "Compare groups (numeric vs categorical)":
    if not num_cols:
        st.error("No numeric columns detected. Switch to 'Association (two categorical)' or 'Graph builder', or upload data with numeric values.")
        st.stop()

    st.subheader("Compare groups: set variables")
    y = st.selectbox("Numeric outcome (Y)", options=num_cols, index=0)
    group_opts = [c for c in all_cols if c != y]
    if not group_opts:
        st.error("No grouping (categorical) variable available. Add a categorical column or choose another dataset.")
        st.stop()
    g = st.selectbox("Grouping variable (categorical)", options=group_opts, index=0)

    if y not in raw.columns or g not in raw.columns:
        st.error("Selected columns not found in the dataset.")
        st.stop()

    df_ng = raw[[y, g]].dropna().copy()
    if df_ng.empty:
        st.error("Selected columns have only missing values after preprocessing. Try different columns or preprocessing options.")
        st.stop()

    df_ng[g] = df_ng[g].astype(str)
    if df_ng[g].nunique() < 2:
        st.error("Grouping variable must contain at least two distinct groups.")
        st.stop()

    st.markdown("### Assumption checks")
    sugg = suggest_test_numeric_vs_group(df_ng[y], df_ng[g])
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(sugg["norm_table"], use_container_width=True)
    with col2:
        lev_disp = "NA" if np.isnan(sugg['levene_p']) else f"{sugg['levene_p']:.4g}"
        st.metric("Levene p (homogeneity)", lev_disp)
        st.metric("All groups normal?", "Yes" if sugg["all_normal"] else "No")
        st.metric("Variances homogeneous?", "Yes" if sugg["homogeneous"] else "No")

    st.info(f"📌 Suggested test: **{sugg['suggestion']}**")

    if st.button("Run test"):
        k = df_ng[g].nunique()
        export_tables: Dict[str, pd.DataFrame] = {}
        interpretation: str = ""
        meta = {"analysis": "numeric_vs_group", "Y": y, "group": g, "suggested": sugg['suggestion']}
        model_for_diag = None

        if k == 2:
            if sugg['suggestion'] == 'Independent t-test':
                tbl, msg, eff = run_independent_t(df_ng, y, g)
                export_tables["t_test"] = tbl
                interpretation = msg
                st.markdown("### Results (independent t-test)")
                st.dataframe(tbl, use_container_width=True)
                st.success(msg)
                st.info(f"Effect size — Cohen's d: {eff['Cohen_d']:.3f}")
                # Fit OLS for diagnostics
                model_for_diag = smf.ols(f"{y} ~ C({g})", data=df_ng).fit()
            elif sugg['suggestion'] == 'Welch t-test':
                tbl, msg, eff = run_welch_t(df_ng, y, g)
                export_tables["welch_t"] = tbl
                interpretation = msg
                st.markdown("### Results (Welch t-test)")
                st.dataframe(tbl, use_container_width=True)
                st.success(msg)
                st.info(f"Effect size — Cohen's d: {eff['Cohen_d']:.3f}")
                model_for_diag = smf.ols(f"{y} ~ C({g})", data=df_ng).fit()
            else:
                tbl, msg, eff = run_mannwhitney(df_ng, y, g)
                export_tables["mann_whitney"] = tbl
                interpretation = msg
                st.markdown("### Results (Mann–Whitney U)")
                st.dataframe(tbl, use_container_width=True)
                st.success(msg)
                st.info(f"Effect size — Cliff's δ: {eff['Cliffs_delta']:.3f}")
        else:
            if sugg['suggestion'] in ['One-way ANOVA']:
                anova, posthoc, msg, eff, model = run_oneway_anova(df_ng, y, g)
                export_tables["anova_omnibus"] = anova
                export_tables["tukey_posthoc"] = posthoc
                interpretation = msg
                model_for_diag = model
                st.markdown("### Omnibus (one‑way ANOVA)")
                st.dataframe(anova, use_container_width=True)
                st.markdown("### Post‑hoc (Tukey HSD)")
                st.dataframe(posthoc, use_container_width=True)
                st.success(msg)
                st.info(f"Effect sizes — η²: {eff['eta_squared']:.3f}, ω²: {eff['omega_squared']:.3f}")
            elif sugg['suggestion'] in ['Welch ANOVA']:
                anova, posthoc, msg, eff, model = run_welch_anova(df_ng, y, g)
                export_tables["welch_anova_omnibus"] = anova
                export_tables["games_howell_posthoc"] = posthoc
                interpretation = msg
                model_for_diag = model
                st.markdown("### Omnibus (Welch ANOVA)")
                st.dataframe(anova, use_container_width=True)
                st.markdown("### Post‑hoc (Games–Howell)")
                st.dataframe(posthoc, use_container_width=True)
                st.success(msg)
                st.info(f"Effect sizes — η²: {eff['eta_squared']:.3f}, ω²: {eff['omega_squared']:.3f}")
            else:
                omni, dunn, msg, eff, model = run_kruskal(df_ng, y, g)
                export_tables["kruskal_omnibus"] = omni
                export_tables["dunn_posthoc_bonf"] = dunn
                interpretation = msg
                model_for_diag = model
                st.markdown("### Omnibus (Kruskal–Wallis)")
                st.dataframe(omni, use_container_width=True)
                st.markdown("### Post‑hoc (Dunn, Bonferroni)")
                st.dataframe(dunn, use_container_width=True)
                st.success(msg)
                st.info(f"Effect size — Epsilon²: {eff['epsilon_squared']:.3f}")

        # Diagnostics (if we have an OLS model)
        with st.expander("Diagnostics (QQ & Residuals)"):
            if model_for_diag is not None:
                ols_diagnostics(model_for_diag)
            else:
                st.caption("Diagnostics are most informative for parametric OLS models (t/ANOVA). For nonparametric tests, inspect distribution plots below.")
                # Provide per-group QQs for reference
                for lvl, arr in df_ng.groupby(g)[y]:
                    qqplot_series(arr, title=f"QQ: {y} in group {lvl}")

        # Downloads
        if export_tables:
            zip_bytes = build_zip_export(name_prefix=f"{y}_by_{g}", tables=export_tables, interpretation=interpretation, meta=meta)
            st.download_button("⬇️ Download results (ZIP)", data=zip_bytes, file_name=f"{y}_by_{g}_results.zip", mime="application/zip")

elif analysis == "Correlation (two numeric)":
    if len(num_cols) < 2:
        st.error("Need at least two numeric columns for correlation analysis. Try 'Association (two categorical)' or 'Graph builder'.")
        st.stop()

    st.subheader("Correlation: set variables")
    x = st.selectbox("Numeric variable X", options=num_cols, index=0)
    y = st.selectbox("Numeric variable Y", options=[c for c in num_cols if c != x], index=0)

    method = suggest_test_correlation(raw[x], raw[y])
    st.info(f"📌 Suggested method: **{method}**")

    if st.button("Run correlation"):
        df_xy = raw.dropna(subset=[x, y])
        if df_xy.empty:
            st.error("Selected columns have only missing values after preprocessing.")
            st.stop()
        if method == "Pearson correlation":
            r, p = stats.pearsonr(df_xy[x], df_xy[y])
            tbl = pd.DataFrame({"r": [r], "p_value": [p]})
            interp = f"Pearson correlation quantified the linear association between {x} and {y}. The p-value was {p:.4g}."
        else:
            rho, p = stats.spearmanr(df_xy[x], df_xy[y])
            tbl = pd.DataFrame({"rho": [rho], "p_value": [p]})
            interp = f"Spearman correlation quantified the rank-based association between {x} and {y}. The p-value was {p:.4g}."
        st.markdown("### Correlation results")
        st.dataframe(tbl, use_container_width=True)
        st.success(interp)

        # Diagnostics: QQ for X & Y, plus residuals of OLS (if Pearson)
        with st.expander("Diagnostics (QQ & Residuals)"):
            qqplot_series(df_xy[x], title=f"QQ: {x}")
            qqplot_series(df_xy[y], title=f"QQ: {y}")
            if method == "Pearson correlation":
                model = smf.ols(f"{y} ~ {x}", data=df_xy).fit()
                ols_diagnostics(model)

        # Download
        zip_bytes = build_zip_export(name_prefix=f"corr_{x}_{y}", tables={"correlation": tbl}, interpretation=interp, meta={"analysis": "correlation", "method": method, "x": x, "y": y})
        st.download_button("⬇️ Download results (ZIP)", data=zip_bytes, file_name=f"corr_{x}_{y}_results.zip", mime="application/zip")

elif analysis == "Association (two categorical)":
    if len(cat_cols) < 2:
        st.error("Need at least two categorical columns for association analysis.")
        st.stop()

    st.subheader("Association: set variables")
    a = st.selectbox("Categorical variable A", options=cat_cols, index=0)
    b = st.selectbox("Categorical variable B", options=[c for c in cat_cols if c != a], index=0)

    if st.button("Run association test"):
        table, result, interp, test_used, resid, effects = run_categorical_association(raw, a, b)
        st.markdown("### Contingency table")
        st.dataframe(table, use_container_width=True)
        st.markdown("### Test result")
        st.dataframe(result, use_container_width=True)
        if resid is not None:
            st.markdown("### Standardized residuals (>|1.96| noteworthy)")
            st.dataframe(resid, use_container_width=True)
        if effects:
            eff_str = ", ".join([f"{k}: {v:.3f}" for k, v in effects.items()])
            st.info(f"Effect size — {eff_str}")
        st.success(interp)

        # Suggested plots
        st.markdown("### Suggested plots")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Stacked bar")
            fig1 = px.histogram(raw.dropna(subset=[a, b]), x=a, color=b, barmode="stack")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.caption("100% stacked bar")
            fig2 = px.histogram(raw.dropna(subset=[a, b]), x=a, color=b, barmode="stack", barnorm="percent")
            st.plotly_chart(fig2, use_container_width=True)
        with c3:
            st.caption("Mosaic")
            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            mosaic(raw[[a, b]].dropna(), [a, b], ax=ax, title=f"Mosaic: {a} × {b}")
            st.pyplot(fig)

        # Download
        tables = {"contingency": table, "test_result": result}
        if resid is not None:
            tables["std_residuals"] = resid
        zip_bytes = build_zip_export(name_prefix=f"assoc_{a}_x_{b}", tables=tables, interpretation=interp, meta={"analysis": "association", "test": test_used, "A": a, "B": b})
        st.download_button("⬇️ Download results (ZIP)", data=zip_bytes, file_name=f"assoc_{a}_x_{b}_results.zip", mime="application/zip")

else:  # Graph builder
    st.subheader("Graph builder")
    num_avail = num_cols
    cat_avail = [c for c in all_cols if c not in num_cols]

    kind = st.selectbox(
        "Choose a chart type",
        [
            "Histogram (numeric)",
            "Box (numeric by group)",
            "Violin (numeric by group)",
            "Scatter (two numeric)",
            "Bar count (categorical)",
            "Stacked bar (two categorical)",
            "100% stacked bar (two categorical)",
            "Mosaic (two categorical)",
        ],
        index=0,
    )

    def plot_graph(df: pd.DataFrame, kind: str, x: str = None, y: str = None, color: str = None):
        if kind == "Histogram (numeric)":
            fig = px.histogram(df.dropna(subset=[x]), x=x)
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Box (numeric by group)":
            fig = px.box(df.dropna(subset=[x, color]), x=color, y=x, points="all")
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Violin (numeric by group)":
            fig = px.violin(df.dropna(subset=[x, color]), x=color, y=x, box=True, points="all")
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Scatter (two numeric)":
            fig = px.scatter(df.dropna(subset=[x, y]), x=x, y=y, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Bar count (categorical)":
            fig = px.histogram(df.dropna(subset=[x]), x=x)
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Stacked bar (two categorical)":
            fig = px.histogram(df.dropna(subset=[x, color]), x=x, color=color, barmode="stack")
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "100% stacked bar (two categorical)":
            fig = px.histogram(df.dropna(subset=[x, color]), x=x, color=color, barmode="stack", barnorm="percent")
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Mosaic (two categorical)":
            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            mosaic(df[[x, color]].dropna(), [x, color], ax=ax, title=f"Mosaic: {x} × {color}")
            st.pyplot(fig)

    if kind == "Histogram (numeric)":
        if not num_avail:
            st.error("No numeric columns available.")
        else:
            x = st.selectbox("Numeric column", options=num_avail, index=0)
            plot_graph(raw, kind, x=x)

    elif kind in ["Box (numeric by group)", "Violin (numeric by group)"]:
        if not num_avail or not all_cols:
            st.error("Need a numeric column and a grouping column.")
        else:
            y = st.selectbox("Numeric column", options=num_avail, index=0)
            g = st.selectbox("Grouping column", options=[c for c in all_cols if c != y], index=0)
            plot_graph(raw, kind, x=y, color=g)

    elif kind == "Scatter (two numeric)":
        if len(num_avail) < 2:
            st.error("Need at least two numeric columns.")
        else:
            x = st.selectbox("X (numeric)", options=num_avail, index=0)
            y = st.selectbox("Y (numeric)", options=[c for c in num_avail if c != x], index=0)
            plot_graph(raw, kind, x=x, y=y)

    elif kind == "Bar count (categorical)":
        if not all_cols:
            st.error("Need at least one column.")
        else:
            x = st.selectbox("Categorical column", options=all_cols, index=0)
            plot_graph(raw, kind, x=x)

    elif kind in ["Stacked bar (two categorical)", "100% stacked bar (two categorical)", "Mosaic (two categorical)"]:
        if len(all_cols) < 2:
            st.error("Need at least two columns.")
        else:
            a = st.selectbox("A (categorical)", options=all_cols, index=0)
            b = st.selectbox("B (categorical)", options=[c for c in all_cols if c != a], index=0)
            plot_graph(raw, kind, x=a, color=b)

st.divider()
st.caption("Now with effect sizes (Cohen’s d, η²/ω², Cliff’s δ, epsilon²), diagnostics (QQ & residuals), and one‑click ZIP downloads of results.")
st.caption("Developed by Muhammad Abdullah Tanveer-BioInfoQuant (https://bioinfoquant.com)")