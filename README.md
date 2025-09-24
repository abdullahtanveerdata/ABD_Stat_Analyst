A Streamlit app to run common statistical analyses with clean tables, effect sizes, diagnostics, and ready-made charts—no coding required.

✨ Features

Data import: CSV/XLS/XLSX; Excel sheet picker; toggle to treat the first row as headers.

Preprocessing: Trim strings, standardize missing values, optional imputation (mean/median/mode).

Assumption checks: Shapiro–Wilk per group and Levene’s test for homogeneity.

Auto-suggested tests: Chooses the right test based on normality, variance equality, and group count.

Effect sizes + diagnostics: Cohen’s d, η²/ω², Cliff’s δ, epsilon², Cramér’s V/φ; QQ plots & residual checks.

Graph builder: Histogram, box/violin, scatter (with trendline), stacked/100% stacked bars, mosaic.

Downloadable reports: One-click ZIP with all result tables (CSVs) + interpretation + meta.

Built-in demos: Load sample datasets to try the flow instantly.

🔬 Analyses Supported

Scenario	Suggested tests	Post-hoc	Effect size(s)
Two groups (numeric ~ categorical)	Independent t-test / Welch t-test / Mann–Whitney U	—	Cohen’s d (t/Welch), Cliff’s δ (MWU)
≥ Three groups (numeric ~ categorical)	One-way ANOVA / Welch ANOVA / Kruskal–Wallis	Tukey HSD / Games–Howell / Dunn (Bonf.)	η², ω² (ANOVA/Welch), epsilon² (Kruskal)
Two numeric	Pearson / Spearman correlation	—	—
Two categorical	Chi-square (or Fisher’s exact for 2×2 with low counts)	—	Cramér’s V (χ²), φ (2×2)

The app shows omnibus results, post-hoc (when applicable), and a concise, thesis-ready interpretation for each analysis.

📊 Visualizations

Numeric: Histogram; Box/Violin by group; Scatter with OLS trendline

Categorical: Bar count; Stacked / 100% stacked bar; Mosaic plot

Diagnostics: QQ plot of residuals; Residuals-vs-Fitted (for parametric OLS models)

🧭 Usage

Upload CSV/XLS/XLSX (select sheet if Excel; set “First row contains column names” if needed).

Preprocess (trim text, handle missing values, optional imputation).

Pick analysis:

Compare groups (numeric ~ categorical)

Correlation (two numeric)

Association (two categorical)

Graph builder

Review assumptions → Run test → Read interpretation → Download ZIP (tables + interpretation + meta).

Tip: Use the demo buttons in the sidebar to test quickly (tips, iris, etc.).

📦 Outputs

Omnibus table, post-hoc table(s), effect sizes

Interpretation text and metadata

One-click ZIP export with CSVs + INTERPRETATION.txt + META.txt

🗺️ Roadmap

Confidence intervals for effect sizes (d, r, η²/ω², δ)

DOCX/PDF report export

More diagnostics (influence, Cook’s distance), multiple outcomes

Additional visual themes

⚠️ Disclaimer

This tool helps streamline common analyses, but it’s not a substitute for a statistician. Always confirm assumptions and the suitability of each method for your study design.
To RUN this tool
https://abdstatanalyst-evzi9vcrmkdqml2p6ufjjy.streamlit.app/
