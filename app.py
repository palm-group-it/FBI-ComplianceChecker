import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Független Biztosításközvetítő Iroda – Risk kalkulátor",
    page_icon="🕵️",
    layout="wide"
)

# -------------------------------------------------
# LIGHT UI CSS
# -------------------------------------------------
CUSTOM_CSS = """
<style>
  .stApp {
    background: #f6f7fb;
    color: #0f172a;
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }

  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  /* Top hero */
  .hero {
    background: #ffffff;
    border: 1px solid #e7e9f2;
    border-radius: 20px;
    padding: 22px 22px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 16px;
  }
  .hero h1 {
    font-size: 2.0rem;
    margin: 0 0 6px 0;
    letter-spacing: 0.2px;
  }
  .hero p {
    margin: 0;
    color: #475569;
    font-size: 1.02rem;
    line-height: 1.6;
  }

  /* Cards */
  .card {
    background: #ffffff;
    border: 1px solid #e7e9f2;
    border-radius: 16px;
    padding: 16px 16px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
  }
  .card-title {
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 8px;
    color: #0f172a;
  }
  .muted {
    color: #64748b;
    font-size: 0.95rem;
  }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: .3px;
    margin-right: 6px;
    border: 1px solid #e7e9f2;
  }
  .badge-up   { background: #eafaf0; color: #166534; border-color:#c7f0d6; }
  .badge-down { background: #fdecec; color: #991b1b; border-color:#f7caca; }
  .badge-info { background: #eef2ff; color: #3730a3; border-color:#dfe3ff; }

  /* Buttons */
  .stButton > button {
    /* Download button (st.download_button) */
  .stDownloadButton > button {
    border-radius: 12px !important;
    padding: 0.65rem 1rem !important;
    font-weight: 700 !important;
    border: 1px solid #e7e9f2 !important;
    background: #0f172a !important;   /* sötét háttér */
    color: #ffffff !important;        /* FEHÉR szöveg, mindig látszik */
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.10) !important;
  }

  .stDownloadButton > button:hover {
    background: #111c34 !important;
    transform: translateY(-1px);
  }

    border-radius: 12px !important;
    padding: 0.65rem 1rem !important;
    font-weight: 700 !important;
    border: 1px solid #e7e9f2 !important;
    background: #0f172a !important;
    color: #ffffff !important;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.10) !important;
  }
  .stButton > button:hover {
    background: #111c34 !important;
    transform: translateY(-1px);
  }

  /* Dataframe wrap */
  .dataframe-wrap {
    background: #ffffff;
    border: 1px solid #e7e9f2;
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    margin-top: 8px;
  }

  /* Metric cards tweak */
  div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e7e9f2;
    border-radius: 14px;
    padding: 12px 10px;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------
# HERO / HEADER (Hungarian)
# -------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>
        Független Biztosításközvetítő Iroda – Risk kalkulátor
        <span class="badge badge-info">Darabszám alapú eltérésvizsgálat</span>
      </h1>
      <p>
        Ez az eszköz az FB Irodán belüli megfelelőségi ellenőrzést támogatja:
        a feltöltött Excel állomány alapján kiszámolja a teljes cég biztosító-mixét
        ágazatonként, majd megmutatja, hogy az egyes üzletkötők mely ágazatokban térnek el
        ettől jelentősen. A kiugró (felfelé és lefelé) eltéréseket egy megadott küszöb
        felett jelzi.
      </p>
      <p class="muted" style="margin-top:8px;">
        Logika: baseline (cég) vs. üzletkötői mix ágazaton belül, csak darabszám alapján.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Inputs (cards)
# -------------------------------------------------
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown('<div class="card"><div class="card-title">1) Excel feltöltése</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"], label_visibility="collapsed")
    st.markdown('<div class="muted">Tipp: válaszd a “DosszieAdatok282 - eredeti” sheetet.</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">2) Küszöb beállítása</div>', unsafe_allow_html=True)
    threshold_pct = st.number_input(
        "Eltérési küszöb (százalékpont, pp)",
        min_value=0.0, max_value=100.0, value=20.0, step=0.5,
        help="Ha az üzletkötő aránya a baseline-hoz képest +küszöb felett vagy −küszöb alatt tér el, jelölést kap."
    )
    st.markdown(
        f"""
        <div style="margin-top:6px;">
          <span class="badge badge-up">UP ha diff &gt; +{threshold_pct:.1f} pp</span>
          <span class="badge badge-down">DOWN ha diff &lt; −{threshold_pct:.1f} pp</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")

# -------------------------------------------------
# Core count-only logic
# -------------------------------------------------
line_col = "Megnevezés"
insurer_col = "RövidNév"
agent_col = "UkKodja1"

def compute_outliers_count_only(df: pd.DataFrame, threshold_pct: float):
    d = df.copy()
    d["count"] = 1  # each row = one contract

    # 1) baseline per line -> insurer
    base = d.groupby([line_col, insurer_col], dropna=False).agg(
        base_count=("count", "sum")
    ).reset_index()
    base_totals = d.groupby([line_col], dropna=False).agg(
        line_total=("count", "sum")
    ).reset_index()
    base = base.merge(base_totals, on=line_col, how="left")
    base["base_share"] = base["base_count"] / base["line_total"]

    # 2) agent mix per line -> insurer
    agent = d.groupby([agent_col, line_col, insurer_col], dropna=False).agg(
        agent_count=("count", "sum")
    ).reset_index()
    agent_totals = d.groupby([agent_col, line_col], dropna=False).agg(
        agent_line_total=("count", "sum")
    ).reset_index()
    agent = agent.merge(agent_totals, on=[agent_col, line_col], how="left")
    agent["agent_share"] = agent["agent_count"] / agent["agent_line_total"]

    # 3) compare
    out = agent.merge(base, on=[line_col, insurer_col], how="left")
    out["diff_pp"] = (out["agent_share"] - out["base_share"]) * 100

    out["direction"] = np.select(
    [out["diff_pp"] > threshold_pct, out["diff_pp"] < -threshold_pct],
    ["UP", "DOWN"],
    default=None
)

    outliers = out[out["direction"].notna()].copy()

    # presentation columns
    outliers["Company Share %"] = (outliers["base_share"] * 100).round(2)
    outliers["Agent Share %"] = (outliers["agent_share"] * 100).round(2)
    outliers["Difference (pp)"] = outliers["diff_pp"].round(2)

    outliers = outliers.rename(columns={
        agent_col: "Üzletkötő kód",
        line_col: "Ágazat",
        insurer_col: "Biztosító",
        "base_count": "Céges db",
        "line_total": "Céges ágazati db",
        "agent_count": "Üzletkötői db",
        "agent_line_total": "Üzletkötői ágazati db",
        "direction": "Irány"
    })

    # sort by abs diff desc within agent+line
    outliers["abs_diff"] = outliers["Difference (pp)"].abs()
    outliers = outliers.sort_values(
        by=["Üzletkötő kód", "Ágazat", "abs_diff"],
        ascending=[True, True, False]
    ).drop(columns=["abs_diff", "base_share", "agent_share", "diff_pp"])

    return outliers

# Color UP/DOWN in table
def color_diff(val):
    if pd.isna(val):
        return ""
    return "background-color: #eafaf0; color:#166534; font-weight:700;" if val > 0 \
        else "background-color: #fdecec; color:#991b1b; font-weight:700;"

def color_direction(val):
    if val == "UP":
        return "color:#166534; font-weight:800;"
    if val == "DOWN":
        return "color:#991b1b; font-weight:800;"
    return ""

# -------------------------------------------------
# Run + results
# -------------------------------------------------
if not uploaded:
    st.info("Tölts fel egy Excel fájlt az elemzés indításához.")
else:
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Sheet kiválasztása", xls.sheet_names, index=0)
    df = pd.read_excel(uploaded, sheet_name=sheet)

    m1, m2, m3 = st.columns(3)
    m1.metric("Sorok száma", f"{len(df):,}")
    m2.metric("Üzletkötők száma", f"{df[agent_col].nunique(dropna=False):,}")
    m3.metric("Ágazatok száma", f"{df[line_col].nunique(dropna=False):,}")

    st.markdown("")
    if st.button("Elemzés futtatása"):
        outliers = compute_outliers_count_only(df, threshold_pct)

        st.markdown(
            """
            <div class="card" style="margin-top:6px;">
              <div class="card-title">Kiugró eltérések</div>
              <div class="muted">Csak azok a sorok jelennek meg, ahol az eltérés abszolút értéke meghaladja a küszöböt (UP vagy DOWN).</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="dataframe-wrap">', unsafe_allow_html=True)
        styled = (
            outliers.style
              .applymap(color_diff, subset=["Difference (pp)"])
              .applymap(color_direction, subset=["Irány"])
        )
        st.dataframe(styled, use_container_width=True, height=520)
        st.markdown('</div>', unsafe_allow_html=True)

        export_name = f"fbi_outliers_threshold_{threshold_pct:.1f}.xlsx"
        with pd.ExcelWriter(export_name, engine="xlsxwriter") as writer:
            outliers.to_excel(writer, index=False, sheet_name="outliers")

        with open(export_name, "rb") as f:
            st.download_button(
                "Eredmények letöltése (.xlsx)",
                data=f,
                file_name=export_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.markdown(
            """
            <div class="muted" style="margin-top:12px;">
              Megjegyzés: a kiugró eltérés nem automatikusan részrehajlás bizonyítéka; inkább azt jelzi, hogy érdemes kontextus alapján áttekinteni az adott üzletkötő portfólióját.
            </div>
            """,
            unsafe_allow_html=True
        )
