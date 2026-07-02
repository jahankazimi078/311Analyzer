"""Export deploy parquet artifacts into docs/data.js for the static GitHub Pages site.

Run after data/build_deploy_artifacts.py:
    ./.venv/bin/python data/build_docs_data.py
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = str(ROOT / "data" / "deploy" / "analytics" / "requests_2025_2026_")
OUT = str(ROOT / "docs" / "data.js")


def load(name):
    return pd.read_parquet(BASE + name + ".parquet")


def clean(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    return v


def recs(df, cols=None):
    if cols:
        df = df[cols]
    out = []
    for row in df.to_dict("records"):
        out.append({k: clean(v) for k, v in row.items()})
    return out


data = {}

# ── headline / quality ──
q = load("dashboard_eda_quality").iloc[0]
data["quality"] = {
    "complaints": int(q.complaints),
    "resolved_share": round(float(q.resolved_with_valid_date_share), 4),
    "median_resolution_days": round(float(q.median_resolution_days), 4),
    "max_created_date": str(q.max_created_date)[:10],
}

# ── EDA ──
data["borough_counts"] = recs(load("dashboard_eda_borough_counts"))
mo = load("dashboard_eda_monthly")
mo["month_label"] = mo["month_label"].astype(str)
data["monthly"] = recs(mo, ["month_label", "year_label", "complaints"])
data["hourly"] = recs(load("dashboard_eda_hourly"))
wh = load("dashboard_eda_weekday_hour")
data["weekday_hour"] = recs(wh, ["created_weekday", "created_hour", "complaints"])
data["seasonal"] = recs(load("dashboard_eda_seasonal"))
data["top_complaint_types"] = recs(load("dashboard_eda_top_complaint_types"))
data["resolution_buckets"] = recs(load("dashboard_eda_resolution_bucket_counts"))
mix = load("dashboard_eda_borough_mix")
data["borough_mix"] = recs(mix, ["borough", "complaint_type", "complaints"])

sr = load("dashboard_eda_sample_rows")
sr["created_date"] = sr["created_date"].astype(str).str[:16]
data["sample_rows"] = recs(sr, ["created_date", "borough", "complaint_type", "descriptor", "status"])

# ── complaint types w/ resolution behaviour ──
ct = load("complaint_type_metrics").sort_values("complaints", ascending=False).head(15)
data["complaint_type_metrics"] = recs(
    ct,
    ["complaint_type", "complaints", "median_resolution_days", "p90_resolution_days",
     "resolved_share", "top_agency"],
)

# ── agencies ──
ag = load("agency_metrics").sort_values("complaints", ascending=False)
data["agency_metrics"] = recs(
    ag,
    ["agency", "agency_name", "complaints", "median_resolution_days",
     "p90_resolution_days", "resolved_share", "unresolved_share", "top_complaint_type"],
)

# ── NLP ──
data["issue_families"] = recs(load("dashboard_nlp_issue_families"))
data["outcome_groups"] = recs(load("dashboard_nlp_outcome_groups"))
nlp = load("dashboard_nlp_overall").iloc[0]
data["nlp_overall"] = {
    "modeled_share": round(float(nlp.modeled_share), 4),
    "resolution_text_share": round(float(nlp.resolution_text_share), 4),
    "high_confidence_subtype_share": round(float(nlp.high_confidence_subtype_share), 4),
}
data["modeled_subtypes"] = recs(load("dashboard_nlp_modeled_subtypes"))

# ── geography: complaint-built map of NYC ──
gp = load("grid_persistence")
gp["lon"] = (gp.grid_lon_min + gp.grid_lon_max) / 2
gp["lat"] = (gp.grid_lat_min + gp.grid_lat_max) / 2
data["grid"] = recs(
    gp,
    ["lon", "lat", "total_complaints", "hotspot_month_share", "unresolved_share"],
)

cb = load("community_board_metrics").sort_values("complaints", ascending=False).head(10)
data["top_boards"] = recs(
    cb,
    ["community_board", "community_board_borough", "complaints",
     "top_complaint_type", "median_resolution_days", "unresolved_share"],
)

# ── fairness ──
z = load("zcta_fairness_metrics")
z = z[z.population.notna() & (z.population > 500) & z.median_household_income.notna()]
data["zcta"] = recs(
    z,
    ["zcta", "top_borough", "population", "median_household_income", "poverty_share",
     "complaints_per_10k", "unresolved_per_10k", "median_resolution_days"],
)
zq = z.groupby("income_quintile", observed=True).agg(
    complaints_per_10k=("complaints_per_10k", "median"),
    median_resolution_days=("median_resolution_days", "median"),
    unresolved_per_10k=("unresolved_per_10k", "median"),
    zctas=("zcta", "count"),
).reset_index()
data["income_quintiles"] = recs(zq)

# ── predictive ──
mm = load("resolution_bucket_model_metrics")
acc = mm[(mm.metric == "accuracy") & (mm.metric_scope == "overall")]
data["model_accuracy"] = recs(
    acc.drop_duplicates(["model_name", "feature_set"]),
    ["model_name", "feature_set", "metric_value"],
)
f1 = mm[(mm.metric == "f1") & (mm.metric_scope == "class")
        & (mm.model_name == "hist_gradient_boosting") & (mm.feature_set == "post_routing")]
data["model_f1_per_class"] = recs(f1, ["target_class", "metric_value"])

cmx = load("resolution_bucket_confusion_matrix")
cmx = cmx[(cmx.model_name == "hist_gradient_boosting") & (cmx.feature_set == "post_routing")]
data["confusion"] = recs(
    cmx,
    ["actual_resolution_bucket", "predicted_resolution_bucket", "complaints",
     "actual_bucket_share"],
)

fi = load("resolution_bucket_feature_importance")
fi = fi[(fi.model_name == "multinomial_logistic") & (fi.feature_set == "post_routing")]
fig = fi.groupby("feature_group").agg(
    importance=("mean_abs_coefficient", "sum")).reset_index().sort_values(
    "importance", ascending=False).head(8)
data["feature_groups"] = recs(fig)

js = "window.DATA = " + json.dumps(data, separators=(",", ":")) + ";"
with open(OUT, "w") as f:
    f.write(js)
print(f"wrote {OUT}: {len(js)/1024:.0f} KB")
for k, v in data.items():
    n = len(v) if isinstance(v, list) else 1
    print(f"  {k}: {n}")
