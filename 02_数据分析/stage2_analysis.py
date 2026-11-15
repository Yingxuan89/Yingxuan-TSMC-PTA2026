"""
第二阶段：Inline 工艺/量测参数与 EOL 电性参数关联分析
数据集：semiconductor_yield_forecasting_data.csv（1250 片晶圆，50 个批次）

运行：python stage2_analysis.py path/to/semiconductor_yield_forecasting_data.csv
输出：figures/ 下的图表，results/ 下的统计结果表
依赖：pandas, numpy, scipy, scikit-learn, matplotlib
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

CSV = sys.argv[1] if len(sys.argv) > 1 else "semiconductor_yield_forecasting_data.csv"
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); RES = os.path.join(OUT, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

d = pd.read_csv(CSV)
NODES = ["7nm", "10nm", "14nm", "22nm", "28nm"]
COLORS = dict(zip(NODES, ["#1f6feb", "#e8710a", "#2da44e", "#8250df", "#cf222e"]))

INLINE = ["etch_rate", "pressure", "temperature", "exposure_time", "focus_offset",
          "dose", "deposition_rate", "thickness_uniformity", "implant_energy", "tilt_angle",
          "critical_dimension", "oxide_thickness", "resistivity", "defect_count", "defect_density"]
EOL = ["vth", "leakage_current", "resistance", "yield"]
NODE_DEP = ["critical_dimension", "oxide_thickness", "vth"]   # 随节点变化的参数

summary = {}

# ---------- 1. 数据质量 ----------
summary["n_wafers"] = len(d)
summary["n_lots"] = d.lot_id.nunique()
summary["missing"] = int(d.isna().sum().sum())
summary["duplicates"] = int(d.duplicated().sum())
lot_attr = d.groupby("lot_id")[["technology_node", "product_type", "etch_tool", "litho_tool",
                                "deposition_tool", "implant_tool"]].nunique().max()
summary["one_value_per_lot"] = bool((lot_attr == 1).all())
summary["r_uniformity_defect"] = round(d.thickness_uniformity.corr(d.defect_density), 3)

# ---------- 2. 相关性：整体 vs 节点内 ----------
d["log_leakage"] = np.log10(d.leakage_current)
EOL_A = ["vth", "log_leakage", "resistance", "yield"]
centered = d[INLINE + EOL_A] - d.groupby("technology_node")[INLINE + EOL_A].transform("mean")

def corr_table(df):
    rows = []
    for x in INLINE:
        for y in EOL_A:
            r, p = stats.pearsonr(df[x], df[y])
            rows.append((x, y, r, p))
    t = pd.DataFrame(rows, columns=["inline", "eol", "r", "p"])
    t["p_bonf"] = np.minimum(t.p * len(t), 1.0)   # Bonferroni 多重比较校正
    return t

overall = corr_table(d); within = corr_table(centered)
overall.to_csv(os.path.join(RES, "corr_overall.csv"), index=False, float_format="%.4g")
within.to_csv(os.path.join(RES, "corr_within_node.csv"), index=False, float_format="%.4g")

fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), sharey=True)
for ax, t, title in [(axes[0], overall, "整体相关（未区分节点）"), (axes[1], within, "节点内相关（扣除节点均值后）")]:
    M = t.pivot(index="inline", columns="eol", values="r").loc[INLINE, EOL_A]
    P = t.pivot(index="inline", columns="eol", values="p_bonf").loc[INLINE, EOL_A]
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]; star = "*" if P.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.45 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Vt", "log(漏电流)", "电阻", "良率"])
    ax.set_yticks(range(len(INLINE))); ax.set_yticklabels(INLINE, fontsize=9)
    ax.set_title(title, fontsize=12)
fig.colorbar(im, ax=axes, shrink=0.6, label="皮尔逊相关系数 r")
fig.text(0.01, 0.01, "* 表示经 Bonferroni 校正后 p < 0.05", fontsize=9)
fig.savefig(os.path.join(FIG, "fig1_corr_heatmap.png"), bbox_inches="tight"); plt.close(fig)

# ---------- 3. 混杂效应：CD 与良率 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for n in NODES:
    g = d[d.technology_node == n]
    axes[0].scatter(g.critical_dimension, g["yield"], s=8, alpha=0.6, color=COLORS[n], label=n)
axes[0].set_xlabel("critical_dimension"); axes[0].set_ylabel("良率 yield")
axes[0].set_title(f"整体 r = {d.critical_dimension.corr(d['yield']):+.2f}：看似 CD 越大良率越低")
axes[0].legend(title="技术节点", fontsize=8)
wr = {n: d[d.technology_node == n].critical_dimension.corr(d[d.technology_node == n]["yield"]) for n in NODES}
axes[1].bar(NODES, [wr[n] for n in NODES], color=[COLORS[n] for n in NODES])
axes[1].axhline(d.critical_dimension.corr(d["yield"]), ls="--", color="gray")
axes[1].text(4.4, d.critical_dimension.corr(d["yield"]), "整体 r", va="bottom", ha="right", fontsize=9, color="gray")
axes[1].set_ylim(-0.7, 0.1); axes[1].set_ylabel("节点内 r（CD vs 良率）")
axes[1].set_title("分节点后相关明显减弱")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_cd_yield_confounding.png")); plt.close(fig)
summary["cd_yield_overall_r"] = round(d.critical_dimension.corr(d["yield"]), 3)
summary["cd_yield_within_r"] = {n: round(v, 3) for n, v in wr.items()}

# ---------- 4. 关键参数对：节点内斜率 ----------
pairs = [("critical_dimension", "vth", "CD → Vt", 1000, "mV / 单位CD"),
         ("oxide_thickness", "resistance", "氧化层厚度 → 电阻", 1, "Ω / 单位厚度"),
         ("resistivity", "resistance", "电阻率 → 电阻", 1, "Ω / 单位电阻率"),
         ("defect_density", "log_leakage", "缺陷密度 → log(漏电流)", 1, "decade / 单位缺陷密度"),
         ("defect_density", "yield", "缺陷密度 → 良率", 100, "百分点 / 单位缺陷密度"),
         ("dose", "vth", "注入剂量 → Vt", 1000 * 1e13, "mV / 10¹³ cm⁻²")]
slopes = []
for x, y, name, scale, unit in pairs:
    for n in NODES:
        g = d[d.technology_node == n]
        s = stats.linregress(g[x], g[y])
        slopes.append((name, n, s.rvalue, s.slope * scale, unit, s.pvalue))
    s = stats.linregress(centered[x], centered[y])
    slopes.append((name, "节点内合并", s.rvalue, s.slope * scale, unit, s.pvalue))
slopes = pd.DataFrame(slopes, columns=["关系", "节点", "r", "斜率", "单位", "p"])
slopes.to_csv(os.path.join(RES, "slopes_by_node.csv"), index=False, float_format="%.4g")

fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
for ax, (x, y, title) in zip(axes.flat, [("critical_dimension", "vth", "CD 与阈值电压（各节点分别拟合）"),
                                          ("defect_density", "log_leakage", "缺陷密度与漏电流"),
                                          ("defect_density", "yield", "缺陷密度与良率"),
                                          ("resistivity", "resistance", "电阻率与电阻")]):
    for n in NODES:
        g = d[d.technology_node == n]
        ax.scatter(g[x], g[y], s=6, alpha=0.45, color=COLORS[n], label=n)
        s = stats.linregress(g[x], g[y]); xs = np.linspace(g[x].min(), g[x].max(), 10)
        ax.plot(xs, s.intercept + s.slope * xs, color=COLORS[n], lw=1.6)
    ax.set_xlabel(x); ax.set_ylabel({"log_leakage": "log10(漏电流)"}.get(y, y)); ax.set_title(title, fontsize=11)
axes[0, 0].legend(title="节点", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_key_pairs.png")); plt.close(fig)

# ---------- 5. 多元回归：良率 ----------
Xn = pd.get_dummies(d.technology_node, drop_first=True).astype(float)
cols = ["critical_dimension", "oxide_thickness", "defect_density", "resistivity", "dose",
        "focus_offset", "exposure_time", "implant_energy", "tilt_angle", "etch_rate",
        "pressure", "temperature", "deposition_rate"]
# 用节点内标准差标准化，避免 CD 等随节点变化的参数因节点差异被放大
Z = centered[cols] / centered[cols].std()
r2_node = LinearRegression().fit(Xn, d["yield"]).score(Xn, d["yield"])
Xf = pd.concat([Xn, Z], axis=1); mf = LinearRegression().fit(Xf, d["yield"])
r2_full = mf.score(Xf, d["yield"])
lm = d.groupby("lot_id")["yield"].transform("mean")
r2_lot = 1 - ((d["yield"] - lm) ** 2).sum() / ((d["yield"] - d["yield"].mean()) ** 2).sum()
coef = pd.Series(mf.coef_[-len(cols):], index=cols).sort_values(key=abs, ascending=False)
coef.to_csv(os.path.join(RES, "yield_regression_std_coef.csv"), header=["每1σ变化对应良率变化"], float_format="%.4f")
summary.update(r2_node=round(r2_node, 3), r2_node_plus_params=round(r2_full, 3), r2_lot=round(r2_lot, 3),
               yield_coef={k: round(v, 4) for k, v in coef.items()})

fig, ax = plt.subplots(figsize=(7.5, 4.6))
c = coef[::-1] * 100
ax.barh(c.index, c.values, color=["#cf222e" if v < 0 else "#2da44e" for v in c.values])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("参数增加 1σ 时良率变化（百分点，已控制技术节点）")
ax.set_title("良率多元回归：各参数的独立影响")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_yield_regression.png")); plt.close(fig)

# ---------- 6. 机台 / 批次效应 ----------
# 每个批次只用一台机台、一个节点，所以机台比较必须在“批次”层面做，
# 否则同一批次的 25 片会被当成 25 个独立样本，p 值被严重低估（伪重复）。
lot = centered.assign(lot_id=d.lot_id).groupby("lot_id").mean(numeric_only=True)
lot_tools = d.groupby("lot_id")[["etch_tool", "litho_tool", "deposition_tool", "implant_tool", "process_date"]].first()
lot = lot.join(lot_tools)
tool_rows = []
for t in ["etch_tool", "litho_tool", "deposition_tool", "implant_tool"]:
    for y in ["critical_dimension", "oxide_thickness", "defect_density", "vth", "yield"]:
        p_raw = stats.f_oneway(*[g[y] for _, g in d.groupby(t)]).pvalue
        p_adj = stats.f_oneway(*[centered.loc[g.index, y] for _, g in d.groupby(t)]).pvalue
        p_lot = stats.f_oneway(*[g[y] for _, g in lot.groupby(t)]).pvalue
        tool_rows.append((t, y, p_raw, p_adj, p_lot))
tools = pd.DataFrame(tool_rows, columns=["机台", "参数", "p_晶圆级未控制节点", "p_晶圆级控制节点", "p_批次级控制节点"])
summary["tool_min_p_lot_level"] = round(tools["p_批次级控制节点"].min(), 3)

# ---------- 6b. 时间漂移 ----------
drift_cols = ["defect_density", "log_leakage", "yield", "temperature", "etch_rate", "deposition_rate",
              "critical_dimension", "vth", "resistance"]
lot = lot.sort_values("process_date"); day = np.arange(len(lot))
drift = []
for c in drift_cols:
    rho, p = stats.spearmanr(day, lot[c])
    drift.append((c, rho, p, lot[c].iloc[:10].mean(), lot[c].iloc[-10:].mean()))
drift = pd.DataFrame(drift, columns=["参数", "spearman_rho", "p", "前10天均值(相对节点)", "后10天均值(相对节点)"])
drift.to_csv(os.path.join(RES, "time_drift.csv"), index=False, float_format="%.4g")
summary["drift"] = {r["参数"]: [round(r.spearman_rho, 2), float(f"{r.p:.2g}"),
                               round(r["后10天均值(相对节点)"] - r["前10天均值(相对节点)"], 4)] for _, r in drift.iterrows()}

fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True)
for ax, (c, lab) in zip(axes.flat, [("defect_density", "缺陷密度"), ("yield", "良率"),
                                    ("temperature", "工艺温度 temperature"), ("deposition_rate", "沉积速率")]):
    ax.plot(pd.to_datetime(lot.process_date), lot[c], "o-", ms=3, lw=1, color="#1f6feb")
    s = stats.linregress(day, lot[c]); ax.plot(pd.to_datetime(lot.process_date), s.intercept + s.slope * day, "--", color="#cf222e")
    rho = drift.set_index("参数").loc[c, "spearman_rho"]
    ax.set_title(f"{lab}（相对同节点均值）  ρ = {rho:+.2f}", fontsize=11); ax.axhline(0, color="gray", lw=0.6)
fig.autofmt_xdate(); fig.suptitle("按批次的时间趋势（每点 = 1 个批次 / 1 天）", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_time_drift.png")); plt.close(fig)
tools.to_csv(os.path.join(RES, "tool_anova.csv"), index=False, float_format="%.3g")
summary["defect_by_etch_tool_centered"] = centered.groupby(d.etch_tool).defect_density.mean().round(3).to_dict()

# ---------- 7. 异常样本 ----------
def limits(col, g):
    v = np.log10(g[col]) if col == "leakage_current" else g[col]
    m, s = v.mean(), v.std()
    lo, hi = m - 3 * s, m + 3 * s
    return (10 ** lo, 10 ** hi) if col == "leakage_current" else (lo, hi)

flags = []
for col in INLINE + EOL:
    if col == "yield":
        continue
    groups = d.groupby("technology_node") if col in NODE_DEP else [("全部", d)]
    for n, g in groups:
        lo, hi = limits(col, g)
        if col in ("thickness_uniformity", "defect_density", "defect_count", "leakage_current"):
            lo = -np.inf   # 越小越好，只设上限
        bad = g[(g[col] < lo) | (g[col] > hi)]
        for i, r in bad.iterrows():
            flags.append((i, r.lot_id, r.wafer_id, r.technology_node, col, r[col],
                          "偏高" if r[col] > hi else "偏低", lo, hi))
flags = pd.DataFrame(flags, columns=["idx", "lot_id", "wafer_id", "node", "参数", "值", "方向", "下限", "上限"])
flags.to_csv(os.path.join(RES, "outlier_flags.csv"), index=False, float_format="%.4g")

# 节点内良率最低 2.5% 作为良率异常
d["yield_z"] = (d["yield"] - d.groupby("technology_node")["yield"].transform("mean")) / d.groupby("technology_node")["yield"].transform("std")
low_y = d[d.yield_z < d.yield_z.quantile(0.025)]
zc = (centered[INLINE + EOL_A] / centered[INLINE + EOL_A].std())
low_profile = zc.loc[low_y.index].mean().sort_values()
low_profile.to_csv(os.path.join(RES, "low_yield_profile_z.csv"), header=["低良率晶圆平均z值"], float_format="%.3f")
leak_out = flags[flags["参数"] == "leakage_current"].idx.values
leak_profile = zc.loc[leak_out].mean().sort_values(ascending=False)
leak_profile.to_csv(os.path.join(RES, "leakage_outlier_profile_z.csv"), header=["漏电异常晶圆平均z值"], float_format="%.3f")
summary.update(n_flag_rows=len(flags), n_flag_wafers=int(flags.idx.nunique()),
               flags_by_param=flags["参数"].value_counts().to_dict(),
               n_low_yield=len(low_y), n_leak_out=len(leak_out),
               low_yield_top=low_profile.round(2).head(4).to_dict(),
               low_yield_top_pos=low_profile.round(2).tail(4).to_dict(),
               leak_profile_top=leak_profile.round(2).head(5).to_dict(),
               low_yield_lots=low_y.lot_id.value_counts().head(5).to_dict(),
               leak_out_lots=d.loc[leak_out].lot_id.value_counts().head(5).to_dict(),
               leak_out_dd_pct=float((d.loc[leak_out, "defect_density"] > d.defect_density.quantile(0.75)).mean()))

fig, ax = plt.subplots(figsize=(10, 4))
ok = ~d.index.isin(leak_out)
ax.scatter(d.index[ok], d.leakage_current[ok], s=5, color="#8c959f", label="正常")
ax.scatter(leak_out, d.leakage_current[leak_out], s=18, color="#cf222e", label="超上限")
lo, hi = limits("leakage_current", d)
ax.axhline(hi, ls="--", color="#cf222e", lw=1); ax.text(len(d), hi, " 上限（对数 +3σ）", va="bottom", ha="right", fontsize=9, color="#cf222e")
ax.set_yscale("log"); ax.set_xlabel("晶圆序号（按批次排列）"); ax.set_ylabel("漏电流 leakage_current")
ax.set_title(f"漏电流异常晶圆：{len(leak_out)} 片超出管控上限"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_leakage_outliers.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
lp = low_profile.drop("yield")
top = pd.concat([lp.head(2), lp.tail(6)]).sort_values()
ax.barh(top.index, top.values, color=["#cf222e" if v > 0 else "#1f6feb" for v in top.values])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("相对同节点平均水平的偏离（z 值）")
ax.set_title(f"低良率晶圆（节点内最低 2.5%，{len(low_y)} 片）的参数特征")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_low_yield_profile.png")); plt.close(fig)

with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
