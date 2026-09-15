"""Matplotlib implementations for the v1 tidy-CSV plot catalog."""

from __future__ import annotations

from typing import Any

from pheno_rwe.plots.base import PlotInputError

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def _modules() -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


def _require(frame: Any, columns: list[str] | tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise PlotInputError(f"plot CSV is missing required columns: {', '.join(missing)}")


def _finish(fig: Any, ax: Any, params: dict[str, Any]) -> Any:
    title = params.get("title")
    if title:
        ax.set_title(str(title), loc="left", fontweight="bold")
    subtitle = params.get("subtitle")
    if subtitle:
        ax.text(
            0,
            1.01,
            str(subtitle),
            transform=ax.transAxes,
            va="bottom",
            fontsize="small",
            color="#555555",
        )
    fig.tight_layout()
    return fig


class KmCurveRenderer:
    kind = "km_curve"
    required_columns = ("group", "time", "survival")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        np, plt = _modules()
        _require(frame, self.required_columns)
        show_at_risk = bool(params.get("show_at_risk", True)) and "at_risk" in frame
        if show_at_risk:
            fig, (ax, risk_ax) = plt.subplots(
                2, 1, figsize=(7.2, 5.2), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
            )
        else:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            risk_ax = None
        groups = sorted(frame["group"].dropna().unique().tolist(), key=str)
        if not groups:
            raise PlotInputError("KM CSV contains no groups")
        for index, group in enumerate(groups):
            data = frame.loc[frame["group"] == group].sort_values("time", kind="mergesort")
            color = PALETTE[index % len(PALETTE)]
            ax.step(data["time"], data["survival"], where="post", label=str(group), color=color)
            if {"lower", "upper"}.issubset(frame.columns):
                ax.fill_between(
                    data["time"].to_numpy(dtype=float),
                    data["lower"].to_numpy(dtype=float),
                    data["upper"].to_numpy(dtype=float),
                    step="post",
                    alpha=0.14,
                    color=color,
                    linewidth=0,
                )
        ax.set(xlabel="Time", ylabel="Survival probability", ylim=(0, 1.02))
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        if risk_ax is not None:
            maximum = float(frame["time"].max())
            ticks = np.linspace(0, maximum, 5) if maximum > 0 else np.asarray([0.0])
            risk_ax.set_ylim(-0.5, len(groups) - 0.5)
            risk_ax.set_yticks(range(len(groups)), [str(group) for group in groups])
            risk_ax.set_ylabel("At risk", rotation=0, ha="right", va="center")
            risk_ax.set_xticks(ticks)
            for row_index, group in enumerate(groups):
                data = frame.loc[frame["group"] == group].sort_values("time", kind="mergesort")
                for tick in ticks:
                    eligible = data.loc[data["time"] <= tick]
                    at_risk = (
                        int(eligible.iloc[-1]["at_risk"])
                        if len(eligible)
                        else int(data.iloc[0]["at_risk"])
                    )
                    risk_ax.text(
                        tick, row_index, str(at_risk), ha="center", va="center", fontsize="small"
                    )
            risk_ax.spines[["left", "right", "top"]].set_visible(False)
            risk_ax.tick_params(axis="y", length=0)
            risk_ax.grid(False)
        return _finish(fig, ax, params)


class ForestRenderer:
    kind = "forest"
    required_columns = ("estimate", "lower", "upper")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        np, plt = _modules()
        _require(frame, self.required_columns)
        label_column = params.get("label_column")
        if not label_column:
            label_column = next(
                (
                    column
                    for column in ("label", "outcome", "term", "estimand", "variable")
                    if column in frame
                ),
                None,
            )
        if not label_column:
            raise PlotInputError(
                "forest CSV needs a label, outcome, term, estimand, or variable column"
            )
        data = frame.dropna(subset=["estimate", "lower", "upper"]).copy()
        if data.empty:
            raise PlotInputError("forest CSV has no complete confidence intervals")
        data = data.reset_index(drop=True)
        y = np.arange(len(data))
        estimates = data["estimate"].to_numpy(dtype=float)
        lower = data["lower"].to_numpy(dtype=float)
        upper = data["upper"].to_numpy(dtype=float)
        if ((estimates - lower) < 0).any() or ((upper - estimates) < 0).any():
            raise PlotInputError("forest confidence intervals must contain their estimates")
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.38 * len(data) + 1.8)))
        ax.errorbar(
            estimates,
            y,
            xerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o",
            color=PALETTE[0],
            ecolor=PALETTE[0],
            capsize=3,
        )
        ratio = bool(params.get("log_scale", False)) or (
            "estimand" in data
            and data["estimand"]
            .astype(str)
            .str.contains("ratio|hazard|odds", case=False, regex=True)
            .any()
        )
        reference = float(params.get("reference", 1.0 if ratio else 0.0))
        if ratio:
            if (lower <= 0).any():
                raise PlotInputError("log-scale forest intervals must be positive")
            ax.set_xscale("log")
        ax.axvline(reference, color="#666666", linestyle="--", linewidth=1)
        ax.set_yticks(y, data[label_column].astype(str))
        ax.invert_yaxis()
        ax.set_xlabel(str(params.get("xlabel", "Effect estimate (95% CI)")))
        ax.grid(axis="x", alpha=0.2)
        return _finish(fig, ax, params)


class LovePlotRenderer:
    kind = "love_plot"
    required_columns = ("covariate",)

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        np, plt = _modules()
        _require(frame, self.required_columns)
        if {"smd_before", "smd_after"}.issubset(frame.columns):
            wide = frame[["covariate", "smd_before", "smd_after"]].copy()
        elif {"phase", "smd"}.issubset(frame.columns):
            wide = frame.pivot(index="covariate", columns="phase", values="smd").reset_index()
            before = next(
                (
                    column
                    for column in wide
                    if str(column).lower() in {"before", "unadjusted", "pre"}
                ),
                None,
            )
            after = next(
                (column for column in wide if str(column).lower() in {"after", "adjusted", "post"}),
                None,
            )
            if before is None or after is None:
                raise PlotInputError("long love-plot CSV must include before and after phases")
            wide = wide.rename(columns={before: "smd_before", after: "smd_after"})
        else:
            raise PlotInputError("love_plot needs smd_before/smd_after or phase/smd columns")
        wide = wide.dropna(subset=["smd_before", "smd_after"]).copy()
        wide["max_abs"] = wide[["smd_before", "smd_after"]].abs().max(axis=1)
        wide = wide.sort_values(["max_abs", "covariate"], ascending=[True, True], kind="mergesort")
        y = np.arange(len(wide))
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.35 * len(wide) + 1.8)))
        ax.scatter(wide["smd_before"].abs(), y, label="Before", color=PALETTE[1], marker="o")
        ax.scatter(wide["smd_after"].abs(), y, label="After", color=PALETTE[0], marker="D")
        threshold = float(params.get("threshold", 0.1))
        ax.axvline(
            threshold,
            color="#666666",
            linestyle="--",
            linewidth=1,
            label=f"Threshold ({threshold:g})",
        )
        ax.set_yticks(y, wide["covariate"].astype(str))
        ax.set_xlabel("Absolute standardized mean difference")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.2)
        return _finish(fig, ax, params)


class UmapScatterRenderer:
    kind = "umap_scatter"
    required_columns = ("umap_1", "umap_2")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        _, plt = _modules()
        _require(frame, self.required_columns)
        color_by = str(params.get("color_by") or ("cluster" if "cluster" in frame else ""))
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        if color_by and color_by in frame:
            groups = sorted(frame[color_by].dropna().unique().tolist(), key=str)
            for index, group in enumerate(groups):
                data = frame.loc[frame[color_by] == group]
                ax.scatter(
                    data["umap_1"],
                    data["umap_2"],
                    s=28,
                    alpha=0.82,
                    label=str(group),
                    color=PALETTE[index % len(PALETTE)],
                )
            ax.legend(title=color_by, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.scatter(frame["umap_1"], frame["umap_2"], s=28, alpha=0.82, color=PALETTE[0])
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
        ax.grid(alpha=0.15)
        return _finish(fig, ax, params)


class TrajectoryRenderer:
    kind = "trajectory"
    required_columns = ("time", "value")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        _, plt = _modules()
        _require(frame, self.required_columns)
        group_column = "group" if "group" in frame else None
        groups = (
            sorted(frame[group_column].dropna().unique().tolist(), key=str)
            if group_column
            else ["Overall"]
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        show_individuals = bool(params.get("show_individuals", True))
        raw = frame.loc[frame["row_type"].eq("individual")] if "row_type" in frame else frame
        summary = frame.loc[frame["row_type"].eq("summary")] if "row_type" in frame else None
        for index, group in enumerate(groups):
            color = PALETTE[index % len(PALETTE)]
            data = raw.loc[raw[group_column] == group] if group_column else raw
            if show_individuals and "person_id" in data:
                for _, patient in data.groupby("person_id", sort=True):
                    patient = patient.sort_values("time", kind="mergesort")
                    ax.plot(
                        patient["time"], patient["value"], color=color, alpha=0.12, linewidth=0.7
                    )
            if summary is not None and len(summary):
                average = summary.loc[summary[group_column] == group] if group_column else summary
                average = average.sort_values("time", kind="mergesort")
            else:
                average = data.groupby("time", sort=True)["value"].mean().reset_index()
            ax.plot(average["time"], average["value"], color=color, label=str(group), linewidth=2)
            if {"lower", "upper"}.issubset(average.columns):
                complete = average.dropna(subset=["lower", "upper"])
                ax.fill_between(
                    complete["time"],
                    complete["lower"],
                    complete["upper"],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
        ax.set(
            xlabel=str(params.get("xlabel", "Time relative to index")),
            ylabel=str(params.get("ylabel", "Measurement")),
        )
        if len(groups) > 1:
            ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        return _finish(fig, ax, params)


class AttritionRenderer:
    kind = "attrition"
    required_columns = ("stage", "n")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        _, plt = _modules()
        _require(frame, self.required_columns)
        data = frame.dropna(subset=["stage", "n"]).reset_index(drop=True)
        if data.empty:
            raise PlotInputError("attrition CSV contains no stages")
        fig, ax = plt.subplots(figsize=(7.2, max(3.5, len(data) * 1.1)))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(data) - 0.5)
        for index, row in data.iterrows():
            y = len(data) - index - 1
            label = f"{row['stage']}\nN = {int(row['n']):,}"
            if "excluded" in row and row["excluded"] == row["excluded"]:
                label += f"  (excluded {int(row['excluded']):,})"
            ax.text(
                0.5,
                y,
                label,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": PALETTE[0]},
            )
            if index < len(data) - 1:
                ax.annotate(
                    "",
                    xy=(0.5, y - 0.72),
                    xytext=(0.5, y - 0.28),
                    arrowprops={"arrowstyle": "->", "color": "#666666"},
                )
        ax.axis("off")
        return _finish(fig, ax, params)


class MappingCoverageRenderer:
    kind = "mapping_coverage"
    required_columns = ("domain", "mapping_status")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        np, plt = _modules()
        _require(frame, self.required_columns)
        value_column = "count" if "count" in frame else ("rows" if "rows" in frame else None)
        if value_column is None:
            data = (
                frame.groupby(["domain", "mapping_status"], sort=True)
                .size()
                .rename("count")
                .reset_index()
            )
            value_column = "count"
        else:
            data = (
                frame.groupby(["domain", "mapping_status"], sort=True)[value_column]
                .sum()
                .reset_index()
            )
        if params.get("domain"):
            data = data.loc[data["domain"].astype(str) == str(params["domain"])]
        pivot = (
            data.pivot(index="domain", columns="mapping_status", values=value_column)
            .fillna(0)
            .sort_index()
        )
        preferred = ["ALREADY_STANDARD", "MAPPED", "UNCHECKED", "UNMAPPED"]
        columns = [column for column in preferred if column in pivot] + [
            column for column in pivot if column not in preferred
        ]
        pivot = pivot[columns]
        totals = pivot.sum(axis=1).replace(0, np.nan)
        fractions = pivot.div(totals, axis=0)
        fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.45 * len(pivot) + 1.8)))
        left = np.zeros(len(fractions))
        for index, status in enumerate(columns):
            values = fractions[status].to_numpy(dtype=float)
            ax.barh(
                fractions.index.astype(str),
                values,
                left=left,
                label=str(status),
                color=PALETTE[index % len(PALETTE)],
            )
            left += np.nan_to_num(values)
        ax.set(xlabel="Fraction of mapped rows", xlim=(0, 1))
        ax.legend(
            frameon=False,
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            ncol=min(4, len(columns)),
        )
        ax.grid(axis="x", alpha=0.2)
        return _finish(fig, ax, params)


class IncidenceBarRenderer:
    kind = "incidence_bar"
    required_columns = ("group", "rate")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        np, plt = _modules()
        _require(frame, self.required_columns)
        data = frame.sort_values("group", kind="mergesort").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        positions = np.arange(len(data))
        rates = data["rate"].to_numpy(dtype=float)
        kwargs: dict[str, Any] = {}
        if {"lower", "upper"}.issubset(data.columns):
            lower = data["lower"].to_numpy(dtype=float)
            upper = data["upper"].to_numpy(dtype=float)
            kwargs["yerr"] = np.vstack([rates - lower, upper - rates])
            kwargs["capsize"] = 4
        ax.bar(
            positions, rates, color=[PALETTE[index % len(PALETTE)] for index in positions], **kwargs
        )
        ax.set_xticks(positions, data["group"].astype(str))
        scale = data["rate_scale"].iloc[0] if "rate_scale" in data else None
        ylabel = f"Incidence rate per {scale:g} person-time units" if scale else "Incidence rate"
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        return _finish(fig, ax, params)


class PathwayBarsRenderer:
    kind = "pathway_bars"
    required_columns = ("sequence", "count")

    def draw(self, frame: Any, params: dict[str, Any]) -> Any:
        _, plt = _modules()
        _require(frame, self.required_columns)
        maximum = int(params.get("max_pathways", 20))
        data = frame.sort_values(
            ["count", "sequence"], ascending=[False, True], kind="mergesort"
        ).head(maximum)
        data = data.sort_values(["count", "sequence"], ascending=[True, False], kind="mergesort")
        fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.38 * len(data) + 1.8)))
        ax.barh(data["sequence"].astype(str), data["count"], color=PALETTE[0])
        ax.set_xlabel("Patients")
        ax.grid(axis="x", alpha=0.2)
        return _finish(fig, ax, params)


BUILTIN_RENDERERS = (
    KmCurveRenderer(),
    ForestRenderer(),
    LovePlotRenderer(),
    UmapScatterRenderer(),
    TrajectoryRenderer(),
    AttritionRenderer(),
    MappingCoverageRenderer(),
    IncidenceBarRenderer(),
    PathwayBarsRenderer(),
)
