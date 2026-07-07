from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Write experiment configuration, metrics, and lightweight SVG loss curves."""

    fieldnames = [
        "step",
        "elapsed_seconds",
        "tokens_seen",
        "train_loss",
        "val_loss",
        "learning_rate",
    ]

    def __init__(self, log_dir: str | Path, run_name: str | None, config: Any):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name or timestamp
        self.run_dir = Path(log_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.json"
        self.step_curve_path = self.run_dir / "loss_curves_steps.svg"
        self.time_curve_path = self.run_dir / "loss_curves_time.svg"
        self.start_time = time.perf_counter()
        self.rows: list[dict[str, float | int | None]] = []

        config_data = (
            asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
        )
        config_data["run_name"] = self.run_name
        config_data["started_at"] = datetime.now().isoformat(timespec="seconds")
        with open(self.config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_data, config_file, indent=2, default=str)

        self._register_run(config_data)

        with open(self.metrics_path, "w", encoding="utf-8", newline="") as metrics_file:
            csv.DictWriter(metrics_file, fieldnames=self.fieldnames).writeheader()

    def _register_run(self, config_data: dict[str, Any]) -> None:
        index_path = self.run_dir.parent / "experiment_log.csv"
        fieldnames = [
            "run_name",
            "started_at",
            "config_path",
            "metrics_path",
            "configuration",
        ]
        write_header = not index_path.exists()
        row = {
            "run_name": self.run_name,
            "started_at": config_data["started_at"],
            "config_path": str(self.config_path),
            "metrics_path": str(self.metrics_path),
            "configuration": json.dumps(config_data, default=str, sort_keys=True),
        }
        with open(index_path, "a", encoding="utf-8", newline="") as index_file:
            writer = csv.DictWriter(index_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def log(
        self,
        step: int,
        tokens_seen: int,
        learning_rate: float,
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        row: dict[str, float | int | None] = {
            "step": step,
            "elapsed_seconds": self.elapsed_seconds(),
            "tokens_seen": tokens_seen,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
        }
        self.rows.append(row)
        with open(self.metrics_path, "a", encoding="utf-8", newline="") as metrics_file:
            csv.DictWriter(metrics_file, fieldnames=self.fieldnames).writerow(row)
        self._write_loss_curves(self.step_curve_path, "step", "Gradient step")
        self._write_loss_curves(
            self.time_curve_path, "elapsed_seconds", "Wall-clock time (seconds)"
        )

    def _write_loss_curves(self, output_path: Path, x_key: str, x_label: str) -> None:
        series = {
            "train": [
                (float(row[x_key]), float(row["train_loss"]))
                for row in self.rows
                if row["train_loss"] is not None
            ],
            "validation": [
                (float(row[x_key]), float(row["val_loss"]))
                for row in self.rows
                if row["val_loss"] is not None
            ],
        }
        points = series["train"] + series["validation"]
        if not points:
            return

        width, height = 900, 520
        left, right, top, bottom = 80, 30, 40, 70
        plot_width = width - left - right
        plot_height = height - top - bottom
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if x_min == x_max:
            x_max = x_min + 1
        if y_min == y_max:
            y_max = y_min + 1

        def project(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            px = left + (x - x_min) / (x_max - x_min) * plot_width
            py = top + (y_max - y) / (y_max - y_min) * plot_height
            return px, py

        colors = {"train": "#2563eb", "validation": "#dc2626"}
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827"/>',
            f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="16">{x_label}</text>',
            f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="sans-serif" font-size="16">Loss</text>',
            f'<text x="{left}" y="24" font-family="sans-serif" font-size="18" font-weight="bold">Loss curves</text>',
        ]
        for index in range(6):
            fraction = index / 5
            x_value = x_min + fraction * (x_max - x_min)
            x = left + fraction * plot_width
            y_value = y_max - fraction * (y_max - y_min)
            y = top + fraction * plot_height
            svg.append(
                f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.0f}</text>'
            )
            svg.append(
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.3g}</text>'
            )
        for name, values in series.items():
            if not values:
                continue
            path_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(project, values))
            svg.append(
                f'<polyline points="{path_points}" fill="none" stroke="{colors[name]}" stroke-width="2"/>'
            )
        svg.extend(
            [
                f'<line x1="{width - 210}" y1="24" x2="{width - 180}" y2="24" stroke="{colors["train"]}" stroke-width="3"/>',
                f'<text x="{width - 172}" y="29" font-family="sans-serif" font-size="13">train</text>',
                f'<line x1="{width - 110}" y1="24" x2="{width - 80}" y2="24" stroke="{colors["validation"]}" stroke-width="3"/>',
                f'<text x="{width - 72}" y="29" font-family="sans-serif" font-size="13">validation</text>',
                "</svg>",
            ]
        )
        output_path.write_text("\n".join(svg), encoding="utf-8")
