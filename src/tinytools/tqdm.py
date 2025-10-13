"""Logger tools."""

from __future__ import annotations

import tqdm as tqdm_og
from rich.progress import ProgressColumn, Task, Text, filesize
from tqdm import rich as tqdm_rich

tqdm_og.tqdm = tqdm_rich.tqdm


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit: str = "", unit_scale: bool = False, unit_divisor: int = 1000) -> None:
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"? {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed, ["", "K", "M", "G", "T", "P", "E", "Z", "Y"], self.unit_divisor
            )
            precision = 0 if unit == 1 else 1
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
            precision = 2
        return (
            Text(f"{speed / unit:,.{precision}f} {suffix}{self.unit}/s", style="progress.data.speed")
            if speed / unit >= 1
            else Text(f"{unit / speed:,.{precision}f} {suffix}s/{self.unit}", style="progress.data.speed")
        )


def setup_prettier_tqdm() -> None:
    """Set tqdm to use rich.

    Fixes tqdm.rich rate column to support s/it when it/s < 1. Also, sets rate column floating point precision to 1.

    """
    tqdm_og.tqdm = tqdm_rich.tqdm
    tqdm_rich.RateColumn = RateColumn
