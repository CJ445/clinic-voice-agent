"""Parse nvidia-smi and /proc/meminfo output, compute VRAM headroom, and check result records.

Exists so bench/vram.py stays thin and the numbers it publishes come from tested code.
Pure functions only: callers run the commands and pass the text in, so CI can test this on CPU.
"""

from dataclasses import dataclass

MIN_HEADROOM_MIB = 500  # PLAN M0.3 exit criterion
NOT_AVAILABLE = "[N/A]"
REQUIRED_KEYS = ("commit", "command", "gpu", "driver")  # Gate G4 provenance


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_mib: int
    used_mib: int
    driver: str


def parse_mib(raw: str) -> int | None:
    """Parse one `nounits` MiB field. `[N/A]` becomes None; anything else non-numeric raises."""
    value = raw.strip()
    if value == NOT_AVAILABLE:
        return None
    return int(value)


def parse_gpu_query(text: str) -> GpuInfo:
    """Parse `--query-gpu=name,memory.total,memory.used,driver_version` for exactly one GPU."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one GPU from nvidia-smi, got {len(lines)} lines")
    name, total, used, driver = (field.strip() for field in lines[0].split(","))
    total_mib, used_mib = parse_mib(total), parse_mib(used)
    if total_mib is None or used_mib is None:
        raise ValueError("nvidia-smi did not report GPU memory; cannot compute headroom")
    return GpuInfo(name=name, total_mib=total_mib, used_mib=used_mib, driver=driver)


def parse_compute_apps(text: str) -> dict[int, int | None]:
    """Parse `--query-compute-apps=pid,used_memory` into {pid: MiB}; [N/A] memory becomes None."""
    apps: dict[int, int | None] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        pid, used = line.split(",")
        apps[int(pid)] = parse_mib(used)
    return apps


def parse_meminfo(text: str) -> tuple[int, int]:
    """Return (total_mib, used_mib) from /proc/meminfo, where used = MemTotal - MemAvailable."""
    kib: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            kib[key] = int(rest.split()[0])
    total_kib, available_kib = kib["MemTotal"], kib["MemAvailable"]
    return total_kib // 1024, (total_kib - available_kib) // 1024


def headroom_mib(*, total_mib: int, peak_used_mib: int) -> int:
    return total_mib - peak_used_mib


def fits(headroom: int) -> bool:
    return headroom >= MIN_HEADROOM_MIB


def check_record(record: dict[str, object], *, home: str) -> None:
    """Raise ValueError if provenance is missing or any value leaks the home directory."""
    for key in REQUIRED_KEYS:
        if not record.get(key):
            raise ValueError(f"result record is missing {key!r}")
    if home in repr(record):
        raise ValueError(f"result record contains the home directory ({home}); strip it first")
