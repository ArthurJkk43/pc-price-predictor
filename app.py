# app.py
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

from component_schema import COMPONENTS

app = Flask(__name__)

MODEL_CACHE = {}

# =============================================================================
# Custom classes for joblib load (MUST exist in __main__ for unpickle)
# =============================================================================
class Gpu180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class Cpu180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class Ssd180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class Ram180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class Mobo180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class Monitor180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class Psu180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class Case180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class CaseFan180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


class CpuCooler180dForecastPipeline:
    """future = price_now * exp(model.predict(X))"""
    def __init__(self, model=None, feature_cols=None, cat_cols=None, price_col="price_now"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.cat_cols = cat_cols or []
        self.price_col = price_col

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            X[c] = X[c].astype(str).fillna("UNKNOWN")
        for c in self.feature_cols:
            if c not in self.cat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        base_price = pd.to_numeric(X[self.price_col], errors="coerce").to_numpy(float)
        log_ratio = self.model.predict(X[self.feature_cols]).astype(float)
        return base_price * np.exp(log_ratio)


# =============================================================================
# Defaults caches (for dropdown + auto-fill)
# =============================================================================
GPU_DEFAULTS_CSV = os.path.join("data", "gpu_chipset_defaults_ui.csv")
GPU_TRAIN_UI_PARQUET = "gpu_keepa_weekly_180d_train_fresh_ui.parquet"
GPU_DEFAULTS_CACHE = None
GPU_DEFAULTS_BY_CHIPSET = {}

CPU_DEFAULTS_CSV = os.path.join("data", "cpu_chipset_defaults.csv")
CPU_DEFAULTS_CACHE = None
CPU_DEFAULTS_BY_NAME = {}

SSD_DEFAULTS_CSV = os.path.join("data", "ssd_defaults.csv")
SSD_DEFAULTS_CACHE = None
SSD_DEFAULTS_BY_KEY = {}

RAM_DEFAULTS_CSV = os.path.join("data", "ram_defaults.csv")
RAM_DEFAULTS_CACHE = None
RAM_DEFAULTS_BY_KEY = {}

MOBO_DEFAULTS_CSV = os.path.join("data", "mobo_defaults.csv")
MOBO_DEFAULTS_CACHE = None
MOBO_DEFAULTS_BY_KEY = {}

MONITOR_DEFAULTS_CSV = os.path.join("data", "monitor_defaults.csv")
MONITOR_DEFAULTS_CACHE = None
MONITOR_DEFAULTS_BY_KEY = {}

PSU_DEFAULTS_CSV = os.path.join("data", "psu_defaults.csv")
PSU_DEFAULTS_CACHE = None
PSU_DEFAULTS_BY_KEY = {}

CASE_DEFAULTS_CSV = os.path.join("data", "case_defaults.csv")
CASE_DEFAULTS_CACHE = None
CASE_DEFAULTS_BY_KEY = {}

CASE_FAN_DEFAULTS_CSV = os.path.join("data", "case_fan_defaults.csv")
CASE_FAN_DEFAULTS_CACHE = None
CASE_FAN_DEFAULTS_BY_KEY = {}

CPU_COOLER_DEFAULTS_CSV = os.path.join("data", "cpu_cooler_defaults.csv")
CPU_COOLER_DEFAULTS_CACHE = None
CPU_COOLER_DEFAULTS_BY_KEY = {}

AIB_BRANDS = [
    "ASUS", "MSI", "GIGABYTE", "ZOTAC", "SAPPHIRE", "POWERCOLOR", "XFX", "ASROCK",
    "PNY", "PALIT", "GAINWARD", "INNO3D", "COLORFUL", "GALAX", "KFA2", "SPARKLE",
    "BIOSTAR", "MAXSUN", "YESTON", "LEADTEK", "EVGA", "NVIDIA", "AMD", "INTEL", "OTHER"
]


# =============================================================================
# Helpers
# =============================================================================
def is_component_cfg(cfg):
    return isinstance(cfg, dict) and "model_path" in cfg


def list_components():
    return [k for k, v in COMPONENTS.items() if is_component_cfg(v)]


def load_model(model_path):
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    obj = joblib.load(model_path)
    MODEL_CACHE[model_path] = obj
    return obj


def unwrap_model(obj):
    """
    Supports:
    - dict bundle: {'pipeline': ..., 'feature_cols': ..., 'metrics': ..., optional 'log_target': bool}
    - pipeline directly
    Returns (model, log_target_flag)
    """
    if isinstance(obj, dict):
        model = obj.get("pipeline") or obj.get("model") or obj.get("estimator")
        if model is None:
            raise ValueError("Loaded model dict does not contain 'pipeline'/'model'/'estimator'.")

        log_target = bool(obj.get("log_target", True))

        # Ratio pipelines return PRICE already -> don't expm1
        cls_name = model.__class__.__name__
        if cls_name in (
            "Gpu180dForecastPipeline",
            "Cpu180dForecastPipeline",
            "Ssd180dForecastPipeline",
            "Ram180dForecastPipeline",
            "Mobo180dForecastPipeline",
            "Monitor180dForecastPipeline",
            "Psu180dForecastPipeline",
            "Case180dForecastPipeline",
            "CaseFan180dForecastPipeline",
            "CpuCooler180dForecastPipeline",
            "RatioToPriceWrapper",
        ):
            log_target = False

        return model, log_target

    return obj, True


def parse_float(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_int01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    return 1 if s in ("1", "true", "yes", "on") else 0


def build_input_row(comp_key, cfg, form):
    num_cols = cfg.get("num_cols", [])
    cat_cols = cfg.get("cat_cols", [])
    row = {}

    for c in num_cols:
        raw = form.get(f"{comp_key}__{c}")
        if (
            "has" in c.lower()
            or c.lower().startswith("is_")
            or c.lower().startswith("support")
            or c.lower().endswith("_flag")
            or c.lower() in ("pwm",)
        ):
            row[c] = parse_int01(raw) if raw is not None else 0
        else:
            row[c] = parse_float(raw)

    for c in cat_cols:
        raw = form.get(f"{comp_key}__{c}")
        s = "" if raw is None else str(raw).strip()
        row[c] = s if s != "" else None

    return pd.DataFrame([row], columns=(num_cols + cat_cols))


def _mode(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty:
        return None
    return s.value_counts().idxmax()


def _align_to_feature_cols(model_obj, X: pd.DataFrame) -> pd.DataFrame:
    if isinstance(model_obj, dict) and "feature_cols" in model_obj:
        cols = model_obj["feature_cols"]
        for c in cols:
            if c not in X.columns:
                X[c] = np.nan
        return X[cols]
    return X


def _predict_price(model, log_target: bool, X: pd.DataFrame) -> float:
    y = float(model.predict(X)[0])
    if log_target:
        y = float(np.expm1(y))
    if not np.isfinite(y) or y < 0:
        y = 0.0
    return y


# =============================================================================
# Defaults loaders
# =============================================================================
def build_gpu_chipset_defaults_from_train_ui():
    if not os.path.exists(GPU_TRAIN_UI_PARQUET):
        raise FileNotFoundError(f"Missing {GPU_TRAIN_UI_PARQUET} in your ml folder.")

    df = pd.read_parquet(GPU_TRAIN_UI_PARQUET)
    need = [
        "chipset",
        "gpu_memory_gb",
        "gpu_core_clock_mhz",
        "gpu_boost_clock_mhz",
        "gpu_release_year",
        "gpu_bus_width_bits",
        "memory_bandwidth_proxy",
        "gpu_memory_type",
        "gpu_architecture",
        "price_now",
        "ma_4w",
        "ma_12w",
        "vol_12w",
        "mom_4w",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"GPU train UI parquet missing columns: {missing}")

    g = df.groupby("chipset", dropna=True)
    out = g.agg(
        price_now_default=("price_now", "median"),
        ma_4w_default=("ma_4w", "median"),
        ma_12w_default=("ma_12w", "median"),
        vol_12w_default=("vol_12w", "median"),
        mom_4w_default=("mom_4w", "median"),
        gpu_memory_gb=("gpu_memory_gb", "median"),
        gpu_core_clock_mhz=("gpu_core_clock_mhz", "median"),
        gpu_boost_clock_mhz=("gpu_boost_clock_mhz", "median"),
        gpu_release_year=("gpu_release_year", "median"),
        gpu_bus_width_bits=("gpu_bus_width_bits", "median"),
        memory_bandwidth_proxy=("memory_bandwidth_proxy", "median"),
    ).reset_index()

    mem_type = g["gpu_memory_type"].apply(_mode).reset_index(name="gpu_memory_type")
    arch = g["gpu_architecture"].apply(_mode).reset_index(name="gpu_architecture")
    out = out.merge(mem_type, on="chipset", how="left").merge(arch, on="chipset", how="left")
    out["clock_delta_default"] = out["gpu_boost_clock_mhz"] - out["gpu_core_clock_mhz"]

    os.makedirs(os.path.dirname(GPU_DEFAULTS_CSV), exist_ok=True)
    out.to_csv(GPU_DEFAULTS_CSV, index=False)
    return out


def load_gpu_defaults():
    global GPU_DEFAULTS_CACHE, GPU_DEFAULTS_BY_CHIPSET
    if GPU_DEFAULTS_CACHE is not None:
        return GPU_DEFAULTS_CACHE

    if os.path.exists(GPU_DEFAULTS_CSV):
        df = pd.read_csv(GPU_DEFAULTS_CSV)
    else:
        df = build_gpu_chipset_defaults_from_train_ui()

    df["chipset"] = df["chipset"].astype(str)
    GPU_DEFAULTS_CACHE = df
    GPU_DEFAULTS_BY_CHIPSET = {r["chipset"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_cpu_defaults():
    global CPU_DEFAULTS_CACHE, CPU_DEFAULTS_BY_NAME
    if CPU_DEFAULTS_CACHE is not None:
        return CPU_DEFAULTS_CACHE

    if not os.path.exists(CPU_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing CPU defaults: {CPU_DEFAULTS_CSV}")

    df = pd.read_csv(CPU_DEFAULTS_CSV)
    df["cpu_name"] = df["cpu_name"].astype(str)
    CPU_DEFAULTS_CACHE = df
    CPU_DEFAULTS_BY_NAME = {r["cpu_name"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_ssd_defaults():
    global SSD_DEFAULTS_CACHE, SSD_DEFAULTS_BY_KEY
    if SSD_DEFAULTS_CACHE is not None:
        return SSD_DEFAULTS_CACHE

    if not os.path.exists(SSD_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing SSD defaults: {SSD_DEFAULTS_CSV}")

    df = pd.read_csv(SSD_DEFAULTS_CSV)
    if "ssd_model_key" not in df.columns:
        raise ValueError("ssd_defaults.csv must contain ssd_model_key")

    df["ssd_model_key"] = df["ssd_model_key"].astype(str)
    SSD_DEFAULTS_CACHE = df
    SSD_DEFAULTS_BY_KEY = {r["ssd_model_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_ram_defaults():
    global RAM_DEFAULTS_CACHE, RAM_DEFAULTS_BY_KEY
    if RAM_DEFAULTS_CACHE is not None:
        return RAM_DEFAULTS_CACHE

    if not os.path.exists(RAM_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing RAM defaults: {RAM_DEFAULTS_CSV}")

    df = pd.read_csv(RAM_DEFAULTS_CSV)
    if "ram_model_key" not in df.columns:
        raise ValueError("ram_defaults.csv must contain ram_model_key")

    df["ram_model_key"] = df["ram_model_key"].astype(str)
    RAM_DEFAULTS_CACHE = df
    RAM_DEFAULTS_BY_KEY = {r["ram_model_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_mobo_defaults():
    global MOBO_DEFAULTS_CACHE, MOBO_DEFAULTS_BY_KEY
    if MOBO_DEFAULTS_CACHE is not None:
        return MOBO_DEFAULTS_CACHE

    if not os.path.exists(MOBO_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing MOBO defaults: {MOBO_DEFAULTS_CSV}")

    df = pd.read_csv(MOBO_DEFAULTS_CSV)
    if "mobo_key" not in df.columns:
        raise ValueError("mobo_defaults.csv must contain mobo_key")

    df["mobo_key"] = df["mobo_key"].astype(str)
    MOBO_DEFAULTS_CACHE = df
    MOBO_DEFAULTS_BY_KEY = {r["mobo_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_monitor_defaults():
    global MONITOR_DEFAULTS_CACHE, MONITOR_DEFAULTS_BY_KEY
    if MONITOR_DEFAULTS_CACHE is not None:
        return MONITOR_DEFAULTS_CACHE

    if not os.path.exists(MONITOR_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing MONITOR defaults: {MONITOR_DEFAULTS_CSV}")

    df = pd.read_csv(MONITOR_DEFAULTS_CSV)
    if "monitor_key" not in df.columns:
        raise ValueError("monitor_defaults.csv must contain monitor_key")

    df["monitor_key"] = df["monitor_key"].astype(str)
    MONITOR_DEFAULTS_CACHE = df
    MONITOR_DEFAULTS_BY_KEY = {r["monitor_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_psu_defaults():
    global PSU_DEFAULTS_CACHE, PSU_DEFAULTS_BY_KEY
    if PSU_DEFAULTS_CACHE is not None:
        return PSU_DEFAULTS_CACHE

    if not os.path.exists(PSU_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing PSU defaults: {PSU_DEFAULTS_CSV}")

    df = pd.read_csv(PSU_DEFAULTS_CSV)
    if "psu_key" not in df.columns:
        raise ValueError("psu_defaults.csv must contain psu_key")

    df["psu_key"] = df["psu_key"].astype(str)
    PSU_DEFAULTS_CACHE = df
    PSU_DEFAULTS_BY_KEY = {r["psu_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_case_defaults():
    global CASE_DEFAULTS_CACHE, CASE_DEFAULTS_BY_KEY
    if CASE_DEFAULTS_CACHE is not None:
        return CASE_DEFAULTS_CACHE

    if not os.path.exists(CASE_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing CASE defaults: {CASE_DEFAULTS_CSV}")

    df = pd.read_csv(CASE_DEFAULTS_CSV)
    if "case_key" not in df.columns:
        raise ValueError("case_defaults.csv must contain case_key")

    df["case_key"] = df["case_key"].astype(str)
    CASE_DEFAULTS_CACHE = df
    CASE_DEFAULTS_BY_KEY = {r["case_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_case_fan_defaults():
    global CASE_FAN_DEFAULTS_CACHE, CASE_FAN_DEFAULTS_BY_KEY
    if CASE_FAN_DEFAULTS_CACHE is not None:
        return CASE_FAN_DEFAULTS_CACHE

    if not os.path.exists(CASE_FAN_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing CASE FAN defaults: {CASE_FAN_DEFAULTS_CSV}")

    df = pd.read_csv(CASE_FAN_DEFAULTS_CSV)
    if "fan_key" not in df.columns:
        raise ValueError("case_fan_defaults.csv must contain fan_key")

    df["fan_key"] = df["fan_key"].astype(str)
    CASE_FAN_DEFAULTS_CACHE = df
    CASE_FAN_DEFAULTS_BY_KEY = {r["fan_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


def load_cpu_cooler_defaults():
    global CPU_COOLER_DEFAULTS_CACHE, CPU_COOLER_DEFAULTS_BY_KEY
    if CPU_COOLER_DEFAULTS_CACHE is not None:
        return CPU_COOLER_DEFAULTS_CACHE

    if not os.path.exists(CPU_COOLER_DEFAULTS_CSV):
        raise FileNotFoundError(f"Missing CPU COOLER defaults: {CPU_COOLER_DEFAULTS_CSV}")

    df = pd.read_csv(CPU_COOLER_DEFAULTS_CSV)
    if "cooler_key" not in df.columns:
        raise ValueError("cpu_cooler_defaults.csv must contain cooler_key")

    df["cooler_key"] = df["cooler_key"].astype(str)
    CPU_COOLER_DEFAULTS_CACHE = df
    CPU_COOLER_DEFAULTS_BY_KEY = {r["cooler_key"]: r.to_dict() for _, r in df.iterrows()}
    return df


# =============================================================================
# UI: dropdown options
# =============================================================================
def build_select_options():
    select_options = {}

    # GPU 180d
    try:
        dfg = load_gpu_defaults()
        chipsets = sorted(dfg["chipset"].dropna().astype(str).unique().tolist())
        chipset_opts = []
        for c in chipsets:
            row = GPU_DEFAULTS_BY_CHIPSET.get(c, {})
            p = row.get("price_now_default", None)
            vram = row.get("gpu_memory_gb", None)
            label = c
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            try:
                if vram is not None and np.isfinite(float(vram)):
                    label += f" | {float(vram):.0f}GB"
            except Exception:
                pass
            chipset_opts.append((c, label))
        select_options["gpu_180d__chipset"] = chipset_opts
        select_options["gpu_180d__brand"] = [(b, b) for b in AIB_BRANDS]
        mem_types = sorted(dfg["gpu_memory_type"].dropna().astype(str).unique().tolist())
        select_options["gpu_180d__gpu_memory_type"] = [(m, m) for m in mem_types]
        archs = sorted(dfg["gpu_architecture"].dropna().astype(str).unique().tolist())
        select_options["gpu_180d__gpu_architecture"] = [(a, a) for a in archs]
        vram_vals = sorted({int(round(x)) for x in pd.to_numeric(dfg["gpu_memory_gb"], errors="coerce").dropna().tolist()})
        select_options["gpu_180d__gpu_memory_gb"] = [(str(v), f"{v} GB") for v in vram_vals]
    except Exception:
        select_options["gpu_180d__chipset"] = []
        select_options["gpu_180d__brand"] = [(b, b) for b in AIB_BRANDS]
        select_options["gpu_180d__gpu_memory_type"] = []
        select_options["gpu_180d__gpu_architecture"] = []
        select_options["gpu_180d__gpu_memory_gb"] = []

    # CPU 180d
    try:
        dfc = load_cpu_defaults()
        cpu_list = sorted(dfc["cpu_name"].dropna().astype(str).unique().tolist())
        cpu_opts = []
        for name in cpu_list:
            row = CPU_DEFAULTS_BY_NAME.get(name, {})
            p = row.get("price_now_default", None)
            label = name
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            cpu_opts.append((name, label))
        select_options["cpu_180d__cpu_name"] = cpu_opts
    except Exception:
        select_options["cpu_180d__cpu_name"] = []

    # SSD 180d
    try:
        dfs = load_ssd_defaults()
        keys = sorted(dfs["ssd_model_key"].dropna().astype(str).unique().tolist())
        key_opts = []
        for k in keys:
            row = SSD_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            cap = row.get("capacity_gb", None)
            iface = row.get("interface", "")
            ff = row.get("form_factor", "")
            brand = row.get("brand", "")
            label = f"{brand} | {iface} | {ff}"
            try:
                if cap is not None and np.isfinite(float(cap)):
                    label += f" | {float(cap):.0f}GB"
            except Exception:
                pass
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            key_opts.append((k, label))
        select_options["ssd_180d__ssd_model_key"] = key_opts
    except Exception:
        select_options["ssd_180d__ssd_model_key"] = []

    # RAM 180d
    try:
        dfr = load_ram_defaults()
        keys = sorted(dfr["ram_model_key"].dropna().astype(str).unique().tolist())
        opts = []
        for k in keys:
            row = RAM_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            label = k
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            opts.append((k, label))
        select_options["ram_180d__ram_model_key"] = opts
    except Exception:
        select_options["ram_180d__ram_model_key"] = []

    # MOBO 180d
    try:
        df_mobo = load_mobo_defaults()
        keys = sorted(df_mobo["mobo_key"].dropna().astype(str).unique().tolist())
        opts = []
        for k in keys:
            row = MOBO_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            label = k
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            opts.append((k, label))
        select_options["mobo_180d__mobo_key"] = opts
    except Exception:
        select_options["mobo_180d__mobo_key"] = []

    # MONITOR 180d
    try:
        df_mon = load_monitor_defaults()
        keys = sorted(df_mon["monitor_key"].dropna().astype(str).unique().tolist())
        opts = []
        for k in keys:
            row = MONITOR_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            label = k
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            opts.append((k, label))
        select_options["monitor_180d__monitor_key"] = opts
    except Exception:
        select_options["monitor_180d__monitor_key"] = []

    # PSU 180d
    try:
        df_psu = load_psu_defaults()
        keys = sorted(df_psu["psu_key"].dropna().astype(str).unique().tolist())
        opts = []
        for k in keys:
            row = PSU_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            label = k
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            opts.append((k, label))
        select_options["psu_180d__psu_key"] = opts
    except Exception:
        select_options["psu_180d__psu_key"] = []

    # CASE 180d
    try:
        df_case = load_case_defaults()
        keys = sorted(df_case["case_key"].dropna().astype(str).unique().tolist())
        opts = []
        for k in keys:
            row = CASE_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            label = k
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            opts.append((k, label))
        select_options["case_180d__case_key"] = opts
    except Exception:
        select_options["case_180d__case_key"] = []

    # CASE FAN 180d
    try:
        df_fan = load_case_fan_defaults()
        keys = sorted(df_fan["fan_key"].dropna().astype(str).unique().tolist())
        opts = []
        for k in keys:
            row = CASE_FAN_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            label = k
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            opts.append((k, label))
        select_options["case_fan_180d__fan_key"] = opts
    except Exception:
        select_options["case_fan_180d__fan_key"] = []

    # CPU COOLER 180d
    try:
        dfcc = load_cpu_cooler_defaults()
        keys = sorted(dfcc["cooler_key"].dropna().astype(str).unique().tolist())
        opts = []
        for k in keys:
            row = CPU_COOLER_DEFAULTS_BY_KEY.get(k, {})
            p = row.get("price_now_default", None)
            brand = row.get("brand", "")
            ctype = row.get("cooler_type", "")
            label = k
            if brand or ctype:
                label = f"{brand} {ctype}".strip() + f" | {k}"
            try:
                if p is not None and np.isfinite(float(p)):
                    label += f" | ~${float(p):.0f}"
            except Exception:
                pass
            opts.append((k, label))
        select_options["cpu_cooler_180d__cooler_key"] = opts
    except Exception:
        select_options["cpu_cooler_180d__cooler_key"] = []

    return select_options


# =============================================================================
# Prediction per component
# =============================================================================
def predict_component(comp_key, cfg, form):
    model_obj = load_model(cfg["model_path"])
    model, log_target = unwrap_model(model_obj)

    # GPU 180d
    if comp_key == "gpu_180d":
        load_gpu_defaults()
        chipset = (form.get("gpu_180d__chipset") or "").strip()
        if not chipset:
            raise ValueError("Please select a chipset for GPU 180d forecast.")
        if chipset not in GPU_DEFAULTS_BY_CHIPSET:
            raise ValueError(f"Unknown chipset: {chipset}")

        base = dict(GPU_DEFAULTS_BY_CHIPSET[chipset])
        brand = (form.get("gpu_180d__brand") or "").strip()
        price_now = parse_float(form.get("gpu_180d__price_now"))

        vram_override = (form.get("gpu_180d__gpu_memory_gb") or "").strip()
        memtype_override = (form.get("gpu_180d__gpu_memory_type") or "").strip()
        arch_override = (form.get("gpu_180d__gpu_architecture") or "").strip()

        row = {
            "gpu_memory_gb": float(vram_override) if vram_override else base.get("gpu_memory_gb", np.nan),
            "gpu_core_clock_mhz": base.get("gpu_core_clock_mhz", np.nan),
            "gpu_boost_clock_mhz": base.get("gpu_boost_clock_mhz", np.nan),
            "gpu_release_year": base.get("gpu_release_year", np.nan),
            "gpu_bus_width_bits": base.get("gpu_bus_width_bits", np.nan),
            "memory_bandwidth_proxy": base.get("memory_bandwidth_proxy", np.nan),
            "gpu_memory_type": memtype_override if memtype_override else base.get("gpu_memory_type", None),
            "gpu_architecture": arch_override if arch_override else base.get("gpu_architecture", None),
            "brand": brand if brand else "OTHER",
            "ma_4w": base.get("ma_4w_default", np.nan),
            "ma_12w": base.get("ma_12w_default", np.nan),
            "vol_12w": base.get("vol_12w_default", np.nan),
            "mom_4w": base.get("mom_4w_default", np.nan),
        }
        row["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        if np.isfinite(row.get("gpu_boost_clock_mhz", np.nan)) and np.isfinite(row.get("gpu_core_clock_mhz", np.nan)):
            row["clock_delta"] = float(row["gpu_boost_clock_mhz"] - row["gpu_core_clock_mhz"])
        else:
            row["clock_delta"] = base.get("clock_delta_default", np.nan)

        X = pd.DataFrame([row])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        return {"pred_price": round(y, 2), "meta": {"chipset": chipset, "brand": row["brand"], "price_now": round(float(row["price_now"]), 2), "pred_price": round(y, 2)}}

    # CPU 180d
    if comp_key == "cpu_180d":
        load_cpu_defaults()
        cpu_name = (form.get("cpu_180d__cpu_name") or "").strip()
        if not cpu_name:
            raise ValueError("Please select a CPU model for CPU 180d forecast.")
        if cpu_name not in CPU_DEFAULTS_BY_NAME:
            raise ValueError(f"Unknown CPU model: {cpu_name}")

        base = dict(CPU_DEFAULTS_BY_NAME[cpu_name])
        price_now = parse_float(form.get("cpu_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        return {"pred_price": round(y, 2), "meta": {"cpu_name": cpu_name, "price_now": round(float(base["price_now"]), 2), "pred_price": round(y, 2)}}

    # SSD 180d
    if comp_key == "ssd_180d":
        load_ssd_defaults()
        key = (form.get("ssd_180d__ssd_model_key") or "").strip()
        if not key:
            raise ValueError("Please select an SSD model for SSD 180d forecast.")
        if key not in SSD_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown SSD key: {key}")

        base = dict(SSD_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("ssd_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "ssd_model_key": key,
            "brand": base.get("brand"),
            "interface": base.get("interface"),
            "form_factor": base.get("form_factor"),
            "capacity_gb": base.get("capacity_gb"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # RAM 180d
    if comp_key == "ram_180d":
        load_ram_defaults()
        key = (form.get("ram_180d__ram_model_key") or "").strip()
        if not key:
            raise ValueError("Please select a RAM kit for RAM 180d forecast.")
        if key not in RAM_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown RAM key: {key}")

        base = dict(RAM_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("ram_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "ram_model_key": key,
            "brand": base.get("brand"),
            "ddr_gen": base.get("ddr_gen"),
            "speed_mts": base.get("speed_mts"),
            "total_capacity_gb": base.get("total_capacity_gb"),
            "module_count": base.get("module_count"),
            "cas_latency": base.get("cas_latency"),
            "form_factor": base.get("form_factor"),
            "is_rgb": base.get("is_rgb"),
            "is_ecc": base.get("is_ecc"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # MOBO 180d
    if comp_key == "mobo_180d":
        load_mobo_defaults()
        key = (form.get("mobo_180d__mobo_key") or "").strip()
        if not key:
            raise ValueError("Please select a motherboard key for MOBO 180d forecast.")
        if key not in MOBO_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown motherboard key: {key}")

        base = dict(MOBO_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("mobo_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "mobo_key": key,
            "brand": base.get("brand"),
            "chipset": base.get("chipset"),
            "socket": base.get("socket"),
            "memory_type": base.get("memory_type"),
            "form_factor": base.get("form_factor"),
            "has_wifi": base.get("has_wifi"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # MONITOR 180d
    if comp_key == "monitor_180d":
        load_monitor_defaults()
        key = (form.get("monitor_180d__monitor_key") or "").strip()
        if not key:
            raise ValueError("Please select a monitor key for Monitor 180d forecast.")
        if key not in MONITOR_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown monitor key: {key}")

        base = dict(MONITOR_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("monitor_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "monitor_key": key,
            "brand": base.get("brand"),
            "resolution_class": base.get("resolution_class"),
            "panel_type": base.get("panel_type"),
            "screen_size_in": base.get("screen_size_in"),
            "refresh_hz": base.get("refresh_hz"),
            "is_ultrawide": base.get("is_ultrawide"),
            "is_oled": base.get("is_oled"),
            "is_mini_led": base.get("is_mini_led"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # PSU 180d
    if comp_key == "psu_180d":
        load_psu_defaults()
        key = (form.get("psu_180d__psu_key") or "").strip()
        if not key:
            raise ValueError("Please select a PSU key for PSU 180d forecast.")
        if key not in PSU_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown PSU key: {key}")

        base = dict(PSU_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("psu_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "psu_key": key,
            "brand": base.get("brand"),
            "efficiency_rating": base.get("efficiency_rating"),
            "modular": base.get("modular"),
            "form_factor": base.get("form_factor"),
            "wattage": base.get("wattage"),
            "is_atx3": base.get("is_atx3"),
            "has_pcie5": base.get("has_pcie5"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # CASE 180d
    if comp_key == "case_180d":
        load_case_defaults()
        key = (form.get("case_180d__case_key") or "").strip()
        if not key:
            raise ValueError("Please select a case key for Case 180d forecast.")
        if key not in CASE_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown case key: {key}")

        base = dict(CASE_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("case_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "case_key": key,
            "brand": base.get("brand"),
            "case_size": base.get("case_size"),
            "has_glass": base.get("has_glass"),
            "supports_itx": base.get("supports_itx"),
            "supports_matx": base.get("supports_matx"),
            "supports_atx": base.get("supports_atx"),
            "supports_eatx": base.get("supports_eatx"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # CASE FAN 180d
    if comp_key == "case_fan_180d":
        load_case_fan_defaults()
        key = (form.get("case_fan_180d__fan_key") or "").strip()
        if not key:
            raise ValueError("Please select a case fan key for Case Fan 180d forecast.")
        if key not in CASE_FAN_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown case fan key: {key}")

        base = dict(CASE_FAN_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("case_fan_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "fan_key": key,
            "brand": base.get("brand"),
            "size_mm": base.get("size_mm"),
            "pwm": base.get("pwm"),
            "is_rgb": base.get("is_rgb"),
            "pack_count": base.get("pack_count"),
            "rpm_max": base.get("rpm_max"),
            "airflow_cfm": base.get("airflow_cfm"),
            "noise_dba": base.get("noise_dba"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # CPU COOLER 180d
    if comp_key == "cpu_cooler_180d":
        load_cpu_cooler_defaults()
        key = (form.get("cpu_cooler_180d__cooler_key") or "").strip()
        if not key:
            raise ValueError("Please select a CPU cooler model.")
        if key not in CPU_COOLER_DEFAULTS_BY_KEY:
            raise ValueError(f"Unknown CPU cooler key: {key}")

        base = dict(CPU_COOLER_DEFAULTS_BY_KEY[key])
        price_now = parse_float(form.get("cpu_cooler_180d__price_now"))
        base["price_now"] = float(price_now) if np.isfinite(price_now) else base.get("price_now_default", np.nan)
        base["ma_4w"] = base.get("ma_4w_default", np.nan)
        base["ma_12w"] = base.get("ma_12w_default", np.nan)
        base["vol_12w"] = base.get("vol_12w_default", np.nan)
        base["mom_4w"] = base.get("mom_4w_default", np.nan)

        X = pd.DataFrame([base])
        X = _align_to_feature_cols(model_obj, X)
        y = _predict_price(model, log_target, X)

        meta = {
            "cooler_key": key,
            "brand": base.get("brand"),
            "cooler_type": base.get("cooler_type"),
            "price_now": round(float(base["price_now"]), 2) if np.isfinite(base.get("price_now", np.nan)) else None,
            "pred_price": round(y, 2),
        }
        return {"pred_price": round(y, 2), "meta": meta}

    # Default/manual components (if any remain)
    X = build_input_row(comp_key, cfg, form)
    X = _align_to_feature_cols(model_obj, X)
    y = _predict_price(model, log_target, X)
    return round(y, 2)


# =============================================================================
# UI fields
# =============================================================================
def get_fields_for_component(comp_key, cfg):
    if "fields" in cfg and isinstance(cfg["fields"], list) and len(cfg["fields"]) > 0:
        return cfg["fields"]

    fields = []
    for c in cfg.get("cat_cols", []):
        fields.append({"name": c, "label": c, "type": "text"})
    for c in cfg.get("num_cols", []):
        if "has" in c.lower() or c.lower().startswith("is_") or c.lower().startswith("support") or c.lower() in ("pwm",):
            fields.append({"name": c, "label": c, "type": "checkbox"})
        else:
            fields.append({"name": c, "label": c, "type": "number", "step": "0.1"})
    return fields


# =============================================================================
# Routes
# =============================================================================
@app.route("/", methods=["GET", "POST"])
def index():
    comp_keys = list_components()
    comps = [(k, COMPONENTS[k]) for k in comp_keys]

    selected = set()
    results = None
    errors = []

    select_options = build_select_options()

    if request.method == "POST":
        selected_list = request.form.getlist("selected_components")
        selected = set(selected_list)

        per_component = {}
        total = 0.0

        for comp_key, cfg in comps:
            if comp_key not in selected:
                continue
            try:
                out = predict_component(comp_key, cfg, request.form)

                if isinstance(out, dict) and "pred_price" in out:
                    per_component[comp_key] = float(out["pred_price"])
                    total += float(out["pred_price"])
                    per_component[f"{comp_key}__meta"] = out.get("meta", {})
                else:
                    per_component[comp_key] = float(out)
                    total += float(out)

            except Exception as e:
                errors.append(f"{comp_key}: {e}")

        results = {"per_component": per_component, "total": round(total, 2)}

    ui_components = []
    for comp_key, cfg in comps:
        fields = get_fields_for_component(comp_key, cfg)
        ui_components.append({
            "key": comp_key,
            "display_name": cfg.get("display_name", comp_key),
            "fields": fields,
        })

    return render_template(
        "index.html",
        components=ui_components,
        selected=selected,
        results=results,
        errors=errors,
        form_data=request.form,
        select_options=select_options,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)