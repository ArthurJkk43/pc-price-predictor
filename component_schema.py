# component_schema.py

COMPONENTS = {

    "gpu_180d": {
    "display_name": "GPU",
    "model_path": "models/gpu_180d_ratio.joblib",
    "fields": [
        {"name": "chipset", "label": "Chipset", "type": "text"},
        {"name": "brand", "label": "Brand (AIB)", "type": "text"},
        {"name": "gpu_memory_gb", "label": "VRAM (override, optional)", "type": "text"},
        {"name": "gpu_memory_type", "label": "Memory Type (override, optional)", "type": "text"},
        {"name": "gpu_architecture", "label": "Architecture (override, optional)", "type": "text"},
        {"name": "price_now", "label": "Current Price (USD)", "type": "number", "step": "0.01"},
    ],
    },

    "cpu_180d": {
    "display_name": "CPU",
    "model_path": "models/cpu_180d_ratio.joblib",
    "fields": [
        {"name": "cpu_name", "label": "CPU Model (chipset)", "type": "text"},   # becomes dropdown via select_options
        {"name": "price_now", "label": "Current Price (USD, optional)", "type": "number", "step": "0.01"},
    ],
    },

    "ssd_180d": {
    "display_name": "SSD",
    "model_path": "models/ssd_180d_ratio.joblib",
    "fields": [
        {"name": "ssd_model_key", "label": "SSD Model (brand | iface | ff | cap)", "type": "text"},  # becomes dropdown
        {"name": "price_now", "label": "Current Price (USD, optional)", "type": "number", "step": "0.01"},
    ],
    },

    "ram_180d": {
    "display_name": "RAM",
    "model_path": "models/ram_180d_ratio.joblib",
    "fields": [
        {"name": "ram_model_key", "label": "RAM Kit (Brand | DDR | SPD | CAP | Mods | CL | FF | RGB | ecc)", "type": "text"},
        {"name": "price_now", "label": "Current Price (USD, optional)", "type": "number", "step": "0.01"},
    ],
    },
    
    "motherboard_180d": {
    "display_name": "Motherboard",
    "model_path": "models/mobo_180d_ratio.joblib",
    "fields": [
        {"name": "mobo_key", "label": "Motherboard (brand | chipset | socket | ddr | ff | wifi)", "type": "text"},
        {"name": "price_now", "label": "Current Price (USD, optional)", "type": "number", "step": "0.01"},
    ],
    },  

    "monitor_180d": {
    "display_name": "Monitor",
    "model_path": "models/monitor_180d_ratio.joblib",
    "fields": [
        {"name": "monitor_key", "label": "Monitor (brand | size | res | hz | panel | uw)", "type": "text"},
        {"name": "price_now", "label": "Current Price (USD, optional)", "type": "number", "step": "0.01"},
    ],
    },

    "psu_180d": {
    "display_name": "PSU",
    "model_path": "models/psu_180d_ratio.joblib",
    "fields": [
        {"name": "psu_key", "label": "PSU (brand | watt | eff | mod | ff | ATX3 | PCIe5)", "type": "text"},
        {"name": "price_now", "label": "Current Price (USD, optional)", "type": "number", "step": "0.01"},
    ],
    },

    "case_180d": {
    "display_name": "Case",
    "model_path": "models/case_180d_ratio.joblib",
    "num_cols": ["price_now"],
    "cat_cols": ["case_key"],

    "fields": [
        {"name": "case_key", "label": "Case Key", "type": "select"},
        {"name": "price_now", "label": "Current Price (USD, optional override)", "type": "number", "step": "0.01"},
    ],
    },

    "cpu_cooler_180d": {
    "display_name": "CPU Cooler",
    "model_path": "models/cpu_cooler_180d_ratio.joblib",
    "fields": [
        {"name": "cooler_key", "label": "CPU Cooler model", "type": "select"},
        {"name": "price_now", "label": "Current price (USD) override", "type": "number", "step": "0.01"},
    ],
    },

    # to be added more components...
}
