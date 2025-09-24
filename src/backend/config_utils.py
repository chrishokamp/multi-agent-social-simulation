"""
Configuration utilities for dynamic variable processing.
"""
import copy
import random
import math


def normalize_simulation_config(request_data):
    """
    Normalize config structure to handle both CLI and UI formats.

    Args:
        request_data (dict): Raw request data from create endpoint

    Returns:
        tuple: (config_dict, variables_dict, num_runs)

    Handles:
        - CLI format: {"config": {...}, "variables": {...}, "num_runs": N}
        - UI format: {"config": {"config": {...}, "variables": {...}}, "num_runs": N}
        - Direct config: {"agents": [...], ...} with num_runs
        - Path-based: {"path": "..."} with num_runs
    """
    if not isinstance(request_data, dict):
        raise ValueError("Request data must be a dictionary")

    # Extract num_runs first (required for all formats)
    num_runs = request_data.get("num_runs")
    if not num_runs or not isinstance(num_runs, int) or num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")

    # Case 1: Direct agents config (raw JSON)
    if "agents" in request_data:
        config = {k: v for k, v in request_data.items() if k != "num_runs"}
        variables = config.pop("variables", None)
        return config, variables, num_runs

    # Case 2: Nested config structure (CLI/UI)
    elif "config" in request_data:
        nested_config = request_data["config"]

        # UI format: {"config": {"config": {...}, "variables": {...}}}
        if isinstance(nested_config, dict) and "config" in nested_config:
            config = nested_config["config"]
            variables = nested_config.get("variables")
            return config, variables, num_runs

        # CLI format: {"config": {...}} (variables at top level)
        else:
            config = nested_config
            variables = request_data.get("variables")
            return config, variables, num_runs

    # Case 3: Path-based config
    elif "path" in request_data:
        from pathlib import Path
        import json

        path = Path(request_data["path"])
        if not path.exists():
            raise ValueError(f"Config file not found: {path}")

        with open(path, "r") as f:
            file_config = json.load(f)

        # File might have variables at top level
        config = file_config.get("config", file_config)
        variables = file_config.get("variables")
        return config, variables, num_runs

    else:
        raise ValueError("Invalid request format. Must provide 'agents', 'config', or 'path'.")


def materialise_config(cfg: dict) -> dict:
    """
    Return a *static* copy of `cfg` by
      1) sampling all entries under cfg['variables'] (if present);
      2) filling `{var}` placeholders in agent prompts / strategy dicts;
      3) removing the `variables` section.
    A config lacking the 'variables' key is considered already static.
    """
    static = copy.deepcopy(cfg)

    var_rules = static.pop("variables", None)
    if not var_rules:            # already static
        return static.get('config', static)
    if "config" not in static:
        return static

    SAFE = {"randint": random.randint, "choice": random.choice,
            "min": min, "max": max, "abs": abs, "round": round, "math": math}

    values = {}
    for name, rule in var_rules.items():
        if "range" in rule:
            r = rule["range"]
            step = r.get("step", 1)
            values[name] = random.randrange(r["min"], r["max"]+1, step)
        elif "choice" in rule:
            values[name] = random.choice(rule["choice"])
        elif "expr" in rule:
            values[name] = eval(rule["expr"], SAFE, values)
        else:
            raise ValueError(f"Unknown rule for variable '{name}'")

    for ag in static["config"]["agents"]:
        # prompt
        ag["prompt"] = ag["prompt"].format(**values)
        # strategy dict
        ag["strategy"] = {k: values.get(v, v) for k, v in ag["strategy"].items()}

    return static["config"]