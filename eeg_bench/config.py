import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_config = None


@dataclass
class LeJEPAConfig:
    """Configuration for LeJEPA models (BCI and Clinical)."""
    eegfm_path: Optional[str] = None
    pos_bank_path: str = "./REVE_posbank"
    freeze_encoder: bool = True
    # Checkpoint resolution (shared by BCI and Clinical)
    checkpoint_base_path: Optional[str] = None
    checkpoint_version: Optional[int] = None
    checkpoint_full_path: Optional[str] = None  # Direct override

    def get_checkpoint_path(self) -> Optional[str]:
        """Resolve checkpoint path. full_path takes priority over base_path+version."""
        if self.checkpoint_full_path:
            return self.checkpoint_full_path
        if self.checkpoint_base_path and self.checkpoint_version is not None:
            base = Path(self.checkpoint_base_path)
            return str(base / f"version_{self.checkpoint_version}" / "checkpoints" / "last.ckpt")
        return None

def load_config():
    global _config
    if _config is None:
        config_path = os.environ.get("EEG_BENCH_CONFIG")
        if config_path:
            config_path = Path(config_path).expanduser()
        else:
            script_dir = Path(__file__).resolve().parent
            config_path = script_dir / "config.json"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                _config = json.load(f)
        else:
            _config = {}
    return _config

def get_config_value(key, default=None):
    # Check environment variable override
    env_key = f"EEG_BENCHMARK_{key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]

    config = load_config()
    return config.get(key, default)

def get_data_path(dataset_key=None, fallback_subdir=None):
    """
    Get path for a specific dataset or general data dir.
    Priority:
        1. Env var (e.g., EEG_BENCHMARK_TUEP)
        2. config.json value
        3. Default: <project_root>/data/<fallback_subdir>
    """
    if dataset_key:
        path = get_config_value(dataset_key)
        if path:
            return Path(path).expanduser()

    # Otherwise, use general "data" path from config or default
    base_data_path = get_config_value("data")
    if base_data_path:
        base_data_path = Path(base_data_path).expanduser()
    else:
        # Use project root's /data/ folder
        script_dir = Path(__file__).resolve().parent
        base_data_path = script_dir.parent / "data"

    if fallback_subdir:
        return base_data_path / fallback_subdir
    return base_data_path


def load_lejepa_config(config_file_override: Optional[str] = None) -> dict:
    """
    Load LeJEPA config from JSON file.

    Priority:
    1. config_file_override (if provided via --lejepa-config)
    2. Default config.json's "lejepa" section
    3. Hardcoded defaults
    """
    defaults = {
        "eegfm_path": None,
        "pos_bank_path": "./REVE_posbank",
        "freeze_encoder": True,
        "checkpoint": {
            "base_path": None,
            "version": None,
            "full_path": None
        }
    }

    if config_file_override:
        config_path = Path(config_file_override).expanduser()
        if config_path.exists():
            with open(config_path, "r") as f:
                file_config = json.load(f)
            return _deep_merge(defaults, file_config)
        else:
            print(f"[Warning] LeJEPA config file not found: {config_file_override}. Using defaults.")

    # Load from default config.json
    config = load_config()
    lejepa_config = config.get("lejepa", {})
    return _deep_merge(defaults, lejepa_config)


def merge_lejepa_config_with_cli(base_config: dict, cli_args) -> LeJEPAConfig:
    """
    Merge JSON config with CLI arguments. CLI takes precedence.

    Args:
        base_config: Config dict loaded from JSON
        cli_args: argparse Namespace

    Returns:
        LeJEPAConfig dataclass instance
    """
    checkpoint_config = base_config.get("checkpoint", {})

    # Start with JSON config values
    config = LeJEPAConfig(
        eegfm_path=base_config.get("eegfm_path"),
        pos_bank_path=base_config.get("pos_bank_path", "./REVE_posbank"),
        freeze_encoder=base_config.get("freeze_encoder", True),
        checkpoint_base_path=checkpoint_config.get("base_path"),
        checkpoint_version=checkpoint_config.get("version"),
        checkpoint_full_path=checkpoint_config.get("full_path"),
    )

    # Override with CLI args (if provided)
    if getattr(cli_args, "lejepa_checkpoint_base_path", None):
        config.checkpoint_base_path = cli_args.lejepa_checkpoint_base_path

    if getattr(cli_args, "lejepa_checkpoint_version", None) is not None:
        config.checkpoint_version = cli_args.lejepa_checkpoint_version

    if getattr(cli_args, "lejepa_checkpoint_full_path", None):
        config.checkpoint_full_path = cli_args.lejepa_checkpoint_full_path

    if getattr(cli_args, "lejepa_pos_bank_path", None):
        config.pos_bank_path = cli_args.lejepa_pos_bank_path

    # Handle freeze_encoder boolean flags
    if getattr(cli_args, "lejepa_freeze_encoder", False):
        config.freeze_encoder = True
    elif getattr(cli_args, "lejepa_no_freeze_encoder", False):
        config.freeze_encoder = False

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif value is not None:  # Only override if value is not None
            result[key] = value
    return result
