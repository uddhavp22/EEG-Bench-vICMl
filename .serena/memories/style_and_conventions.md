# Style and Conventions

- Python with type hints throughout
- `@dataclass` for config objects (e.g., `LeJEPAConfig`, `ExperimentConfig`)
- Logging via `logging.getLogger(__name__)`
- Cache paths built via `Path` + f-strings; use `EMBED_CACHE_VERSION` constant to bust caches
- Result filenames encode experiment metadata via suffix tokens (_LP, _ATTN, _MLP, _pctXX, _ckpt_ID)
- `get_completed_experiments` parses result filenames via regex for resume support
- No docstrings on private helpers; public functions have one-liner or short docstrings
- BCI and clinical models are parallel implementations — changes to one usually need mirroring in the other
