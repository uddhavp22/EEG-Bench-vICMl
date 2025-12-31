try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


_RUN = None


def init_run(*, project, name, config=None, entity=None, group=None, mode="online"):
    if wandb is None:
        return None
    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        mode=mode,
        config=config or {},
    )


def set_run(run):
    global _RUN
    _RUN = run


def log(metrics, step=None):
    if _RUN is None:
        return
    if step is None:
        _RUN.log(metrics)
        return
    _RUN.log(metrics, step=step)


def finish():
    global _RUN
    if _RUN is None:
        return
    _RUN.finish()
    _RUN = None
