from ..abstract_dataset import AbstractDataset
from typing import Tuple, Optional, Sequence, Callable, Any
from ...enums.bci_classes import BCIClasses
from ...config import get_config_value
from pathlib import Path
import hashlib
import pickle

class BaseBCIDataset(AbstractDataset):
    def __init__(
        self,
        name: str,
        interval: Tuple[float, float],
        target_classes: Sequence[BCIClasses],
        available_classes: Sequence[BCIClasses],
        subjects: Sequence[int],
        sampling_frequency: int,
        channel_names: Sequence[str],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = False,
    ):
        super().__init__(
            target_classes=target_classes,
            subjects=subjects,
        )
        self._interval = interval
        self.name = name
        if target_classes is not None:
            assert all(
                [c in available_classes for c in target_classes]
            ), "Target classes must be a subset of the available classes"
        self.available_classes = available_classes
        self._channel_names = channel_names
        # Default value, to be overridden by subclasses
        self._sampling_frequency = sampling_frequency
        # Default value, to be overridden by subclasses
        if target_channels is not None:
            assert all(
                [channel in self._channel_names for channel in target_channels]
            ), "Target channels must be a subset of the available channels"
        self._target_channels = target_channels
        self._target_frequency = target_frequency
        self._preload = preload
        # TODO make this more generic with a config file and parameters

    def _moabb_cache_key(self, paradigm_name: str) -> str:
        subjects_key = ",".join(map(str, self.subjects)) if self.subjects else "none"
        classes_key = ",".join([c.name for c in self.target_classes]) if self.target_classes else "all"
        target_freq = str(self._target_frequency) if self._target_frequency is not None else "native"
        target_chans = ",".join(self._target_channels) if self._target_channels else "all"
        interval_key = f"{self._interval[0]}-{self._interval[1]}"
        return f"{self.name}|{paradigm_name}|{subjects_key}|{classes_key}|{target_freq}|{target_chans}|{interval_key}"

    def _load_moabb_cached(self, cache_key: str, loader: Callable[[], Any]):
        cache_root = Path(get_config_value("cache", "./cache/")).expanduser()
        cache_dir = cache_root / "moabb"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir / f"{self.name}_{digest}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        data = loader()
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data

    def __getitem__(self, index):
        if self.data is None:
            self.load_data()
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self.name

    def _download(self):
        pass
