"""
Real dataset loaders for Bifrost experiments.

Provides unified loading for:
- SpeechCommands (audio, 35 classes, 16kHz)
- CIFAR-10 (images, 10 classes, 32x32 RGB)
- ESC-50 (environmental audio, 50 classes, 44.1kHz)
- UCI HAR (sensor, 6 activities, 50Hz IMU)

All loaders return (data, labels, sample_rate_or_metadata) tuples
with consistent interfaces for experiment scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter


DATA_DIR = Path(__file__).parent.parent / "data"


def load_speechcommands(
    n_classes: int = 10,
    n_samples_per_class: int = 200,
    sample_rate: int = 16000,
    duration: float = 1.0,
    subset: str = "training",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Load SpeechCommands dataset.

    Parameters
    ----------
    n_classes : int
        Number of classes to use (top-N most frequent). Default 10.
    n_samples_per_class : int
        Maximum samples per class. Default 200.
    sample_rate : int
        Target sample rate. Default 16000.
    duration : float
        Duration in seconds (pads/truncates). Default 1.0.
    subset : str
        "training", "validation", or "testing".
    seed : int
        Random seed for sampling.

    Returns
    -------
    waveforms : torch.Tensor (N, T)
    labels : torch.Tensor (N,)
    class_names : List[str]
    """
    import torchaudio

    target_len = int(sample_rate * duration)
    ds = torchaudio.datasets.SPEECHCOMMANDS(
        root=str(DATA_DIR / "speechcommands"),
        download=False,
        subset=subset,
    )

    # Collect all samples with labels
    all_samples: Dict[str, List[torch.Tensor]] = {}
    for i in range(len(ds)):
        wav, sr, label, *_ = ds[i]
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.squeeze(0)  # (T,)
        # Pad or truncate
        if wav.shape[0] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[0]))
        else:
            wav = wav[:target_len]
        if label not in all_samples:
            all_samples[label] = []
        all_samples[label].append(wav)

    # Select top-N classes by frequency
    class_counts = {k: len(v) for k, v in all_samples.items()}
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    selected_classes = [c for c, _ in sorted_classes[:n_classes]]
    selected_classes.sort()  # alphabetical for reproducibility

    rng = np.random.RandomState(seed)
    waveforms = []
    labels = []
    for idx, cls in enumerate(selected_classes):
        samples = all_samples[cls]
        if len(samples) > n_samples_per_class:
            chosen = rng.choice(len(samples), n_samples_per_class, replace=False)
            samples = [samples[i] for i in chosen]
        for wav in samples:
            waveforms.append(wav)
            labels.append(idx)

    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels, selected_classes


def load_cifar10(
    n_classes: int = 10,
    n_samples_per_class: int = 500,
    image_size: int = 32,
    train: bool = True,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Load CIFAR-10 dataset.

    Parameters
    ----------
    n_classes : int
        Number of classes (default 10 = all).
    n_samples_per_class : int
        Max samples per class. Default 500.
    image_size : int
        Resize target. Default 32 (native).
    train : bool
        Training or test set.
    seed : int
        Random seed.

    Returns
    -------
    images : torch.Tensor (N, C, H, W) in [0, 1]
    labels : torch.Tensor (N,)
    class_names : List[str]
    """
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # (C, H, W) in [0, 1]
    ])
    ds = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "cifar10"),
        train=train,
        download=False,
        transform=transform,
    )

    class_names = ds.classes[:n_classes]

    # Subsample
    rng = np.random.RandomState(seed)
    indices_per_class: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for i in range(len(ds)):
        _, label = ds[i]
        if label < n_classes:
            indices_per_class[label].append(i)

    selected = []
    for cls_idx in range(n_classes):
        pool = indices_per_class[cls_idx]
        if len(pool) > n_samples_per_class:
            chosen = rng.choice(len(pool), n_samples_per_class, replace=False)
            pool = [pool[i] for i in chosen]
        selected.extend(pool)

    images = []
    labels = []
    for i in selected:
        img, label = ds[i]
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, class_names


def load_esc50(
    n_classes: int = 10,
    n_samples_per_class: int = 40,
    sample_rate: int = 16000,
    duration: float = 5.0,
    esc10_only: bool = True,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Load ESC-50 dataset.

    Parameters
    ----------
    n_classes : int
        Number of classes. Default 10 (ESC-10 subset).
    n_samples_per_class : int
        Max samples per class. Default 40 (ESC-50 has 40 per class).
    sample_rate : int
        Target sample rate. Default 16000.
    duration : float
        Duration in seconds. Default 5.0.
    esc10_only : bool
        Use only ESC-10 subset (10 classes). Default True.
    seed : int
        Random seed.

    Returns
    -------
    waveforms : torch.Tensor (N, T)
    labels : torch.Tensor (N,)
    class_names : List[str]
    """
    import torchaudio
    import pandas as pd

    target_len = int(sample_rate * duration)
    meta = pd.read_csv(str(DATA_DIR / "esc50" / "meta" / "esc50.csv"))

    if esc10_only:
        meta = meta[meta["esc10"] == True]

    # Select top-N categories
    categories = sorted(meta["category"].unique())[:n_classes]
    meta = meta[meta["category"].isin(categories)]

    rng = np.random.RandomState(seed)
    waveforms = []
    labels = []
    class_names = sorted(categories)

    for idx, cat in enumerate(class_names):
        cat_files = meta[meta["category"] == cat]["filename"].tolist()
        if len(cat_files) > n_samples_per_class:
            chosen = rng.choice(len(cat_files), n_samples_per_class, replace=False)
            cat_files = [cat_files[i] for i in chosen]

        for fname in cat_files:
            fpath = str(DATA_DIR / "esc50" / "audio" / fname)
            wav, sr = torchaudio.load(fpath)
            wav = wav.mean(dim=0)  # mono
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.shape[0] < target_len:
                wav = F.pad(wav, (0, target_len - wav.shape[0]))
            else:
                wav = wav[:target_len]
            waveforms.append(wav)
            labels.append(idx)

    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels, class_names


def load_uci_har(
    n_classes: int = 6,
    n_samples_per_class: int = 200,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Load UCI HAR dataset.

    Downloads if not present. Returns 6-channel IMU signals at 50Hz.

    Returns
    -------
    signals : torch.Tensor (N, 6, 128)
    labels : torch.Tensor (N,)
    class_names : List[str]
    """
    import zipfile

    har_dir = DATA_DIR / "uci_har"
    har_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    train_dir = har_dir / "train"
    if not train_dir.exists():
        # Try to download
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = har_dir / "har.zip"
        if not zip_path.exists():
            print(f"Downloading UCI HAR from {url}...")
            import urllib.request
            urllib.request.urlretrieve(url, str(zip_path))

        print("Extracting UCI HAR...")
        with zipfile.ZipFile(str(zip_path), "r") as z:
            z.extractall(str(har_dir))

        # Move contents up if nested
        nested = har_dir / "UCI HAR Dataset"
        if nested.exists():
            for item in nested.iterdir():
                item.rename(har_dir / item.name)
            nested.rmdir()

    # Load signals
    # Inertial signals: body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
    ]
    class_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                   "SITTING", "STANDING", "LAYING"]

    def load_signal(split, name):
        fpath = har_dir / split / "Inertial Signals" / f"{name}_{split}.txt"
        return np.loadtxt(str(fpath))  # (N, 128)

    def load_labels(split):
        fpath = har_dir / split / f"y_{split}.txt"
        return np.loadtxt(str(fpath)).astype(int) - 1  # 0-indexed

    # Load train and test, combine
    train_signals = []
    test_signals = []
    for name in signal_names:
        train_signals.append(load_signal("train", name))
        test_signals.append(load_signal("test", name))

    train_X = np.stack(train_signals, axis=1)  # (N_train, 6, 128)
    test_X = np.stack(test_signals, axis=1)  # (N_test, 6, 128)
    train_y = load_labels("train")
    test_y = load_labels("test")

    X = np.concatenate([train_X, test_X], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)

    # Subsample
    rng = np.random.RandomState(seed)
    selected = []
    for cls in range(n_classes):
        pool = np.where(y == cls)[0]
        if len(pool) > n_samples_per_class:
            pool = rng.choice(pool, n_samples_per_class, replace=False)
        selected.extend(pool)

    X = X[selected]
    y = y[selected]

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    return torch.from_numpy(X).float(), torch.from_numpy(y).long(), class_names[:n_classes]


def load_dataset(
    name: str,
    n_classes: int = 10,
    n_samples_per_class: int = 200,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Unified dataset loading interface.

    Parameters
    ----------
    name : str
        "speechcommands", "cifar10", "esc50", or "ucihar"
    n_classes : int
    n_samples_per_class : int

    Returns
    -------
    data : torch.Tensor
    labels : torch.Tensor
    class_names : List[str]
    """
    if name == "speechcommands":
        return load_speechcommands(n_classes=n_classes,
                                    n_samples_per_class=n_samples_per_class, **kwargs)
    elif name == "cifar10":
        return load_cifar10(n_classes=n_classes,
                            n_samples_per_class=n_samples_per_class, **kwargs)
    elif name == "esc50":
        return load_esc50(n_classes=n_classes,
                          n_samples_per_class=n_samples_per_class, **kwargs)
    elif name == "ucihar":
        return load_uci_har(n_classes=n_classes,
                            n_samples_per_class=n_samples_per_class, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Options: speechcommands, cifar10, esc50, ucihar")
