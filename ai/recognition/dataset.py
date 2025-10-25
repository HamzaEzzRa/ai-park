from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


class VisitorDataset(tf.data.Dataset):
    """Utility factory that yields a tf.data pipeline for visitor recognition.

    The dataset reads RGBA sprites stored under ``ai/recognition/datasets/visitors``.
    Each image is paired with a JSON label describing ``group_type`` and ``group_size``.
    """

    def __new__(
        cls,
        images_dir: Optional[str | Path] = None,
        labels_dir: Optional[str | Path] = None,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        cache: bool = False,
        prefetch: bool = True,
    ) -> tf.data.Dataset:
        base_path = Path(__file__).resolve().parent / "datasets" / "visitors"
        images_path = Path(images_dir) if images_dir else base_path / "images"
        labels_path = Path(labels_dir) if labels_dir else base_path / "labels"

        if not images_path.exists():
            raise FileNotFoundError(f"Image directory not found: {images_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Label directory not found: {labels_path}")

        image_files = sorted(images_path.glob("*.png"))
        if not image_files:
            raise ValueError(f"No PNG files found in {images_path}")

        image_paths: list[str] = []
        group_types: list[int] = []
        group_sizes: list[int] = []

        for img_path in image_files:
            label_path = labels_path / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            with label_path.open("r", encoding="utf-8") as handle:
                label_data = json.loads(handle.read())

            try:
                group_type = int(label_data["group_type"])
                group_size = int(label_data["group_size"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid label file {label_path}") from exc

            image_paths.append(str(img_path))
            group_types.append(group_type)
            group_sizes.append(group_size)

        if not image_paths:
            raise ValueError("No image/label pairs were found for the visitor dataset.")

        label_tensors = {
            "group_type": np.asarray(group_types, dtype=np.int32),
            "group_size": np.asarray(group_sizes, dtype=np.int32) - 1, # Zero-indexed
        }

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_tensors))

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(image_paths),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        def _load_example(path: tf.Tensor, label: dict[str, tf.Tensor]):
            image_bytes = tf.io.read_file(path)
            image = tf.io.decode_png(image_bytes, channels=4)
            if image_size:
                image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
            if normalize:
                image = tf.image.convert_image_dtype(image, tf.float32)
            else:
                image = tf.cast(image, tf.uint8)
            return image, label

        dataset = dataset.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)

        if cache:
            dataset = dataset.cache()

        dataset = dataset.batch(batch_size)

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

