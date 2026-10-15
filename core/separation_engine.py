"""
Module C: Source Separation Engine
Provides an abstract, pluggable interface for source separation.
Wraps the 1D U-Net as a modular, replaceable component and supports dummy/mock
separators for fast headless testing and embedded benchmarking.
"""

import os
import sys
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


class SeparationEngineError(Exception):
    """Raised when model loading, weight restoration, or inference fails."""
    pass


class BaseSourceSeparator(ABC):
    """
    Abstract source separation interface.
    Decouples neural network inference from input acquisition, buffering,
    resource management, and audio output.
    """

    @property
    @abstractmethod
    def stem_names(self) -> List[str]:
        """List of isolated audio stem names produced by this model (e.g. ['vocals', 'drums'])."""
        pass

    @property
    @abstractmethod
    def supported_segment_length(self) -> int:
        """Expected time-domain sample length per input segment."""
        pass

    @abstractmethod
    def separate_batch(self, segments: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[float]]:
        """
        Performs inference on a batch of audio segments.
        Args:
            segments: Array of shape (batch_size, segment_length, 1).
        Returns:
            separated_stems: Dict mapping stem name to array of shape (batch_size, segment_length).
            latencies_sec: List of inference execution times in seconds per segment.
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Runs dummy inference to initialize computation graphs, cache kernels, and warm up CPU."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Returns model metadata, parameter counts, and configuration."""
        return {
            "model_type": self.__class__.__name__,
            "stems": self.stem_names,
            "segment_length": self.supported_segment_length
        }


class UNetSourceSeparator(BaseSourceSeparator):
    """
    1D U-Net Neural Source Separation implementation.
    Loads and runs the 985K parameter 1D U-Net model.
    """

    def __init__(self, weights_path: str = "./models/model_weights", segment_length: int = 75008):
        self._segment_length = segment_length
        self._weights_path = weights_path
        self._model = None
        self._stems = ["vocals", "drums"]
        self._load_model()

    @property
    def stem_names(self) -> List[str]:
        return self._stems

    @property
    def supported_segment_length(self) -> int:
        return self._segment_length

    def _load_model(self):
        """Initializes and restores TensorFlow 1D U-Net weights with path resolution."""
        try:
            import tensorflow as tf
            # Ensure local Model package imports resolve
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            try:
                from Model.UNetModel import AudioTrackSeparation
            except ImportError:
                from UNetModel import AudioTrackSeparation

            self._model = AudioTrackSeparation()
            self._model.build(input_shape=(None, self._segment_length, 1))

            weights_path = self._weights_path
            # Check directory resolution
            if os.path.isdir(weights_path):
                cand = os.path.join(weights_path, 'model_weights')
                if os.path.exists(cand + '.index') or os.path.exists(cand):
                    weights_path = cand
            elif not (os.path.exists(weights_path + '.index') or os.path.exists(weights_path)):
                # Search relative to project root
                alt = os.path.join(project_root, weights_path)
                if os.path.isdir(alt):
                    cand = os.path.join(alt, 'model_weights')
                    if os.path.exists(cand + '.index'):
                        weights_path = cand
                elif os.path.exists(alt + '.index') or os.path.exists(alt):
                    weights_path = alt

            if not (os.path.exists(weights_path + '.index') or os.path.exists(weights_path)):
                raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

            self._model.load_weights(weights_path).expect_partial()
            self._tf = tf
        except Exception as e:
            raise SeparationEngineError(f"Failed to initialize U-Net model: {str(e)}")

    def warmup(self) -> None:
        """Executes one dummy pass to compile TensorFlow execution graph."""
        dummy_input = np.zeros((1, self._segment_length, 1), dtype=np.float32)
        tensor = self._tf.convert_to_tensor(dummy_input)
        _ = self._model(tensor, training=False)

    def separate_batch(self, segments: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[float]]:
        """
        Executes model inference on segmented audio array.
        Args:
            segments: Shape (num_segments, segment_length, 1) or (num_segments, segment_length).
        Returns:
            stems: {"vocals": np.ndarray, "drums": np.ndarray}
            latencies: execution latency per segment in seconds.
        """
        if segments.ndim == 2:
            segments = np.expand_dims(segments, axis=-1)

        batch_size = segments.shape[0]
        vocals_list = []
        drums_list = []
        latencies = []

        # Process segment by segment to track granular embedded inference latency and control RAM
        for i in range(batch_size):
            chunk = segments[i:i+1]
            t_start = time.perf_counter()

            tensor = self._tf.convert_to_tensor(chunk, dtype=self._tf.float32)
            pred_voc, pred_drm = self._model(tensor, training=False)

            voc_np = pred_voc.numpy().squeeze()
            drm_np = pred_drm.numpy().squeeze()
            t_elapsed = time.perf_counter() - t_start

            vocals_list.append(voc_np)
            drums_list.append(drm_np)
            latencies.append(t_elapsed)

        return {
            "vocals": np.array(vocals_list, dtype=np.float32),
            "drums": np.array(drums_list, dtype=np.float32)
        }, latencies

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "parameters": self._model.count_params() if self._model else 0,
            "weights_path": self._weights_path,
            "sample_rate": 11025
        })
        return info


class DummySourceSeparator(BaseSourceSeparator):
    """
    Lightweight mock separator for unit testing, CI, and embedded CPU profiling
    without requiring heavy neural network dependencies.
    """

    def __init__(self, segment_length: int = 75008, latency_simulation: float = 0.01):
        self._segment_length = segment_length
        self._latency = latency_simulation

    @property
    def stem_names(self) -> List[str]:
        return ["vocals", "drums"]

    @property
    def supported_segment_length(self) -> int:
        return self._segment_length

    def warmup(self) -> None:
        pass

    def separate_batch(self, segments: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[float]]:
        if segments.ndim == 2:
            segments = np.expand_dims(segments, axis=-1)

        batch_size = segments.shape[0]
        voc = []
        drm = []
        latencies = []

        for i in range(batch_size):
            t_start = time.perf_counter()
            if self._latency > 0:
                time.sleep(self._latency)

            chunk = segments[i, :, 0]
            # Simple synthetic split: vocals = upper half, drums = lower half
            voc.append(chunk * 0.7)
            drm.append(chunk * 0.5)
            latencies.append(time.perf_counter() - t_start)

        return {
            "vocals": np.array(voc, dtype=np.float32),
            "drums": np.array(drm, dtype=np.float32)
        }, latencies
