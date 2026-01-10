"""
Performance monitoring for ChunIVision.

Tracks:
- FPS (frames per second)
- Latency at each pipeline stage
- Memory usage
- Dropped frames
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque
import time
import psutil
import numpy as np


@dataclass
class PerformanceStats:
    """
    Performance monitoring statistics.

    Attributes:
        fps: Current frames per second
        avg_latency_ms: Average total latency in milliseconds
        stage_latencies: Dictionary mapping stage name to average latency (ms)
        memory_usage_mb: Current memory usage in megabytes
        dropped_frames: Number of dropped frames
    """
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    dropped_frames: int = 0

    def __str__(self) -> str:
        """String representation of performance stats."""
        stage_str = ", ".join([f"{k}: {v:.2f}ms" for k, v in self.stage_latencies.items()])
        return (
            f"FPS: {self.fps:.1f}, "
            f"Latency: {self.avg_latency_ms:.2f}ms, "
            f"Memory: {self.memory_usage_mb:.1f}MB, "
            f"Dropped: {self.dropped_frames}, "
            f"Stages: [{stage_str}]"
        )


class PerformanceMonitor:
    """
    Tracks system performance metrics.

    Monitors:
    - Frame rate (FPS)
    - Pipeline stage latencies
    - Memory consumption
    - Dropped frames
    """

    def __init__(
        self,
        fps_window: int = 60,
        latency_window: int = 100,
        enable_memory_tracking: bool = True
    ):
        """
        Initialize performance monitor.

        Args:
            fps_window: Number of frames to use for FPS calculation
            latency_window: Number of samples to use for latency averaging
            enable_memory_tracking: Whether to track memory usage
        """
        self.fps_window = fps_window
        self.latency_window = latency_window
        self.enable_memory_tracking = enable_memory_tracking

        # Frame timing
        self._frame_times: deque = deque(maxlen=fps_window)
        self._last_frame_time: Optional[float] = None

        # Latency tracking (per stage)
        self._stage_latencies: Dict[str, deque] = {}

        # Dropped frames
        self._dropped_frames = 0

        # Process handle for memory tracking
        if self.enable_memory_tracking:
            try:
                self._process = psutil.Process()
            except:
                self._process = None
                self.enable_memory_tracking = False
        else:
            self._process = None

        # Total frames processed
        self._total_frames = 0

        # Start time
        self._start_time = time.time()

    def record_frame(self) -> None:
        """
        Record frame processing completion.

        Call this at the end of each frame processing cycle.
        """
        current_time = time.time()

        if self._last_frame_time is not None:
            frame_time = current_time - self._last_frame_time
            self._frame_times.append(frame_time)

        self._last_frame_time = current_time
        self._total_frames += 1

    def record_latency(self, stage: str, latency_ms: float) -> None:
        """
        Record latency for a pipeline stage.

        Args:
            stage: Name of the pipeline stage
            latency_ms: Latency in milliseconds
        """
        if stage not in self._stage_latencies:
            self._stage_latencies[stage] = deque(maxlen=self.latency_window)

        self._stage_latencies[stage].append(latency_ms)

    def record_dropped_frame(self) -> None:
        """Record a dropped frame."""
        self._dropped_frames += 1

    def get_fps(self) -> float:
        """
        Get current FPS.

        Returns:
            Frames per second
        """
        if len(self._frame_times) < 2:
            return 0.0

        # Calculate FPS from average frame time
        avg_frame_time = np.mean(self._frame_times)
        if avg_frame_time > 0:
            return float(1.0 / avg_frame_time)
        return 0.0

    def get_average_latency(self) -> float:
        """
        Get average total latency across all stages.

        Returns:
            Average latency in milliseconds
        """
        if not self._stage_latencies:
            return 0.0

        # Sum average latency from each stage
        total = 0.0
        for latencies in self._stage_latencies.values():
            if latencies:
                total += np.mean(latencies)

        return float(total)

    def get_stage_latencies(self) -> Dict[str, float]:
        """
        Get average latency for each pipeline stage.

        Returns:
            Dictionary mapping stage name to average latency (ms)
        """
        result = {}
        for stage, latencies in self._stage_latencies.items():
            if latencies:
                result[stage] = float(np.mean(latencies))
            else:
                result[stage] = 0.0

        return result

    def get_memory_usage(self) -> float:
        """
        Get current memory usage.

        Returns:
            Memory usage in megabytes
        """
        if not self.enable_memory_tracking or self._process is None:
            return 0.0

        try:
            mem_info = self._process.memory_info()
            return mem_info.rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0

    def get_stats(self) -> PerformanceStats:
        """
        Get current performance statistics.

        Returns:
            PerformanceStats object with all metrics
        """
        return PerformanceStats(
            fps=self.get_fps(),
            avg_latency_ms=self.get_average_latency(),
            stage_latencies=self.get_stage_latencies(),
            memory_usage_mb=self.get_memory_usage(),
            dropped_frames=self._dropped_frames
        )

    def get_detailed_stats(self) -> Dict:
        """
        Get detailed performance statistics.

        Returns:
            Dictionary with detailed metrics
        """
        uptime = time.time() - self._start_time

        return {
            'fps': self.get_fps(),
            'avg_latency_ms': self.get_average_latency(),
            'stage_latencies': self.get_stage_latencies(),
            'memory_mb': self.get_memory_usage(),
            'dropped_frames': self._dropped_frames,
            'total_frames': self._total_frames,
            'uptime_seconds': uptime,
            'avg_fps': self._total_frames / uptime if uptime > 0 else 0.0,
            'drop_rate': self._dropped_frames / max(1, self._total_frames)
        }

    def reset(self) -> None:
        """Reset all performance counters."""
        self._frame_times.clear()
        self._last_frame_time = None
        self._stage_latencies.clear()
        self._dropped_frames = 0
        self._total_frames = 0
        self._start_time = time.time()

    def print_stats(self) -> None:
        """Print current performance statistics to console."""
        stats = self.get_stats()
        print(f"\n{'='*70}")
        print(f"Performance Statistics:")
        print(f"{'='*70}")
        print(f"  FPS:              {stats.fps:>8.1f}")
        print(f"  Avg Latency:      {stats.avg_latency_ms:>8.2f} ms")
        print(f"  Memory Usage:     {stats.memory_usage_mb:>8.1f} MB")
        print(f"  Dropped Frames:   {stats.dropped_frames:>8d}")

        if stats.stage_latencies:
            print(f"\n  Stage Latencies:")
            for stage, latency in stats.stage_latencies.items():
                print(f"    {stage:<20s}: {latency:>6.2f} ms")

        print(f"{'='*70}\n")

    def export_to_file(self, filepath: str) -> None:
        """
        Export performance statistics to file.

        Args:
            filepath: Path to output file
        """
        stats = self.get_detailed_stats()

        with open(filepath, 'w') as f:
            f.write("ChunIVision Performance Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Uptime: {stats['uptime_seconds']:.2f} seconds\n")
            f.write(f"Total Frames: {stats['total_frames']}\n")
            f.write(f"Dropped Frames: {stats['dropped_frames']}\n")
            f.write(f"Drop Rate: {stats['drop_rate']*100:.2f}%\n\n")

            f.write(f"Current FPS: {stats['fps']:.1f}\n")
            f.write(f"Average FPS: {stats['avg_fps']:.1f}\n")
            f.write(f"Average Latency: {stats['avg_latency_ms']:.2f} ms\n")
            f.write(f"Memory Usage: {stats['memory_mb']:.1f} MB\n\n")

            if stats['stage_latencies']:
                f.write("Stage Latencies:\n")
                for stage, latency in stats['stage_latencies'].items():
                    f.write(f"  {stage}: {latency:.2f} ms\n")

    def __str__(self) -> str:
        """String representation showing current stats."""
        return str(self.get_stats())
