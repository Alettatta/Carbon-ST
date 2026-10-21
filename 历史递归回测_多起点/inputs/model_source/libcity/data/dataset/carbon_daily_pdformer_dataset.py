"""Daily-frequency adapter for the upstream PDFormer point dataset.

The transformer/model implementation remains upstream PDFormer. This adapter
only replaces the traffic-specific intraday DTW input with one daily series
per province and sets the temporal unit used for daily pattern-key sampling.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm

from libcity.data.dataset.pdformer_dataset import PDFormerDataset


def _fastdtw_pair(task):
    i, j, left, right, radius = task
    distance, _ = fastdtw(left, right, radius=radius)
    return i, j, distance


class CarbonDailyPDFormerDataset(PDFormerDataset):
    def _get_dtw(self):
        """Compute semantic distances over the supplied daily history."""
        cache_path = os.path.join('./libcity/cache/dataset_cache/', 'dtw_' + self.dataset + '_daily.npy')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            return np.load(cache_path)
        arrays = [self._load_dyna(filename) for filename in self.data_files]
        data = np.concatenate(arrays, axis=0)
        # Data is (days, provinces, channels); calculate DTW independently for
        # each province pair using the single carbon-emissions channel.
        distances = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        series = [data[:, node, 0].astype(np.float64) for node in range(self.num_nodes)]
        tasks = [(i, j, series[i], series[j], 6)
                 for i in range(self.num_nodes) for j in range(i, self.num_nodes)]
        workers = int(self.config.get('dtw_num_workers', min(6, os.cpu_count() or 1)))
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, j, distance in tqdm(pool.map(_fastdtw_pair, tasks, chunksize=2),
                                       total=len(tasks), desc='daily DTW'):
                distances[i, j] = distance
                distances[j, i] = distance
        np.save(cache_path, distances)
        return distances

    def __init__(self, config):
        super().__init__(config)
        # PDFormer normally samples patterns in sub-daily steps. For this
        # daily dataset, one pattern step equals one day. The date-derived
        # intraday feature stays enabled as a constant zero column because
        # upstream DataEmbedding expects it before the weekday one-hot fields.
        self.points_per_hour = 1
        self.points_per_day = 1
