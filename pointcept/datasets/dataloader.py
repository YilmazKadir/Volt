from functools import partial
import weakref
import torch
import torch.utils.data

import pointcept.utils.comm as comm
from pointcept.datasets.utils import point_collate_fn
from pointcept.datasets import ConcatDataset
from pointcept.utils.env import set_seed


class MultiDatasetDummySampler:
    def __init__(self):
        self.dataloader = None

    def set_epoch(self, epoch):
        if comm.get_world_size() > 1:
            for dataloader in self.dataloader.dataloaders:
                dataloader.sampler.set_epoch(epoch)
        return


class MultiDatasetDataloader:
    """
    Multiple Datasets Dataloader, batch data from a same dataset and mix up ratio determined by loop of each sub dataset.
    The overall length is determined by the main dataset (first) and loop of concat dataset.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        batch_size_per_gpu: int,
        num_worker_per_gpu: int,
        mix_prob=0,
        seed=None,
    ):
        self.datasets = concat_dataset.datasets
        self.ratios = [dataset.loop for dataset in self.datasets]
        # reset data loop, original loop serve as ratios
        for dataset in self.datasets:
            dataset.loop = 1
        # determine union training epoch by main dataset
        self.datasets[0].loop = concat_dataset.loop
        self.dataloaders = []
        for dataset_id, dataset in enumerate(self.datasets):
            if comm.get_world_size() > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None

            init_fn = (
                partial(
                    self._worker_init_fn,
                    dataset_id=dataset_id,
                    num_workers=num_worker_per_gpu,
                    num_datasets=len(self.datasets),
                    rank=comm.get_rank(),
                    seed=seed,
                )
                if seed is not None
                else None
            )
            self.dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size_per_gpu,
                    shuffle=(sampler is None),
                    num_workers=num_worker_per_gpu,
                    sampler=sampler,
                    collate_fn=partial(point_collate_fn, mix_prob=mix_prob),
                    pin_memory=True,
                    worker_init_fn=init_fn,
                    drop_last=True,
                    persistent_workers=True,
                )
            )
        self.sampler = MultiDatasetDummySampler()
        self.sampler.dataloader = weakref.proxy(self)

    def __iter__(self):
        iterator = [iter(dataloader) for dataloader in self.dataloaders]
        while True:
            for i in range(len(self.ratios)):
                for _ in range(self.ratios[i]):
                    try:
                        batch = next(iterator[i])
                    except StopIteration:
                        if i == 0:
                            return
                        else:
                            iterator[i] = iter(self.dataloaders[i])
                            batch = next(iterator[i])
                    yield batch

    def __len__(self):
        main_data_loader_length = len(self.dataloaders[0])
        return (
            main_data_loader_length // self.ratios[0] * sum(self.ratios)
            + main_data_loader_length % self.ratios[0]
        )

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, dataset_id, num_datasets, rank, seed):
        worker_seed = (
            num_workers * num_datasets * rank
            + num_workers * dataset_id
            + worker_id
            + seed
        )
        set_seed(worker_seed)


class RatioShuffleSampler(torch.utils.data.Sampler):
    """
    Ratio-based shuffled sampler for ConcatDataset.

    Assumptions:
        - All sub-dataset loops are 1.
        - Ratios are passed explicitly.
        - Batches are globally shuffled, so they are not homogeneous.
        - Works with DDP by sharding indices across ranks.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        ratios: list,
        batch_size_per_gpu: int,
        seed=0,
    ):
        self.ratios = ratios
        self.seed = seed
        self.epoch = 0

        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        self.dataset_lengths = [len(dataset) for dataset in concat_dataset.datasets]

        assert len(self.ratios) == len(concat_dataset.datasets)
        assert all(dataset.loop == 1 for dataset in concat_dataset.datasets)
        assert len(concat_dataset.data_list) == sum(self.dataset_lengths)

        # Cumulative offsets into ConcatDataset index space.
        # Example: lengths [10, 25, 7] -> offsets [0, 10, 35]
        self.offsets = [0]
        for length in self.dataset_lengths[:-1]:
            self.offsets.append(self.offsets[-1] + length)

        # Main dataset determines epoch length.
        self.main_count = self.dataset_lengths[0] * concat_dataset.loop

        self.num_samples_per_dataset = [
            int(round(self.main_count * ratio / self.ratios[0]))
            for ratio in self.ratios
        ]

        self.global_num_samples = sum(self.num_samples_per_dataset)

        # Drop incomplete global batches.
        self.global_batch_size = self.world_size * batch_size_per_gpu
        self.total_size = (
            self.global_num_samples // self.global_batch_size
        ) * self.global_batch_size

        # Per-rank number of samples. This is divisible by batch_size_per_gpu.
        self.num_samples = self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _sample_dataset(self, dataset_idx, count, generator):
        """
        Sample global ConcatDataset indices from one sub-dataset.

        Uses random permutations without replacement. If count is larger than
        the dataset size, it starts a new random permutation.
        """
        length = self.dataset_lengths[dataset_idx]
        offset = self.offsets[dataset_idx]

        sampled = []

        while len(sampled) < count:
            perm = torch.randperm(length, generator=generator).tolist()
            remaining = count - len(sampled)
            take = min(remaining, length)

            sampled.extend(offset + idx for idx in perm[:take])

        return sampled

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = []

        # Ratio-based sampling from each dataset.
        for dataset_idx, count in enumerate(self.num_samples_per_dataset):
            indices.extend(
                self._sample_dataset(
                    dataset_idx=dataset_idx,
                    count=count,
                    generator=generator,
                )
            )

        # Global shuffle across all datasets.
        # This is what makes batches heterogeneous.
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in order]

        # Drop incomplete global batch.
        indices = indices[: self.total_size]

        # DDP sharding.
        indices = indices[self.rank : self.total_size : self.world_size]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
