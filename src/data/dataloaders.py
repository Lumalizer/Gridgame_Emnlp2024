from random import sample
from options import Options
import torch
from data.datasets import ShapesPosGraphDataset, ShapesPosImgDataset
from torch.utils.data import DataLoader
from data.graphstring_builder import GraphstringBuilder
from data.systemic_distractors import SystematicDistractors
from typing import Union


class MyDataLoader(DataLoader):
    def __init__(self, dataset, shuffle=True, **kwargs):
        self.dataset: Union[ShapesPosGraphDataset, ShapesPosImgDataset] = dataset
        self.options: Options = self.dataset.options
        super().__init__(dataset, batch_size=dataset.options.batch_size, shuffle=shuffle, **kwargs)
        self.shuffle = shuffle

    def __iter__(self):
        return SingleBatch(self)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SingleBatch:
    def __init__(self, dataloader: 'MyDataLoader'):
        self.dataloader = dataloader
        self.batch_idx = 0
        self.options = dataloader.options

        self.selections = list(range(len(self.dataloader.dataset)))

        if not self.options._eval:
            self.selections = sample(self.selections, k=len(self.selections))

        self.options._labels_sender_stored = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.selections) < self.options.batch_size:
            if not self.options._eval:
                self.dataloader.dataset.rebuild_outputs()
            raise StopIteration()

        ids = self.selections[:self.options.batch_size]
        self.selections = self.selections[self.options.batch_size:]

        sender, receiver, truth, aux_truths = self.dataloader[ids]
        aux_input = {'data_sender': sender, 'y': truth, 'data_receiver': receiver, 'ids': torch.tensor(ids)}
        aux_input.update(aux_truths)

        if not self.options._eval:
            self.register_labels_for_callbacks(ids)

        self.batch_idx += 1
        return None, truth, None, aux_input

    def renew_selections(self):
        self.selections = sample(list(range(len(self.dataloader.dataset))), k=len(self.dataloader.dataset))

    def register_labels_for_callbacks(self, elements):
        sender_games, _, _, _ = super(type(self.dataloader.dataset),
                                      self.dataloader.dataset).__getitem__(elements)
        mapping = self.dataloader.dataset.mapping
        for i in range(self.options.batch_size):
            idx = sender_games[i].tolist()
            if isinstance(idx, int):
                idx = [idx]
            self.options._labels_sender_stored.append([mapping[int(k)] for k in idx])


def split_get_datasets(options: Options):
    mapping = {i: s for i, s in enumerate(GraphstringBuilder().datastrings)}
    dataset_class = ShapesPosGraphDataset if options.experiment != 'image' else ShapesPosImgDataset

    all_labels = list(mapping.keys())
    train_targets = all_labels[:int(len(all_labels)*options.train_size)]
    test_targets = all_labels[int(len(all_labels)*options.train_size):]

    train_targets = sample(train_targets, len(train_targets))
    test_targets = sample(test_targets, len(test_targets))

    train_distractors, test_distractors = None, None

    if options.systemic_distractors:
        shapes = set([item for sublist in [l.split("_") for l in mapping.values()] for item in sublist])
        shapes.remove('0')
        strain = SystematicDistractors(
            shapes,
            False,
            excluded_graphstrings=[mapping[i] for i in test_targets],
            n_distractors=options.game_size-1,
            use_shape_distractors=options.use_shape_distractors_sys,
            use_position_distractors=options.use_position_distractors_sys)

        stest = SystematicDistractors(
            shapes,
            False,
            excluded_graphstrings=[mapping[i] for i in train_targets],
            n_distractors=options.game_size-1,
            use_shape_distractors=options.use_shape_distractors_sys,
            use_position_distractors=options.use_position_distractors_sys)

        reverse_mapping = {s: i for i, s in mapping.items()}
        train_targets = [reverse_mapping[s] for s in strain.targets]
        test_targets = [reverse_mapping[s] for s in stest.targets]
        train_distractors = [[reverse_mapping[s] for s in distr] for distr in strain.distractors]
        test_distractors = [[reverse_mapping[s] for s in distr] for distr in stest.distractors]

    train = dataset_class(options, mapping, 'train', train_targets, train_distractors)
    test = dataset_class(options, mapping, 'test', test_targets, test_distractors)
    return train, test


def get_dataloaders(options: Options):
    train_set, test_set = split_get_datasets(options)
    train_loader = MyDataLoader(train_set, shuffle=False, drop_last=True)
    test_loader = MyDataLoader(test_set, shuffle=False, drop_last=True)
    return train_loader, test_loader
