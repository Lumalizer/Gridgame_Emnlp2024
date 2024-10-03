from dataclasses import dataclass, field
from random import sample
from typing import Literal
from data.graph_builder import GraphBuilder
from torch_geometric.data import Batch
from data.image_builder import ImageBuilder
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize
from options import Options
import torch
from torch.utils.data import Dataset


@dataclass
class ReferenceGameDataset(Dataset):
    options: Options
    mapping: dict[int, str]
    mode: Literal['train', 'test']
    targets: list[int]
    distractors: list[list[int]] = None
    all_labels: list[int] = field(default_factory=list)
    shapes: dict[int, str] = field(default_factory=dict)
    _prebuilt_distractors: bool = False

    truth: torch.Tensor = None
    aux_truths: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.distractors is not None:
            self._prebuilt_distractors = True
            self.distractors = torch.tensor(self.distractors, dtype=torch.long, device=self.options.device)

        if self.options.use_shape_subtasks or self.options.use_position_subtasks:
            self.set_aux_truths()

        self.all_labels = self.mapping.keys()
        self.rebuild_outputs()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        aux_truths = {k: v[idx] for k, v in self.aux_truths.items()}
        return self.sender_input[idx], self.receiver_input[idx], self.truth[idx], aux_truths

    def return_shuffle_truth(self, target: int, distractors: list):
        assert target not in distractors
        input = torch.tensor(distractors + [target], dtype=torch.long)
        indices = torch.randperm(len(input))
        result = input[indices]
        truth = result.tolist().index(target)
        return result, truth

    def build_outputs(self):
        if self._prebuilt_distractors:
            distractors = self.distractors.tolist()
        else:
            distractors = [sample(list(self.all_labels - {target}),
                                  self.options.game_size-1) for target in self.targets]

        games = [self.return_shuffle_truth(t, d) for t, d in zip(self.targets, distractors)]
        receiver_input, truths = zip(*games)
        receiver_input = torch.stack(receiver_input)
        truth = torch.tensor(truths, dtype=torch.long)
        sender_input = receiver_input[torch.arange(len(games)), truth]

        assert sender_input.all() == torch.tensor(self.targets, dtype=torch.long).all()

        return torch.tensor(distractors), sender_input, receiver_input, truth

    def rebuild_outputs(self):
        self.distractors, self.sender_input, self.receiver_input, self.truth = self.build_outputs()

    def get_eval_labels(self, idx):
        targets = [self.targets[i] for i in idx]
        receiver_input = [self.receiver_input[i] for i in idx]

        target_labels = [self.mapping[t] for t in targets]
        receiver_labels = [[self.mapping[d.tolist()] for d in distr] for distr in receiver_input]
        return target_labels, receiver_labels

    def set_aux_truths(self):
        shapeset = set([item for sublist in [l.split("_") for l in self.mapping.values()]
                       for item in sublist if not item == "0"])
        shapeset = {s: i for i, s in enumerate(shapeset)}

        shape1_truths, shape2_truths, pos1_truths, pos2_truths = [], [], [], []

        for target in [self.mapping[t] for t in self.targets]:
            full_graphstring = target.split("_")
            shapes = [t for t in target.split("_") if t != "0"]

            shape1_truths.append(shapeset[shapes[0]])
            shape2_truths.append(shapeset[shapes[1]] if len(shapes) > 1 else -1)
            pos1_truths.append(full_graphstring.index(shapes[0]))
            pos2_truths.append(full_graphstring.index(shapes[1]) if len(shapes) > 1 else -1)

        shape1_truths = torch.tensor(shape1_truths, dtype=torch.long, device=self.options.device)
        shape2_truths = torch.tensor(shape2_truths, dtype=torch.long, device=self.options.device)
        pos1_truth = torch.tensor(pos1_truths, dtype=torch.long, device=self.options.device)
        pos2_truth = torch.tensor(pos2_truths, dtype=torch.long, device=self.options.device)

        self.aux_truths = {"shape1": shape1_truths, "shape2": shape2_truths,
                           "pos1": pos1_truth, "pos2": pos2_truth}


@dataclass
class ShapesPosImgDataset(ReferenceGameDataset):
    def __post_init__(self):
        super().__post_init__()
        ImageBuilder().assure_images()
        self.image_path = f"assets/output/"
        self.resize = Resize((self.options.image_size, self.options.image_size), antialias=True)
        self.data = torch.stack([self.load_image(filename) for filename in self.mapping.values()])

    def load_image(self, filename, suffix=".png"):
        return self.resize(read_image(self.image_path+filename+suffix, mode=ImageReadMode.UNCHANGED).float() / 255.0)

    def __getitem__(self, idx):
        sender_idx, receiver_idx, truth, aux_truths = super().__getitem__(idx)
        return self.data[sender_idx], self.data[receiver_idx], truth, aux_truths


@dataclass
class ShapesPosGraphDataset(ReferenceGameDataset):
    def __post_init__(self):
        super().__post_init__()
        self.builder = GraphBuilder(embedding_size=self.options.embedding_size,
                                    node_encoding=self.options.node_encoding)
        self.data = self.builder.get_batched_data(datastrings=self.mapping.values())

    def __getitem__(self, idx):
        sender_idx, receiver_idx, truth, aux_truths = super().__getitem__(idx)
        sender_data = Batch.from_data_list(self.data.index_select(sender_idx))
        receiver_data = Batch.from_data_list(self.data.index_select(receiver_idx))
        return sender_data, receiver_data, truth, aux_truths
