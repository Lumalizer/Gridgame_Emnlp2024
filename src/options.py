from dataclasses import dataclass, field
import egg.core as core
from datetime import datetime
import torch


@dataclass
class Options:
    experiment: str
    name: str = ""
    model_generation: int = None
    project_name: str = None
    _wandb_name: str = None

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 32
    n_epochs: int = 30
    train_size: float = 0.8
    random_seed: int = 42

    game_size: int = 2
    max_len: int = 4
    vocab_size: int = 100
    embedding_size: int = 50
    hidden_size: int = 80

    num_shapes: int = 51
    num_node_features: int = 52
    num_edge_features: int = 4
    image_size: int = 120

    node_encoding: str = 'functor'  # 'functor', 'edgeattr', 'leaves', 'posattr'

    systemic_distractors: bool = False
    use_shape_distractors_sys: bool = True
    use_position_distractors_sys: bool = True

    use_shape_subtasks: bool = False
    use_position_subtasks: bool = False

    sender_cell: str = 'gru'  # 'rnn', 'gru', or 'lstm'
    length_cost: float = 0.0
    tau_s: float = 1.0
    use_trainable_temperature: bool = True

    analysis_enabled: bool = True
    print_analysis: bool = False
    print_progress: bool = True
    log_full_pairs: bool = False
    log_interval: int = 1
    topsim_n_samples_max: int = 1000
    n_jobs: int = -2

    _labels_sender_stored: list = field(default_factory=list)
    _target_folder: str = ""
    _timestamp: str = ""
    _eval: bool = False

    @property
    def wandb_name(self):
        if not self._wandb_name:
            model_generation = f"gen{self.model_generation}-" if self.model_generation else ""
            rs = f"rs{self.random_seed}-" if self.random_seed != 42 else ""
            self._wandb_name = f"{model_generation}{rs}{self.experiment}-g{self.game_size}-v{self.vocab_size}-l{self.max_len}"
            if not self.experiment == "image":
                self._wandb_name += f"-{self.node_encoding}"
            if self.systemic_distractors:
                self._wandb_name += "-[SYS"
                self._wandb_name += "-Shape" if self.use_shape_distractors_sys else ""
                self._wandb_name += "-Pos" if self.use_position_distractors_sys else ""
                self._wandb_name += "]"
        return self._wandb_name

    @property
    def timestamp(self):
        if self._timestamp == "":
            self._timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return self._timestamp

    @property
    def plotname(self):
        name = f"{self.experiment}"
        name += f"_{self.name}" if self.name else ""
        name += "_SYS" if self.systemic_distractors else ""
        return name

    def to_dict_wandb(self):
        return {k: v for k, v in vars(self).items() if k in self.__dataclass_fields__
                and not k in ['_timestamp', '_target_folder', 'eval', 'device', 'enable_analysis',
                              'print_analysis', 'print_progress', 'n_separated_shapes', 'results']}

    def __post_init__(self):
        out_options = core.init(params=[f'--random_seed={self.random_seed}',
                                        '--lr=1e-3',
                                        f'--batch_size={self.batch_size}',
                                        f'--n_epochs={self.n_epochs}',
                                        f'--vocab_size={self.vocab_size}'])

        for k, v in vars(out_options).items():
            if k not in self.__dataclass_fields__:
                setattr(self, k, v)

    def __str__(self):
        return f"{self.timestamp + self.plotname + '_' }_maxlen_{self.max_len}_vocab{self.vocab_size}_game{self.game_size}"
