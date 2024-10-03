from egg.core.language_analysis import TopographicSimilarity, Disent
import torch
import json
from egg.core import Interaction
from options import Options
import json
from egg.core.callbacks import ConsoleLogger
import wandb
import warnings
from scipy import stats
import os
from .attributizer import Attributizer
from joblib import Parallel, delayed

try:
    from IPython.core.getipython import get_ipython
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except (ImportError, AttributeError):
    from tqdm import tqdm


warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)


def run_metric_filter(epoch, options: Options):
    return epoch % options.log_interval == 0


def unordered_overlap(seq1, seq2):
    s1, s2 = set(seq1), set(seq2)
    return 1-len(s1.intersection(s2))/max(len(s1), len(s2))


def positional_overlap(seq1, seq2):
    count = 0
    for item1, item2 in zip(seq1, seq2):
        if item1 == item2:
            count += 1
    return 1-(count/max(len(seq1), len(seq2)))


class ResultsCollector(ConsoleLogger):
    def __init__(self, results: list, options: Options):
        super().__init__(True, True)
        self.results = results
        self.options = options
        if options.print_progress:
            self.progress_bar = tqdm(total=options.n_epochs)

    # adapted from egg.core.callbacks.ConsoleLogger
    def aggregate_print(self, loss: float, logs, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        dump.update(dict(mode=mode, epoch=epoch))

        logged = {f"{dump['mode']}/{k}": v for k, v in sorted(dump.items()) if k not in ["mode", "epoch"]}
        logged['epoch'] = epoch
        wandb.log(logged)

        results = json.dumps(dump)
        self.results.append(results)

        if self.options.log_full_pairs and run_metric_filter(epoch, self.options):
            messages = logs.message.argmax(dim=-1)
            messages = [msg.tolist() for msg in messages]
            labels_sender = self.options._labels_sender_stored
            output = {epoch: dict(messages=messages, labels_sender=labels_sender, mode=mode, epoch=epoch)}

            os.makedirs(self.options._target_folder+"/experiments", exist_ok=True)
            filename = self.options._target_folder + f"/experiments/full_pairs_{mode}" + str(self.options) + ".json"
            if not os.path.exists(filename):
                with open(filename, "w") as f:
                    json.dump({}, f)

            with open(filename, "r") as f:
                data = json.load(f)
            with open(filename, "w") as f:
                data.update(output)
                json.dump(data, f)

        if self.options.print_progress:
            if mode == "train":
                self.progress_bar.update(1)
            else:
                mode = "test"

            output_message = ", ".join(
                sorted([f"{k}={round(v, 5) if isinstance(v, float) else v}" for k, v in dump.items() if k != "mode"]))
            output_message = f"mode={mode}: " + output_message
            self.progress_bar.set_description(output_message, refresh=True)


class TopographicSimilarityAtEnd(TopographicSimilarity):
    def __init__(self, options: Options):
        super().__init__('hamming', 'edit', is_gumbel=True, compute_topsim_train_set=True)
        self.options = options
        self.relaxed_distance_fn = lambda s1, s2: 1-len(set(s1).intersection(set(s2)))/max(len(s1), len(s2))
        self.attributizers = self.get_attributizers()

    def on_epoch_end(self, loss: float, logs, epoch: int):
        if run_metric_filter(epoch, self.options):
            super().on_epoch_end(loss, logs, epoch)

    def on_validation_end(self, loss: float, logs, epoch: int):
        if run_metric_filter(epoch, self.options):
            super().on_validation_end(loss, logs, epoch)

    def get_attributizers(self):
        attributizers = {}
        for shape in ['none', 'entangled', 'disentangled']:
            attributizers[shape] = {}
            for position in ['none', 'entangled', 'disentangled', 'double_per_node']:
                if (shape == 'none' and position == 'none'):
                    continue
                attributizers[shape][position] = Attributizer(position_encoding=position, shape_encoding=shape)
        return attributizers

    @staticmethod
    def preprocess_string_for_callback(label: str):
        parts = [(i, p) for i, p in enumerate(label.split('_')) if p != '0']
        return parts

    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:
        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        labels_sender = self.options._labels_sender_stored
        labels_sender = [[self.preprocess_string_for_callback(l) for l in labels] for labels in labels_sender]

        max_elements = min(self.options.topsim_n_samples_max, len(labels_sender))
        ids = torch.randperm(len(labels_sender))[:max_elements]
        messages = messages[ids]
        labels_sender = [labels_sender[i] for i in ids]

        attributizers = self.attributizers
        sender_distance_ = self.sender_input_distance_fn
        message_distance_ = self.message_distance_fn
        compute_topsim = self.compute_topsim

        def do_topsim(attributizer: Attributizer, labels, messages):
            sender_input_distance_fn, message_distance_fn = sender_distance_, message_distance_

            labels = torch.flatten(torch.tensor([[attributizer.process_string(l)
                                   for l in split_labels] for split_labels in labels_sender]), start_dim=1)
            return str(attributizer), compute_topsim(labels, messages, sender_input_distance_fn, message_distance_fn)

        args = []

        for shape in ['entangled', 'disentangled', 'none']:
            for position in ['none', 'entangled', 'disentangled', 'double_per_node']:
                if (shape == 'none' and position == 'none'):
                    continue
                args.append((attributizers[shape][position], labels_sender, messages))

        jobs = self.options.n_jobs
        results = Parallel(n_jobs=jobs)(delayed(do_topsim)(*arg) for arg in args)

        output = {'epoch': epoch}
        for description, topsim in results:
            output[f'{mode}/topsim/{description}'] = topsim

        wandb.log(output)

        self.options._labels_sender_stored = []
        os.makedirs(self.options._target_folder+"/experiments", exist_ok=True)
        with open(self.options._target_folder + "/experiments/topsim_" + str(self.options) + ".json", "a") as f:
            f.write(json.dumps(output) + "\n")


class DisentAtEnd(Disent):
    def __init__(self, options: Options):
        super().__init__(is_gumbel=True, vocab_size=options.vocab_size,
                         compute_bosdis=True, compute_posdis=True, print_train=True)
        self.options = options

    def on_epoch_end(self, loss: float, logs, epoch: int):
        if run_metric_filter(epoch, self.options):
            super().on_epoch_end(loss, logs, epoch)

    def on_validation_end(self, loss: float, logs, epoch: int):
        if run_metric_filter(epoch, self.options):
            super().on_validation_end(loss, logs, epoch)

    def print_message(self, logs: Interaction, tag: str, epoch: int):
        message = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        # sender_input = torch.flatten(logs.aux_input['vectors_sender'], start_dim=1)

        attrs = Attributizer()
        labels_sender = self.options._labels_sender_stored
        labels_sender = torch.tensor([[attrs.process_string(l) for l in labels] for labels in labels_sender])
        sender_input = torch.flatten(labels_sender, start_dim=1)

        posdis = self.posdis(sender_input, message) if self.compute_posdis else None
        bosdis = self.bosdis(sender_input, message, self.vocab_size) if self.compute_bosdis else None
        output = json.dumps(dict(posdis=posdis, bosdis=bosdis, mode=tag, epoch=epoch))

        logged = {"epoch": epoch, f"{tag}/posdis": posdis, f"{tag}/bosdis": bosdis}
        wandb.log(logged)

        os.makedirs(self.options._target_folder+"/experiments", exist_ok=True)
        with open(self.options._target_folder + "/experiments/dissent_" + str(self.options) + ".json", "a") as f:
            f.write(output + "\n")
