from dataclasses import dataclass, field
import logging
from data.dataloaders import get_dataloaders
from analysis.analyze_experiment import results_to_dataframe
from analysis.plot import plot_dataframe
from options import Options
import pandas as pd
from datetime import datetime
import egg.core as core
import torch
import random
import numpy as np
import wandb
from torch.utils.data import DataLoader
from data.dataloaders import MyDataLoader
from analysis.callbacks import ResultsCollector, TopographicSimilarityAtEnd
from experiment.models.agents import ImageSender, ImageReceiver, GraphSender, GraphReceiver
from experiment.models.emecom_graph.agents import GraphSender as EmecomGraphSender
from experiment.models.emecom_graph.agents import GraphReceiver as EmecomGraphReceiver
from experiment.models.emecom_graph.gs_graph2seq_wrapper import Graph2SeqSenderGS, GraphSeqAlignReceiverGS
from experiment.models.loss_functions import loss_crossentropy, loss_nll


@dataclass
class Experiment:
    options: Options
    model: core.Trainer = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    eval_train: pd.DataFrame = None
    eval_test: pd.DataFrame = None
    train_loader: DataLoader = None
    valid_loader: DataLoader = None

    run_completed: bool = False

    def ensure_determinism(self):
        torch.backends.cudnn.deterministic = True
        random.seed(self.options.random_seed)
        np.random.seed(self.options.random_seed)
        torch.manual_seed(self.options.random_seed)
        torch.cuda.manual_seed_all(self.options.random_seed)
        torch.set_num_threads(1)

    def get_game(self):
        opts = self.options

        if opts.experiment.find('emecom') == -1:
            loss_func = lambda *args, **kwargs: loss_nll(*args, **kwargs, options=opts)
        else:
            loss_func = loss_crossentropy

        if opts.experiment == 'image':
            sender_network, receiver_network = ImageSender, ImageReceiver
            sender_wrapper, receiver_wrapper = core.RnnSenderGS, core.RnnReceiverGS
        elif opts.experiment == 'graph':
            sender_network, receiver_network = GraphSender, GraphReceiver
            sender_wrapper, receiver_wrapper = core.RnnSenderGS, core.RnnReceiverGS
        elif opts.experiment == 'emecom_a2':
            sender_network, receiver_network = EmecomGraphSender,  EmecomGraphReceiver
            sender_wrapper, receiver_wrapper = Graph2SeqSenderGS, GraphSeqAlignReceiverGS
        else:
            raise ValueError(f"Experiment {opts.experiment} not supported")

        sender = sender_network(opts)
        sender = sender_wrapper(sender, opts.vocab_size, opts.embedding_size, opts.hidden_size,
                                max_len=opts.max_len, temperature=opts.tau_s, cell=opts.sender_cell,
                                trainable_temperature=opts.use_trainable_temperature)
        receiver = receiver_network(opts)
        receiver = receiver_wrapper(receiver, opts.vocab_size, opts.embedding_size,
                                    opts.hidden_size, cell=opts.sender_cell)

        game = core.SenderReceiverRnnGS(sender, receiver, loss_func, length_cost=opts.length_cost)
        return game

    def perform_training(self):
        options = self.options
        results = []
        callbacks = [ResultsCollector(results, options)]
        if options.analysis_enabled:
            callbacks.extend([TopographicSimilarityAtEnd(options)])

        trainer = core.Trainer(
            game=self.game,
            optimizer=core.build_optimizer(self.game.parameters()),
            train_data=self.train_loader,
            validation_data=self.valid_loader,
            callbacks=callbacks,
            device=options.device,
        )

        trainer.train(n_epochs=options.n_epochs)
        core.close()
        return '\n'.join(results), trainer

    def run(self):
        if self.run_completed:
            return self.results

        self.ensure_determinism()

        options = self.options

        self.train_loader, self.valid_loader = get_dataloaders(self.options)
        self.game = self.get_game()

        run = wandb.init(project=self.options.project_name, config=options.to_dict_wandb(),
                         name=options.wandb_name, settings=wandb.Settings(_disable_stats=True, _disable_meta=True))
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        results, self.model = self.perform_training()

        if options.analysis_enabled:
            self.eval_train = self.evaluate(self.train_loader)
            self.eval_test = self.evaluate(self.valid_loader)

            run.log({'train_eval': wandb.Table(dataframe=self.eval_train)})
            run.log({'test_eval': wandb.Table(dataframe=self.eval_test)})

        wandb.finish()

        self.results = results_to_dataframe(options, results, self.eval_train, self.eval_test)
        self.run_completed = True
        return self.results

    def evaluate(self, loader: MyDataLoader):
        options = loader.options
        options._eval = True

        def parse_message(message: list[list[float]]):
            return [vocab[word_probs.index(1.0)] for word_probs in message]

        with torch.no_grad():
            _, interaction = self.model.eval(loader)
        interaction: core.Interaction

        vocab = {i: i for i in range(options.vocab_size)}
        message = [parse_message(m) for m in interaction.message.tolist()]
        accuracies = interaction.aux['acc'].tolist()
        ids = interaction.aux_input['ids'].tolist()
        target_labels, receiver_labels = loader.dataset.get_eval_labels(ids)

        if 'receiver_probs' in interaction.aux_input:
            probs = interaction.aux_input['receiver_probs'].tolist()
        else:
            probs = [None] * len(ids)

        return pd.DataFrame({'target': target_labels, 'message': message, 'accuracy': accuracies,
                             'receiver_labels': receiver_labels, 'receiver_probs': probs})


@dataclass
class ExperimentGroup:
    name: str
    wandb_workspace_name: str
    experiments: list[Experiment]
    model_generation: int = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_folder: str = None
    facet_row: str = 'game_size'
    facet_col: str = 'mode'

    def run(self):
        now = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
        self.target_folder = f"results/{self.name}_{now}"

        for i, experiment in enumerate(self.experiments):
            logging.info(f"Running experiment {i+1}/{len(self.experiments)} :: {experiment.options}")
            experiment.options._target_folder = self.target_folder
            experiment.options.project_name = self.wandb_workspace_name
            experiment.options.model_generation = self.model_generation
            experiment.run()

        self.results = pd.concat([e.results for e in self.experiments])
        self.results.to_csv(f"{self.target_folder}/results.csv")
        #self.plot_dataframe()

    def plot_dataframe(self, name=None, mode="both", ):
        plot_dataframe(self.results, name if name else self.name, mode=mode, facet_col=self.facet_col,
                       facet_row=self.facet_row, save_target=self.target_folder, show_plot=False)
