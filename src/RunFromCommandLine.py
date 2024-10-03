import argparse
import wandb
from dotenv import load_dotenv
import logging
import coloredlogs
from experiment.experiments import Experiment, ExperimentGroup
from options import Options


def main(args):
    load_dotenv()
    wandb.login()
    logging.basicConfig(level=logging.INFO)
    coloredlogs.install(level='INFO')

    build_experiment = lambda experiment, game_size, vocab_size, max_len, n_epochs=30, systemic_distractors=False, **kwargs: Experiment(
        Options(experiment=experiment, game_size=game_size, max_len=max_len, vocab_size=vocab_size, n_epochs=n_epochs, systemic_distractors=systemic_distractors, **kwargs))

    experiment = args.experiment
    model_generation = args.model_generation
    game_size = args.game_size
    vocab_size = args.vocab_size
    max_len = args.max_len

    wandb_workspace_name = args.wandb_workspace_name
    random_seed = args.random_seed
    node_encoding = args.node_encoding
    n_epochs = args.n_epochs

    experiment = build_experiment(experiment, game_size=game_size, vocab_size=vocab_size,
                                  max_len=max_len, node_encoding=node_encoding, n_epochs=n_epochs, random_seed=random_seed)

    ex = ExperimentGroup('experiment', wandb_workspace_name=wandb_workspace_name,
                         model_generation=model_generation, experiments=[experiment])
    ex.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--model_generation", type=str, required=True)
    parser.add_argument("--game_size", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--max_len", type=int, required=True)

    parser.add_argument("--wandb_workspace_name", type=str, required=False, default='gridgame_emnlp2024')
    parser.add_argument("--random_seed", type=int, required=False, default=42)
    parser.add_argument("--node_encoding", type=str, required=False, default='functor')
    parser.add_argument("--n_epochs", type=int, required=False, default=30)
    args = parser.parse_args()
    print(args)
    main(args)
