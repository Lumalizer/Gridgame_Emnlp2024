import wandb
from dotenv import load_dotenv
import logging
import coloredlogs
from experiment.experiments import Experiment, ExperimentGroup
from options import Options
from itertools import product

load_dotenv()
wandb.login()
logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

build_experiment = lambda experiment, game_size, vocab_size, max_len, n_epochs=30, systemic_distractors=False, **kwargs: Experiment(
    Options(experiment=experiment, game_size=game_size, max_len=max_len, vocab_size=vocab_size, n_epochs=n_epochs, systemic_distractors=systemic_distractors, **kwargs))

# "model_generation" will help to set the wandb model name automatically, eg model_generation=6 will result in "gen6-graph-g5-v52-l6"
model_generation = 18
random_seed = [42]
game_size = [5, 3, 10]
vocab_size = [100]
max_len = [1, 2, 3, 4, 5, 6]
n_epochs = 30
exp_name = ['graph']

experiments = []

for rs, gs, vs, ml in product(random_seed, game_size, vocab_size, max_len):
    logging.info(f"add to exp list: gamesize {gs}, vocabsize {vs}, maxlen {ml}")
    exp = [
        build_experiment(
            name, random_seed=rs,
            game_size=gs,
            vocab_size=vs, max_len=ml,
            n_epochs=n_epochs
        )
        for name in exp_name
    ]

    experiments.append(
        ExperimentGroup(
            f'vocab{vs}_len{ml}', wandb_workspace_name='gridgame_emnlp2024',
            model_generation=f'{model_generation}-rs{rs}', experiments=exp
        )
    )




for ex in experiments:
    ex.run()
