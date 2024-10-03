all_experiments = []

experiment = ['edgeattr', 'leaves', 'posattr', 'functor', 'image']
n_epochs = 30
game_size = [2, 5, 20]
vocab_size = [100]
max_len = [2, 4, 6, 10]
random_seed = [1, 2, 3, 4, 5]

base = "python src/RunFromCommandLine.py --model_generation 20 "

for exp in experiment:
    for gs in game_size:
        for vs in vocab_size:
            for ml in max_len:
                for rs in random_seed:
                    exptype = "image" if exp == "image" else "graph"
                    all_experiments.append(
                        f"{base} --random_seed {rs} --experiment {exptype} --game_size {gs} --vocab_size {vs} --max_len {ml} --node_encoding {exp} --n_epochs {n_epochs}")

print("\n".join(all_experiments))
# write to file
with open("experiments.sh", "w") as f:
    f.write("\n".join(all_experiments))
print(len(all_experiments))
