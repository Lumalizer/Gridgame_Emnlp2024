import os
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
filename = os.path.join(root, 'assets', 'sample_data_analysis',
                        'vocab_train_2024_04_19_01_37_12graph_SYS__maxlen_6_vocab55_game10.json')


def add_difficulties(filename):
    df = pd.read_json(filename)

    probabilities = []
    target_most_likely = []
    for index, row in df.iterrows():
        rec_labels = row['receiver_labels']
        receiver_probs = [round(p, 2) for p in row['receiver_probs']]
        assert (len(rec_labels) == len(receiver_probs))
        d = [(d, p) for d, p in zip(rec_labels, receiver_probs) if p > 0.01]
        d = sorted(d, key=lambda x: x[1], reverse=True)
        probabilities.append(d)
        target_most_likely.append(row['target'] == d[0][0])

    df['probabilities'] = probabilities

    df['unsure'] = df['receiver_probs'].apply(lambda x: sum([d > 0.9 for d in x]) == 0)
    df['target_most_likely'] = target_most_likely
    df['unsure_and_correct'] = df['unsure'] & df['accuracy']
    df['unsure_and_incorrect'] = df['unsure'] & ~df['accuracy']
    df['sure_and_incorrect'] = ~df['unsure'] & ~df['accuracy']

    # the target is the most likely, but accuaracy is False, or the target is not the most likely, but accuracy is True
    df['misaligned'] = df['target_most_likely'] & ~df['accuracy'] | ~df['target_most_likely'] & df['accuracy']

    return df


df = add_difficulties(filename)
df.drop(columns=['receiver_probs', 'receiver_labels'], inplace=True)
print(df['target_most_likely'].mean(), df['misaligned'].mean(), (df['misaligned'] & ~df['accuracy']).sum())
# df = df[df['accuracy'] == True]
df = df[df['unsure'] == True]
print(df.head(10))
# df.to_csv('difficulties.csv')
