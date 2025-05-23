from utils import get_full_descriptors_for_only_seq_classifier
import pandas as pd

data = pd.read_csv('data_with_random_seq.csv', index_col = 0)
sequence_column = 'sequence' 


def prepare_seq_features(df: pd.DataFrame, seq_column: str):
    descriptors = get_full_descriptors_for_only_seq_classifier(
        df, seq_column_name=seq_column
    )
    descriptors.insert(0, "sequence", df[seq_column])
    return descriptors

df = prepare_seq_features(data, sequence_column)
df.insert(1, "target", data['target'])
df.to_csv('data_with_random_featured.csv')
