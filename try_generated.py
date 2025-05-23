import pandas as pd
import joblib

from utils import get_full_descriptors_for_only_seq_classifier

test_data = pd.read_csv('500_random_sample.csv', index_col=0)
test_data = test_data[(test_data['Length'] <= 96) & (test_data['MFE'] <= -10)]
test_data = test_data.drop(['Length', 'MFE'], axis = 1)
test_data.reset_index(drop=True, inplace=True)



def prepare_seq_features(df: pd.DataFrame, seq_column: str):
    descriptors = get_full_descriptors_for_only_seq_classifier(
        df, seq_column_name=seq_column
    )
    descriptors.insert(0, "sequence", df[seq_column])
    return descriptors

def load_model(path: str):
    return joblib.load(path)

def make_predictions(df: pd.DataFrame, model, output_col: str):
    df[output_col] = model.predict(df.iloc[:, 1:])
    return df

model_seq_only = load_model("classifier\models_code\lgbm_model.pkl")
seq_features = prepare_seq_features(test_data, "Sequence")
seq_features = make_predictions(seq_features, model_seq_only, "seq_only_prediction")

#seq_features.to_csv('final_data_with_features.csv')
final_data = seq_features.iloc[:, [0, 60]]
final_data.to_csv('final_data.csv')



