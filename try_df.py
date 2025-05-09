import pickle
import pandas as pd
import joblib

from utils import get_full_descriptors_for_only_seq_classifier, sel_features

# Список металлов
metals = sorted(
    [
        "Nd3+",
        "Cu2+",
        "Pb2+",
        "Zn2+",
        "Er3+",
        "Mg2+",
        "Cr3+",
        "M2+-independent",
        "Na+",
        "Sm3+",
        "metal ion dependency not reported",
        "Ce3+",
        "Ag+",
        "Mn2+",
        "Cd2+",
        "Co2+",
        "Ni2+",
        "Ca2+",
        "Tm3+",
        "Gd3+",
        "Mg2+-independent",
    ]
)


def generate_dna_metal_table(seq_column_name):
    rows = []

    for seq in seq_column_name:
        for metal in metals:
            # One-hot кодирование металлов
            metal_one_hot = [1 if m == metal else 0 for m in metals]

            # Формируем строку: {'sequence': ..., 'metal': ..., 'Nd3+': ..., 'UO22+': ..., ...}
            row = {"sequence": seq, "metal": metal}
            row.update(dict(zip(metals, metal_one_hot)))

            rows.append(row)

    # Создаём DataFrame
    df = pd.DataFrame(rows)
    df = df.drop(["metal"], axis=1)
    return df


test_data = pd.read_csv("data/test.csv")
seq_column_name = "sequence"
sequences = test_data["sequence"]

data1 = get_full_descriptors_for_only_seq_classifier(
    df=test_data, seq_column_name=seq_column_name
)
data1.insert(0, "sequence", sequences)
data2 = generate_dna_metal_table(test_data["sequence"])
data2 = pd.merge(data1, data2, on="sequence", how="inner")

# data1.to_csv('data1.csv')
# data2.to_csv('data2.csv')


model_path_1 = "classifier\models_code\seq_model.pkl"
model_path_2 = "classifier\models_code\seq_cof_model.pkl"

pickled_model_1 = joblib.load(model_path_1)
pickled_model_2 = joblib.load(model_path_2)

data1["seq_only_prediction"] = pickled_model_1.predict(data1.iloc[:, 1:])
data2["seq_and_cofactor_prediction"] = pickled_model_2.predict(data2.iloc[:, 1:])
final_data = pd.merge(data1, data2, on="sequence", how="inner")
selected_columns = final_data.iloc[:, [0, 60] + list(range(120, 142))]
selected_columns.to_csv("final.csv")
