import pandas as pd
import joblib
from utils import get_full_descriptors_for_only_seq_classifier


METALS = sorted(["Nd3+","Cu2+","Pb2+","Zn2+","Er3+",
        "Mg2+","Cr3+","M2+-independent","Na+",
        "Sm3+","metal ion dependency not reported",
        "Ce3+","Ag+","Mn2+","Cd2+","Co2+","Ni2+",
        "Ca2+","Tm3+","Gd3+","Mg2+-independent",])


def generate_metal_features(sequences: pd.Series):
    rows = []
    for seq in sequences:
        for metal in METALS:
            row = {"sequence": seq}
            row.update({m: int(m == metal) for m in METALS})
            rows.append(row)
    return pd.DataFrame(rows)


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



def main(
    test_csv_path: str,
    seq_model_path: str,
    seq_metal_model_path: str,
    output_csv_path: str,
    sequence_column: str = "sequence"
):
    
    test_data = pd.read_csv(test_csv_path)
    sequences = test_data[sequence_column]

    # Получение признаков последовательности и металлов
    seq_features = prepare_seq_features(test_data, seq_column=sequence_column)
    metal_features = generate_metal_features(sequences)

    # Объединение признаков последовательности и металлов
    seq_metal_features = pd.merge(
        seq_features, metal_features, on=sequence_column, how="inner"
    )

    # Загрузка моделей
    model_seq_only = load_model(seq_model_path)
    model_seq_metal = load_model(seq_metal_model_path)

   
    seq_features = make_predictions(seq_features, model_seq_only, "seq_only_prediction")
    seq_metal_features = make_predictions(
        seq_metal_features, model_seq_metal, "seq_and_cofactor_prediction"
    )

    # Объединение результатов и сохранение
    final_data = pd.merge(seq_features, seq_metal_features, on=sequence_column, how="inner")

    # Выбор определенных колонок
    selected_columns = final_data.iloc[:, [0, 60] + list(range(120, 142))]
    selected_columns.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main(
        test_csv_path="data/test.csv",
        seq_model_path="classifier/models_code/seq_model.pkl",
        seq_metal_model_path="classifier/models_code/seq_cof_model.pkl",
        output_csv_path="final.csv",
        sequence_column="sequence"
    )
