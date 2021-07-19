"""Reformat tables for publication."""
import numpy as np
import pandas as pd


def reformat_term_table(df):
    """Reformat topic-term weight DataFrame into top ten terms Dataframe."""
    df = df.T
    column_names = {c: f"Topic {c}" for c in df.columns}
    df = df.rename(columns=column_names)
    df2 = pd.DataFrame(columns=df.columns, index=np.arange(10))
    df2.index.name = "Term"
    for col in df.columns:
        top_ten_terms = df.sort_values(by=col, ascending=False).index.tolist()[:10]
        df2.loc[:, col] = top_ten_terms

    return df2


if __name__ == "__main__":
    for name in ["table_01", "table_02"]:
        df = pd.read_table(f"tables/{name}.tsv", index_col="topic")
        df2 = reformat_term_table(df)
        df2.to_csv(
            f"tables/{name}_reformatted.tsv",
            index_label="Term",
            sep="\t",
            line_terminator="\n",
        )

    table_sort_columns = {
        "table_03": "r",
        "table_04": "probReverse",
        "table_05": "probReverse",
    }
    for name, sort_column in table_sort_columns.items():
        df = pd.read_table(f"tables/{name}.tsv", index_col="feature")
        df2 = df.reindex(df[sort_column].abs().sort_values(ascending=False).index)
        df2 = df2.iloc[:10]
        df2.to_csv(
            f"tables/{name}_reformatted.tsv",
            index_label="Feature",
            sep="\t",
            line_terminator="\n",
            float_format="%.3f",
        )
