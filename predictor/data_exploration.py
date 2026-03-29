import pandas as pd


def dataset_exploration(df: pd.DataFrame) -> str:
    return df.head().to_html(
        classes='table table-bordered table-striped table-sm',
        float_format='%.2f',
        justify='center',
        index=False,
    )


def data_exploration(df: pd.DataFrame) -> str:
    return df.describe(include='all').to_html(
        classes='table table-bordered table-striped table-sm',
        float_format='%.2f',
        justify='center',
    )
