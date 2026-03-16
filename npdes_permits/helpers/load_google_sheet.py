
import urllib
import pandas as pd

def load_google_sheet_csv(sheet_id, sheet_name):
    """Load a Google Sheet tab as a CSV DataFrame."""
    url = (f'https://docs.google.com/spreadsheets/d/{sheet_id}'
           f'/gviz/tq?tqx=out:csv&sheet={urllib.parse.quote(sheet_name)}')
    df = pd.read_csv(url, dtype=str)
    # Drop unnamed trailing columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df[df['NPDES_No'] != 'NPDES_No'].reset_index(drop=True)
    return df