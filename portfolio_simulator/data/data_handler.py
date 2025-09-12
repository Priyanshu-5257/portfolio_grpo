import yfinance as yf
import pandas as pd
import os

class DataHandler:
    def __init__(self, assets_filepath, start_date, end_date, data_dir='data'):
        self.assets_filepath = assets_filepath
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.symbols = self._load_symbols()
        self.data = self._load_data()

    def _load_symbols(self):
        with open(self.assets_filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _get_data_filepath(self):
        return os.path.join(self.data_dir, 'market_data.csv')

    def _load_data(self):
        data_filepath = self._get_data_filepath()
        # if os.path.exists(data_filepath):
        #     print(f"Loading data from {data_filepath}...")
        #     data = pd.read_csv(data_filepath, header=[0, 1], index_col=0, parse_dates=True)
        # else:
        print("Downloading new data...")
        data = self._download_data()
        self._save_data(data)
        data = pd.read_csv(data_filepath, header=[0, 1], index_col=0, parse_dates=True)
        ## parse dates
        # data.index = pd.to_datetime(data.index)
        return data

    def _download_data(self):
        print(f"Downloading data for {len(self.symbols)} assets from {self.start_date} to {self.end_date}...")
        df = yf.download(self.symbols, start=self.start_date, end=self.end_date, auto_adjust=True)
        return df['Close']

    def _save_data(self, data):
        os.makedirs(self.data_dir, exist_ok=True)
        data_filepath = self._get_data_filepath()
        print(f"Saving data to {data_filepath}...")
        # data = data.fillna(0)
        data.to_csv(data_filepath)

    def get_data(self):
        return self.data

    def get_latest_data(self, date):
        return self.data.loc[self.data.index <= date].tail(1)

    def stream_data(self):
        for date, row in self.data.iterrows():
            yield date, row
