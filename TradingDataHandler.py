# Implement basic trading functionalities

from typing import Optional, Union
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging


class TradingDataHandler:
    """
    Base class for event-based backtesting of trading strategies.
    
    Attributes
    ----------
    symbol : str
        Financial instrument to be used (ticker symbol).
    amount : float
        Initial amount to be invested either once or per trade.
    ftc : float
        Fixed transaction costs per trade (buy or sell).
    ptc : float
        Proportional transaction costs per trade (buy or sell).
    ptrain : float
        Training data percentage vs. test data.
    data_source : str
        URL or file path to the data file.
    """
    
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, symbol: str, amount: float, 
                 ftc: float = 0.0, ptc: float = 0.0, 
                 ptrain: float = 0.7, data_source: str = 'http://hilpisch.com/ref_eikon_eod_data.csv') -> None:
        self.symbol = symbol
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.ptrain = ptrain
        self.data_source = data_source
        self.units = 0
        self.position = 0
        self.trades = 0
        self.get_data()
        
    def get_data(self) -> None:
        """Retrieves and prepares the data from a source (URL or local file)."""
        try:
            # Try loading data from URL or file
            if self.data_source.startswith("http"):
                raw = pd.read_csv(self.data_source, index_col=0, parse_dates=True).dropna()
            else:
                raw = pd.read_csv(self.data_source, index_col=0, parse_dates=True).dropna()
            
            self.data = pd.DataFrame(raw[self.symbol])
            self.data.rename(columns={self.symbol: 'price'}, inplace=True)
            self.data['return'] = np.log(self.data['price'] / self.data['price'].shift(1))
            self.data['direction'] = np.where(self.data['return'] > 0, 1, -1)
            self.data.dropna(inplace=True)
            logging.info("Data successfully loaded and prepared.")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")

    def plot_data(self, cols: Optional[list] = None, title: str = None, figsize: tuple = (10, 6)) -> None:
        """Plots specified data columns."""
        if cols is None:
            cols = ['price']
        self.data[cols].plot(figsize=figsize, title=title or self.symbol)
        plt.show()

    def get_date_price(self, bar: int) -> tuple[str, float]:
        """Returns the date and price for a given bar."""
        date = str(self.data.index[bar])[:10]
        price = self.data['price'].iloc[bar]
        return date, price

    def log_balance(self, bar: int) -> None:
        """Logs the current balance."""
        date, price = self.get_date_price(bar)
        logging.info(f"{date} | current balance: ${self.amount:.2f}")

    def log_net_wealth(self, bar: int) -> None:
        """Logs the current net wealth."""
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        logging.info(f"{date} | current net wealth: ${net_wealth:.2f}")

    def place_order(self, bar: int, units: int, is_buy: bool = True) -> None:
        """Places a buy or sell order."""
        date, price = self.get_date_price(bar)
        transaction_cost = (units * price) * (1 + self.ptc) + self.ftc if is_buy else (units * price) * (1 - self.ptc) - self.ftc
        self.amount += -transaction_cost if is_buy else transaction_cost
        self.units += units if is_buy else -units
        self.trades += 1
        order_type = 'buying' if is_buy else 'selling'
        logging.info(f"{date} | {order_type} {units} units at ${price:.2f}")
        if self.get_log_level() == logging.DEBUG:
            self.log_balance(bar)
            self.log_net_wealth(bar)
            
    def place_buy_order(self, bar: int, units: int):
        self.place_order(bar, units)
        
    def place_buy_order(self, bar: int, units: int):
        self.place_order(bar, units, False)

    def close_out(self, bar: int) -> tuple[str, float]:
        """Closes out the position, calculating and logging final balance and performance."""
        date, price = self.get_date_price(bar)
        self.amount += self.units * price  # Account for remaining units
        logging.info(f"{date} | Final inventory: {self.units} units at ${price:.2f}")
        performance = ((self.amount - self.initial_amount) / self.initial_amount) * 100
        logging.info(f"Final balance: ${self.amount:.2f}")
        logging.info(f"Net Performance [%]: {performance:.2f}")
        return date, performance


    def get_log_level(self) -> int:
        """Returns the current logging level as a logging constant (e.g., logging.DEBUG)."""
        logger = logging.getLogger()
        if logger.isEnabledFor(logging.DEBUG):
            return logging.DEBUG
        elif logger.isEnabledFor(logging.INFO):
            return logging.INFO
        elif logger.isEnabledFor(logging.WARNING):
            return logging.WARNING
        elif logger.isEnabledFor(logging.ERROR):
            return logging.ERROR
        else:
            return logging.NOTSET