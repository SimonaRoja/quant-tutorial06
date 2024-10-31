# IImplementation to backtest different trading strategies
#
# Currently only selling long is supported

from TradingDataHandler import *

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class BacktestingStrategies(TradingDataHandler):
    
    def __init__(self, symbol: str, amount: float, 
                 ftc: float = 0.0, ptc: float = 0.0, 
                 ptrain: float = 0.7, data_source: str = 'http://hilpisch.com/ref_eikon_eod_data.csv', 
                 verbose: int = 3) -> None:
        super().__init__(symbol, amount, ftc, ptc, ptrain, data_source)
        self.result = pd.DataFrame()
        self.models = [GaussianNB(), LogisticRegression(), DecisionTreeClassifier(), SVC(), MLPClassifier()]

    def prepare_model(self, model, features: list) -> pd.DataFrame:
        """Prepares and trains a model; returns normalized test data."""
        train_norm, train_labels, test_norm = self.prepare_train_test_data(features)
        model.fit(train_norm, train_labels)
        return test_norm

    def prepare_train_test_data(self, features: list) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepares the training and test data by normalizing specified features."""
        model_data = self.prepare_features(features)
        split_index = int(len(model_data) * self.ptrain)
        train = model_data.iloc[:split_index].copy()
        test = model_data.iloc[split_index:].copy()
        train_labels = train['direction']
        mu, std = train[self.feature_columns].mean(), train[self.feature_columns].std()
        train_norm = (train[self.feature_columns] - mu) / std
        test_norm = (test[self.feature_columns] - mu) / std
        return train_norm, train_labels, test_norm

    def prepare_features(self, features: list, lags: int = 5) -> pd.DataFrame:
        """Generates lagged features for the model training."""
        model_data = self.data.copy()
        self.feature_columns = features.copy()
        for feature in features:
            for lag in range(1, lags + 1):
                col_name = f'{feature}_lag_{lag}'
                model_data[col_name] = model_data[feature].shift(lag)
                self.feature_columns.append(col_name)
        model_data.dropna(inplace=True)
        return model_data

    def run_strategies(self) -> None:
        """Runs the strategy for each model in the models list."""
        for model in self.models:
            logging.info(f'-----------------------------------------------------------------')
            logging.info(f'Running strategy for {model.__class__.__name__}')
            self.run_strategy(model)
        print(f'--------------- TEST RESULTS -------------------------------')
        print(self.feature_columns)
        print(self.result)
        #print(data[['return']].loc[start_index:].sum().apply(np.exp))
        print(f'--------------- TEST RESULTS -------------------------------')

    def run_strategy(self, model, features=['return'], print_result=False) -> None:
        """Runs the strategy for a single model, handling predictions and order placement."""
        test_norm = self.prepare_model(model, features)
        self.position = 0
        self.trades = 0
        self.units = 0
        self.amount = self.initial_amount

        logging.info("Starting strategy execution...")
        for date_index, row in test_norm.iterrows():
            prediction = model.predict(row.to_frame().T)
            bar = self.data.index.get_loc(date_index)

            if prediction[0] == 1 and self.position <= 0:  # Buy condition
                trading_units = int(self.amount / self.data['price'].iloc[bar])
                self.place_buy_order(bar, units=trading_units)
                self.position = 1
            elif prediction[0] == -1 and self.position >= 0:  # Sell condition, but no short selling
                self.place_buy_order(bar, units=self.units)
                self.position = -1

            if self.get_log_level() == logging.DEBUG:
                self.log_balance(bar)
                self.log_net_wealth(bar)

        end_date, perf = self.close_out(bar)
        new_result = pd.DataFrame({'start_date': [self.data.index[0]], 'end_date': [end_date], 
                                   'model': [model.__class__.__name__], 'performance': [perf]})
        self.result = pd.concat([self.result, new_result], ignore_index=True)
        
        if (print_result):
            print(f'--------------- TEST RESULTS -------------------------------')
            print(self.feature_columns)
            print(new_result)
            print(f'--------------- TEST RESULTS -------------------------------')
            
    def add_features(self, features: str, lags: int = 5) -> pd.DataFrame:
        """Generates additional features for model training."""

        # 1. Log Return with Categories
        #model_data['log_return'] = np.log(model_data['price'] / model_data['price'].shift(1))
        #model_data['log_return_cat'] = pd.qcut(model_data['log_return'], 5, labels=False)  # 5 bins/categories

        # 2. Simple Moving Averages (SMA)
        if feature == 'SMA' or feature == 'difference_SMA':
            short_window = 5
            long_window = 42
            data['SMA_short'] = model_data['price'].rolling(short_window).mean()
            data['SMA_long'] = model_data['price'].rolling(long_window).mean()
            
        if feature == 'difference_SMA':
            data['mom_sma'] = np.where(data['sma_short'] > data['sma_long'], 1, -1)


        # 3. Exponentially Weighted Moving Averages (EWMA)
        #model_data['EWMA_short'] = model_data['price'].ewm(span=self.short_window, adjust=False).mean()
        #model_data['EWMA_long'] = model_data['price'].ewm(span=self.long_window, adjust=False).mean()
        #model_data['EWMA_diff'] = model_data['EWMA_short'] - model_data['EWMA_long']

        # 4. Rolling Volatility
        #model_data['vol_short'] = model_data['log_return'].rolling(window=self.short_window).std()
        #model_data['vol_long'] = model_data['log_return'].rolling(window=self.long_window).std()

        # Generate lagged features
        self.feature_columns = []
        for lag in range(1, lags + 1):
            col_name = f'{feature}_lag_{lag}'
            model_data[col_name] = model_data[feature].shift(lag)
            self.feature_columns.append(col_name)

        # Drop rows with NaN values
        model_data.dropna(inplace=True)
        
        # Add new feature columns
        self.feature_columns += ['log_return_cat', 'SMA_diff', 'EWMA_diff', 'vol_short', 'vol_long']
        
        return model_data
    