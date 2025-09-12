# file: portfolio_simulator/rl_agent/environment.py

import numpy as np
import pandas as pd
import random
import math
from ..data.data_handler import DataHandler
from ..portfolio.portfolio_manager import PortfolioManager
from ..execution.execution_handler import ExecutionHandler

class PortfolioEnv:
    """
    The core single-agent portfolio simulation environment.
    This class is now fully compatible with being deep-copied for group simulations.
    """
    def __init__(self, assets_filepath, start_date, end_date, initial_cash=100000.0, lookback_window=60, randomize_start=True, min_episode_days=250):
        # --- (No changes to the __init__ method) ---
        self.assets_filepath = assets_filepath
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.lookback_window = lookback_window
        self.randomize_start = randomize_start
        self.min_episode_days = min_episode_days

        self.data_handler = DataHandler(self.assets_filepath, self.start_date, self.end_date)
        self.symbols = self.data_handler.symbols
        self.portfolio_manager = PortfolioManager(self.symbols, self.initial_cash)
        self.execution_handler = ExecutionHandler()

        self.available_dates = self.data_handler.data.index.tolist()
        self.current_episode_start_idx = 0

        self.action_space_dim = len(self.symbols)
        self.state_space_dim = self._calculate_state_space_dim()

        self.episode_start_date = None
        self.episode_start_value = initial_cash
        self.peak_portfolio_value = initial_cash
        self.previous_portfolio_value = initial_cash
        self.daily_returns = []
        self.reward_log = []

        # Do not call reset in __init__ to allow for clean cloning
        self.max_days = 60
        self.data_stream = None
        self.current_date = None
        self.current_prices = None
        self.done = True # Start in a 'done' state until reset is called

    def _calculate_state_space_dim(self):
        return 1 + len(self.symbols) + (len(self.symbols) * self.lookback_window)

    def reset(self, episode_start_idx=None):
        self.portfolio_manager = PortfolioManager(self.symbols, self.initial_cash)
        action_columns = [f'action_{symbol}' for symbol in self.symbols]
        columns = ['value'] + action_columns
        self.portfolio_manager.portfolio_value = pd.DataFrame(columns=columns)

        if episode_start_idx is not None:
            self.current_episode_start_idx = episode_start_idx
        elif self.randomize_start:
            max_start_idx = len(self.available_dates) - self.min_episode_days - self.lookback_window
            self.current_episode_start_idx = random.randint(self.lookback_window, max(self.lookback_window, max_start_idx))
        else:
            self.current_episode_start_idx = self.lookback_window
        self.current_step_index = self.current_episode_start_idx
        self.done = False
        
        self.episode_start_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.previous_portfolio_value = self.initial_cash
        self.daily_returns = []
        self.reward_log = []
        self.prev_reward = None
        first_date = self.available_dates[self.current_step_index]
        self.current_date = first_date
        self.current_prices = self.data_handler.data.loc[first_date]

        self.episode_start_date = self.current_date
        self.portfolio_manager.record_portfolio_value(self.current_date, self.current_prices)
        self._record_actions([0.0] * len(self.symbols))

        return self._get_state()
    
    def _get_state(self):
        # --- (No changes to this method) ---
        if not self.portfolio_manager.portfolio_value.empty:
            total_value = self.portfolio_manager.portfolio_value['value'].iloc[-1]
        else:
            total_value = self.initial_cash

        if total_value <= 1.0 or not np.isfinite(total_value):
            total_value = 1.0

        cash_balance = self.portfolio_manager.get_cash_balance()
        normalized_cash = np.clip(cash_balance / total_value, 0, 1)

        holdings = self.portfolio_manager.get_holdings()
        holdings_values = np.array([
            holdings.get(symbol, 0) * (self._get_current_price(symbol) or 0)
            for symbol in self.symbols
        ])
        normalized_holdings = np.clip(holdings_values / total_value, 0, 1)

        end_idx = self.data_handler.data.index.get_loc(self.current_date)
        start_idx = max(0, end_idx - self.lookback_window + 1)
        lookback_data = self.data_handler.data.iloc[start_idx:end_idx+1]
        
        first_day_prices = lookback_data.iloc[0].replace(0, 1)
        normalized_lookback = lookback_data.div(first_day_prices).sub(1)
        
        normalized_prices = normalized_lookback.T.fillna(0).values.flatten()
        
        state = np.concatenate(([normalized_cash], normalized_holdings, normalized_prices))
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)


    def step(self, action_vector):
        # --- (No changes to this method) ---
        if self.done: return self._get_state(), 0, self.done, {}

        self._execute_trades(action_vector)

        self.current_step_index += 1

        # Check if we are out of data
        if self.current_step_index >= len(self.available_dates):
            self.done = True
        else:
            # Get the next day's data
            self.current_date = self.available_dates[self.current_step_index]
            self.current_prices = self.data_handler.data.loc[self.current_date]
            self.portfolio_manager.record_portfolio_value(self.current_date, self.current_prices)
            self._record_actions(action_vector)

        current_portfolio_value = self.portfolio_manager.portfolio_value['value'].iloc[-1]
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_portfolio_value)
        
        days_elapsed = len(self.portfolio_manager.portfolio_value) - 1
        if days_elapsed >= self.max_days: self.done = True
        
        reward, reward_breakdown = self._calculate_terminal_reward(current_portfolio_value) if self.done else self._calculate_daily_reward(current_portfolio_value)
        self._log_reward_breakdown(reward_breakdown)

        info = {}
        if self.done:
            info = {
                'final_portfolio_value': current_portfolio_value,
                'episode_start_date': self.episode_start_date,
                'current_date': self.current_date,
                'history': self.portfolio_manager.portfolio_value.copy()
            }
        return self._get_state(), reward, self.done, info

    def _execute_trades(self, action_vector):
        # --- (No changes to this method) ---
        action_threshold, min_trade_value, max_position_size, cash_holding_target = 0.1, 100.0, 0.20, 0.05
        
        sells = [{'symbol': self.symbols[i], 'strength': abs(action)} for i, action in enumerate(action_vector) if action < -action_threshold]
        buys = [{'symbol': self.symbols[i], 'strength': action} for i, action in enumerate(action_vector) if action > action_threshold]

        sells.sort(key=lambda x: x['strength'], reverse=True)
        for sell_info in sells:
            symbol, price = sell_info['symbol'], self._get_current_price(sell_info['symbol'])
            if price is None: continue
            
            quantity_to_sell = self.portfolio_manager.holdings.get(symbol, 0) * sell_info['strength']
            if (quantity_to_sell * price) > min_trade_value:
                signal = {'symbol': symbol, 'action': 'SELL', 'quantity': quantity_to_sell}
                if transaction := self.execution_handler.execute_order(signal, self.current_prices):
                    self.portfolio_manager.update_holdings(symbol, -transaction['quantity'], transaction['price'])

        total_portfolio_value = self.portfolio_manager.get_total_portfolio_value(self.current_prices)
        cash_for_buying = self.portfolio_manager.get_cash_balance() - (total_portfolio_value * cash_holding_target)
        total_buy_strength = sum(buy['strength'] for buy in buys)

        if total_buy_strength > 0 and cash_for_buying > min_trade_value:
            buys.sort(key=lambda x: x['strength'], reverse=True)
            cash_spent_this_step = 0
            for buy_info in buys:
                cash_to_spend = cash_for_buying - cash_spent_this_step
                if cash_to_spend <= min_trade_value: break

                symbol, price = buy_info['symbol'], self._get_current_price(buy_info['symbol'])
                if price is None: continue

                ideal_alloc = cash_to_spend * (buy_info['strength'] / total_buy_strength)
                current_val = self.portfolio_manager.holdings.get(symbol, 0) * price
                max_add_inv = (total_portfolio_value * max_position_size) - current_val
                
                cash_to_allocate = min(ideal_alloc, max_add_inv)
                if cash_to_allocate > min_trade_value:
                    signal = {'symbol': symbol, 'action': 'BUY', 'quantity': cash_to_allocate / price}
                    if transaction := self.execution_handler.execute_order(signal, self.current_prices):
                        self.portfolio_manager.update_holdings(symbol, transaction['quantity'], transaction['price'])
                        cash_spent_this_step += transaction['cost']

    def _get_current_price(self, symbol):
        # --- (No changes to this method) ---
        if (price_series := self.current_prices.get(symbol)) is not None and pd.notna(price := price_series.item()) and price > 1e-6:
            return price
        return None

    def _record_actions(self, action_vector):
        # --- (No changes to this method) ---
        if not self.portfolio_manager.portfolio_value.empty:
            last_idx = self.portfolio_manager.portfolio_value.index[-1]
            for i, symbol in enumerate(self.symbols):
                self.portfolio_manager.portfolio_value.loc[last_idx, f'action_{symbol}'] = action_vector[i]
    
    def _calculate_daily_reward(self, portfolio_value_today):
        # --- (No changes to this method) ---
        # daily_log_return = 0.0
        # if self.previous_portfolio_value > 1e-6:
        #     return_ratio = np.clip(portfolio_value_today / self.previous_portfolio_value, 0.1, 10.0)
        #     daily_log_return = math.log(return_ratio)
        
        # reward is the area under the curve of current portfolio value vs days elapsed
        current_height = portfolio_value_today - self.initial_cash
        days_elapsed = 1 # since we are calculating daily reward
        area_da = current_height * days_elapsed
        if self.prev_reward is None:
            daily_log_return = area_da
            self.prev_reward = daily_log_return
        else:
            self.prev_reward += area_da
            daily_log_return = self.prev_reward
        daily_log_return = daily_log_return/self.initial_cash

        self.daily_returns.append((portfolio_value_today / self.previous_portfolio_value) -1)
        self.previous_portfolio_value = portfolio_value_today
        reward_breakdown = {'daily_log_return': daily_log_return, 'total': daily_log_return}
        # return daily_log_return, reward_breakdown
        return 0.1, reward_breakdown

    def _calculate_terminal_reward(self, portfolio_value_final):
        # --- (No changes to this method) ---
        volatility = np.std(self.daily_returns) * np.sqrt(self.max_days) if len(self.daily_returns) > 1 else 0.0
        days_elapsed = max(1, len(self.portfolio_manager.portfolio_value))
        portfolio_return = (portfolio_value_final / self.initial_cash) - 1
        annualized_return = (1 + portfolio_return)**(self.max_days / days_elapsed) - 1
        sharpe_ratio = annualized_return / (volatility + 1e-8)
        capped_sharpe = sharpe_ratio #np.clip(sharpe_ratio, -3.0, 3.0)
        reward_breakdown = {'annualized_return': annualized_return, 'volatility': volatility, 'sharpe_ratio': sharpe_ratio, 'total': capped_sharpe}
        # return capped_sharpe, reward_breakdown
        return portfolio_return, reward_breakdown

    def _log_reward_breakdown(self, reward_breakdown):
        # --- (No changes to this method) ---
        log_entry = {'date': self.current_date, 'portfolio_value': self.portfolio_manager.portfolio_value['value'].iloc[-1], **reward_breakdown}
        self.reward_log.append(log_entry)