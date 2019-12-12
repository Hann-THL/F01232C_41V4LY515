import lib._util.scaler as scaler

import numpy as np
import pandas as pd

class ForexEnv:
    def __init__(self, source_path, filename, nrows=None, train_size=.7, train=True, random_size=.8, stack_size=5, timeframe=5):
        self.timeframe = timeframe
        self.__train_test_split(source_path, filename, nrows=nrows, train_size=train_size, train=train)

        self.random_range = int(len(self.indexes) * random_size)
        self.stack_size   = stack_size
        
    def __train_test_split(self, source_path, filename, chunk_size=50_000, nrows=None, train_size=.7, train=True):
        source_file = f'{source_path}{filename}'
        df_chunks = pd.read_csv(source_file, sep=',',
                                header=None, names=['datetime', 'bid', 'ask', 'vol'],
                                usecols=['datetime', 'bid', 'ask'],
                                parse_dates=['datetime'],
                                date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d %H%M%S%f"),
                                chunksize=chunk_size, nrows=nrows)
        timeseries_df = pd.concat(df_chunks)
        
        # Convert tick data to OHLC
        bid_df = timeseries_df.set_index('datetime')['bid'].resample(f'{self.timeframe}Min').ohlc().reset_index()
        ask_df = timeseries_df.set_index('datetime')['ask'].resample(f'{self.timeframe}Min').ohlc().reset_index()
        
        row_count  = len(bid_df) if nrows is None else nrows
        split_size = round(row_count * train_size)
        
        if train:
            bid_df = bid_df[:split_size]
            ask_df = ask_df[:split_size]
        else:
            bid_df = bid_df[split_size:]
            ask_df = ask_df[split_size:]
        
        self.indexes   = bid_df.index.values
        self.datetimes = bid_df['datetime'].values
        
        self.open_bids = np.round(bid_df['open'].values, 5)
        self.high_bids = np.round(bid_df['high'].values, 5)
        self.low_bids  = np.round(bid_df['low'].values, 5)
        self.bids      = np.round(bid_df['close'].values, 5)
        
        self.open_asks = np.round(ask_df['open'].values, 5)
        self.high_asks = np.round(ask_df['high'].values, 5)
        self.low_asks  = np.round(ask_df['low'].values, 5)
        self.asks      = np.round(ask_df['close'].values, 5)

        # Reference: https://www.youtube.com/watch?v=WZbOeFsSirM
        # Calculate RSI
        tmp_df = pd.DataFrame({
            'datetime': self.datetimes,

            'open_bid': self.open_bids,
            'high_bid': self.high_bids,
            'low_bid': self.low_bids,
            'bid': self.bids,

            'open_ask': self.open_asks,
            'high_ask': self.high_asks,
            'low_ask': self.low_asks,
            'ask': self.asks
        })

        n_timestep = 14
        tmp_df['bid_movement'] = tmp_df['bid'].diff()
        tmp_df['ask_movement'] = tmp_df['ask'].diff()

        tmp_df['bid_upward_movement']   = np.where(tmp_df['bid_movement'] > 0, tmp_df['bid_movement'], 0)
        tmp_df['bid_downward_movement'] = np.where(tmp_df['bid_movement'] < 0, np.abs(tmp_df['bid_movement']), 0)

        tmp_df['ask_upward_movement']   = np.where(tmp_df['ask_movement'] > 0, tmp_df['ask_movement'], 0)
        tmp_df['ask_downward_movement'] = np.where(tmp_df['ask_movement'] < 0, np.abs(tmp_df['ask_movement']), 0)

        tmp_df.at[n_timestep -1, 'bid_avg_upward_movement']   = tmp_df[['bid_upward_movement']][:n_timestep].values.mean()
        tmp_df.at[n_timestep -1, 'bid_avg_downward_movement'] = tmp_df[['bid_downward_movement']][:n_timestep].values.mean()

        tmp_df.at[n_timestep -1, 'ask_avg_upward_movement']   = tmp_df[['ask_upward_movement']][:n_timestep].values.mean()
        tmp_df.at[n_timestep -1, 'ask_avg_downward_movement'] = tmp_df[['ask_downward_movement']][:n_timestep].values.mean()

        tmp_df = tmp_df[n_timestep -1:].copy()
        tmp_df.reset_index(inplace=True, drop=True)
        
        for row in tmp_df[1:].itertuples():
            tmp_df.at[row.Index, 'bid_avg_upward_movement']   = (tmp_df.at[row.Index -1, 'bid_avg_upward_movement'] * (n_timestep -1) + row.bid_upward_movement) / n_timestep
            tmp_df.at[row.Index, 'bid_avg_downward_movement'] = (tmp_df.at[row.Index -1, 'bid_avg_downward_movement'] * (n_timestep -1) + row.bid_downward_movement) / n_timestep

            tmp_df.at[row.Index, 'ask_avg_upward_movement']   = (tmp_df.at[row.Index -1, 'ask_avg_upward_movement'] * (n_timestep -1) + row.ask_upward_movement) / n_timestep
            tmp_df.at[row.Index, 'ask_avg_downward_movement'] = (tmp_df.at[row.Index -1, 'ask_avg_downward_movement'] * (n_timestep -1) + row.ask_downward_movement) / n_timestep
    
        tmp_df['bid_relative_strength'] = tmp_df['bid_avg_upward_movement'] / tmp_df['bid_avg_downward_movement']
        tmp_df['ask_relative_strength'] = tmp_df['ask_avg_upward_movement'] / tmp_df['ask_avg_downward_movement']

        tmp_df['bid_rsi'] = 100 - (100 / (tmp_df['bid_relative_strength'] + 1))
        tmp_df['ask_rsi'] = 100 - (100 / (tmp_df['ask_relative_strength'] + 1))

        tmp_df.drop(columns=[
            'bid_movement', 'ask_movement',
            'bid_upward_movement', 'bid_downward_movement',
            'ask_upward_movement', 'ask_downward_movement',
            'bid_avg_upward_movement', 'bid_avg_downward_movement',
            'ask_avg_upward_movement', 'ask_avg_downward_movement',
            'bid_relative_strength', 'ask_relative_strength'
        ], inplace=True)

        self.indexes   = tmp_df.index.values
        self.datetimes = tmp_df['datetime'].values
        
        self.open_bids = np.round(tmp_df['open_bid'].values, 5)
        self.high_bids = np.round(tmp_df['high_bid'].values, 5)
        self.low_bids  = np.round(tmp_df['low_bid'].values, 5)
        self.bids      = np.round(tmp_df['bid'].values, 5)
        self.bids_rsi  = np.round(tmp_df['bid_rsi'].values, 0)
        
        self.open_asks = np.round(tmp_df['open_ask'].values, 5)
        self.high_asks = np.round(tmp_df['high_ask'].values, 5)
        self.low_asks  = np.round(tmp_df['low_ask'].values, 5)
        self.asks      = np.round(tmp_df['ask'].values, 5)
        self.asks_rsi  = np.round(tmp_df['ask_rsi'].values, 0)

        max_bid_profit = round(self.bids.max() - self.bids.min(), 5) * 100_000
        max_ask_profit = round(self.asks.max() - self.asks.min(), 5) * 100_000
        self.estimate_max_profit = round((max_bid_profit + max_ask_profit) / 2, 0)
        
    def constant_values(self):
        return {
            'TRADE_STATUS': {
                'OPEN': 'OPEN',
                'CLOSE': 'CLOSED',
                'CLOSE_TRADE': 'CLOSE_TRADE'
            },
            'TRADE_ACTION': {
                'DEFAULT': -1,
                'BUY': 0,
                'SELL': 1,
                'HOLD': 2
            }
        }
        
    def state_space(self):
        return np.array(['buy_float_profit', 'ask_rsi', 'sell_float_profit', 'bid_rsi'])
        
    def state_size(self):
        return len(self.state_space())
        
    def action_space(self):
        const_action_dict = self.constant_values()['TRADE_ACTION']
        return [const_action_dict['BUY'], const_action_dict['SELL'], const_action_dict['HOLD']]
        
    def action_size(self):
        return len(self.action_space())
        
    def available_actions(self):
        const_status_dict = self.constant_values()['TRADE_STATUS']
        actions = self.action_space()
        
        # Have open trades
        trade_dict = self.trading_params_dict['trade_dict']
        if const_status_dict['OPEN'] in trade_dict['status']:
            open_index  = trade_dict['status'].index(const_status_dict['OPEN'])
            open_action = trade_dict['action'][open_index]

            # Ensure agent is able to have only 1 open trade while trading
            actions.remove(open_action)
        return actions
    
    def __price_by_action(self, action, bid, ask, closed_trade):
        const_action_dict = self.constant_values()['TRADE_ACTION']
        
        # Close trade by Selling at Ask price, and Buying at Bid price
        if closed_trade:
            return bid if action == const_action_dict['BUY'] else ask
        
        # Open trade by Buying at Ask price, and Selling at Bid price
        else:
            return ask if action == const_action_dict['BUY'] else bid
    
    def __profit_by_action(self, entry_action, entry_price, curr_bid, curr_ask):
        const_action_dict = self.constant_values()['TRADE_ACTION']
        if entry_action == const_action_dict['BUY']:
            return curr_ask - entry_price
        
        elif entry_action == const_action_dict['SELL']:
            return entry_price - curr_bid
        return 0
    
    def update_timestep(self, index):
        try:
            self.timestep = {
                'index':    self.indexes[index],
                'datetime': self.datetimes[index],
                
                'open_bid': self.open_bids[index],
                'high_bid': self.high_bids[index],
                'low_bid':  self.low_bids[index],
                'bid':      self.bids[index],
                'bid_rsi':  self.bids_rsi[index],
                
                'open_ask': self.open_asks[index],
                'high_ask': self.high_asks[index],
                'low_ask':  self.low_asks[index],
                'ask':      self.asks[index],
                'ask_rsi':  self.asks_rsi[index]
            }
            return False
        
        except IndexError:
            self.timestep = {}
            return True
    
    def update_observe_timestep(self):
        self.observe_timestep = self.timestep.copy()

    def normalize_reward(self, reward):
        return round(reward / self.estimate_max_profit, 2)

    def normalize_state(self, state):
        buy_float_profit, ask_rsi, sell_float_profit, bid_rsi = state

        norm_buy_fp   = scaler.clipping(buy_float_profit, self.estimate_max_profit)
        norm_sell_fp  = scaler.clipping(sell_float_profit, self.estimate_max_profit)
        norm_bid_rsi  = np.round(bid_rsi / 100, 2)
        norm_ask_rsi  = np.round(ask_rsi / 100, 2)

        return np.array([norm_buy_fp, norm_ask_rsi, norm_sell_fp, norm_bid_rsi])
    
    # TODO
    def normalize_stack_states(self, stack_states):
        return stack_states
    
    def reset(self, random=True):
        self.default_action = self.constant_values()['TRADE_ACTION']['DEFAULT']

        # Timestep
        index = np.random.choice(self.indexes[:self.random_range]) if random else 0
        index = self.stack_size -1 if index < self.stack_size else index
        
        # Stacked states
        self.stack_states = np.empty((self.stack_size, self.state_size()))

        for stack_index, x in enumerate(range(index +1 -self.stack_size, index +1)):
            if stack_index == 0:
                self.update_timestep(x)
                self.update_observe_timestep()

            stack_state = np.array([0, self.asks_rsi[x], 0, self.bids_rsi[x]])
            self.stack_states[stack_index] = stack_state
            
        self.update_timestep(x)
        
        # State
        self.state = stack_state
        
        # Trading
        self.trading_params_dict = {
            'orig_bal': 100_000.,
            'acct_bal': 100_000.,
            'unit':     100_000.,
            
            'trade_dict': {
                'action':   [],
                'datetime': [],
                'price':    [],
                'status':   [],
                'profits':  [],
                'acct_bal': []
            }
        }
        return self.state
    
    def __trade_vars(self, trade_dict, action=None):
        # Get entry action & price
        # - if there's no entry action, treat current action as action to open a trade
        # - if there's entry action, treat current action as action to close a trade
        try:
            # NOTE: not to use pd.DataFrame() to convert trade_dict to dataframe, as it is slower
            open_index      = trade_dict['status'].index(self.constant_values()['TRADE_STATUS']['OPEN'])
            trade_actions   = trade_dict['action'][open_index:]
            trade_prices    = trade_dict['price'][open_index:]
            trade_datetimes = trade_dict['datetime'][open_index:]
            
            entry_action = trade_actions[0]
            
            # Not allowed to close open trades with same entry action
            if entry_action == action:
                trade_actions   = []
                trade_prices    = []
                trade_datetimes = []
            
        except ValueError:
            trade_actions   = []
            trade_prices    = []
            trade_datetimes = []

            entry_action = self.default_action

        return entry_action, trade_prices, trade_datetimes

    def step(self, action):
        const_action_dict = self.constant_values()['TRADE_ACTION']
        const_status_dict = self.constant_values()['TRADE_STATUS']
        
        trade_dict = self.trading_params_dict['trade_dict']
        entry_action, trade_prices, trade_datetimes = self.__trade_vars(trade_dict, action)
        
        
        profit            = 0.
        float_profit      = 0.
        closed_trade      = False
        sufficient_margin = True

        if action in [const_action_dict['BUY'], const_action_dict['SELL']]:
            # Close open trades
            for trade_index, trade_price in enumerate(trade_prices):
                profit += self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                
                trade_dict['status'][trade_dict['datetime'].index(trade_datetimes[trade_index])] = const_status_dict['CLOSE']
                closed_trade = True
                
            profit *= self.trading_params_dict['unit']
            profit = round(profit, 5)

            # Add trade transaction
            self.trading_params_dict['acct_bal'] += profit
            price = self.__price_by_action(action, self.timestep['bid'], self.timestep['ask'], closed_trade)

            # Add back free margin upon close trade
            if closed_trade:
                self.trading_params_dict['acct_bal'] += (len(trade_prices) * self.trading_params_dict['unit'])
                
            # Deduct required margin upon opening trade
            else:
                required_margin = self.trading_params_dict['unit']
                if self.trading_params_dict['acct_bal'] < required_margin:
                    sufficient_margin = False
                self.trading_params_dict['acct_bal'] -= required_margin
            
            # Update trade transaction
            trade_dict['action'].append(action)
            trade_dict['datetime'].append(self.timestep['datetime'])
            trade_dict['price'].append(price)
            trade_dict['status'].append(const_status_dict['CLOSE_TRADE'] if closed_trade else const_status_dict['OPEN'])
            trade_dict['profits'].append(profit)
            trade_dict['acct_bal'].append(self.trading_params_dict['acct_bal'])

            # Observe the price at current timestemp if open or closed trades, else observe the entry price
            self.update_observe_timestep()
        
        # Update trade variables
        entry_action, trade_prices, trade_datetimes = self.__trade_vars(trade_dict)


        # Done
        done = self.update_timestep(self.timestep['index'] +1)
        if not done:
            # Stop trading if do not have sufficient balance, and there's no open trade
            if (self.trading_params_dict['acct_bal'] < self.trading_params_dict['unit']) & (const_status_dict['OPEN'] not in trade_dict['status']):
                done = True
                
            # Stop trading if do not have enough balance to pay for required margin
            elif not sufficient_margin:
                done = True
                
            # Consider closing trade as end of episode
            elif closed_trade:
               done = True

        # State
        if done:
            next_state = np.array([0, 0, 0, 0])
        else:
            # Calculate floating P/L
            float_profit = 0.
            if (entry_action != self.default_action) & (not closed_trade):
                for trade_index, trade_price in enumerate(trade_prices):
                    float_profit += self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                
                float_profit *= self.trading_params_dict['unit']
                float_profit = round(float_profit, 5)

            buy_float_profit  = float_profit if entry_action == const_action_dict['BUY'] else 0
            sell_float_profit = float_profit if entry_action == const_action_dict['SELL'] else 0
            next_state = np.array([buy_float_profit, self.timestep['ask_rsi'], sell_float_profit, self.timestep['bid_rsi']])
            
        self.state = next_state
        
        # Stacked states
        self.stack_states = np.vstack([self.stack_states[1:], self.state])
        
        # Reward
        reward = profit
        
        # Additional information
        info_dict = {
            'closed_trade': closed_trade,
            'sufficient_margin': sufficient_margin,
            'float_profit': float_profit,
            'have_open': len(trade_prices) > 0
        }
        return (self.state, reward, done, info_dict)