import lib._util.scaler as scaler

import numpy as np
import pandas as pd

class ForexEnv:
    def __init__(self, source_path, filename, nrows=None, train_size=.7, train=True, random_size=.8, stack_size=5):
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
        bid_df = timeseries_df.set_index('datetime')['bid'].resample('5Min').ohlc().reset_index()
        ask_df = timeseries_df.set_index('datetime')['ask'].resample('5Min').ohlc().reset_index()
        
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
        return np.array(['entry_action', 'close_action', 'pip_change',
                         'open_bid_diff', 'high_bid_diff', 'low_bid_diff', 'bid_diff',
                         'open_ask_diff', 'high_ask_diff', 'low_ask_diff', 'ask_diff'])
        
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
                
                'open_ask': self.open_asks[index],
                'high_ask': self.high_asks[index],
                'low_ask':  self.low_asks[index],
                'ask':      self.asks[index]
            }
            return False
        
        except:
            self.timestep = {}
            return True
    
    def scale_state(self, state):
        entry_action, close_action, pip_change, \
        open_bid_diff, high_bid_diff, low_bid_diff, bid_diff, \
        open_ask_diff, high_ask_diff, low_ask_diff, ask_diff = state
        
        scaled_entry_action = scaler.min_max_scale(entry_action, -1, 1) # -1 = Default, 0 = Buy, 1 = Sell
        scaled_close_action = scaler.min_max_scale(close_action, -1, 1) # -1 = Default, 0 = Buy, 1 = Sell
        
        return np.array([scaled_entry_action, scaled_close_action, pip_change,
                         open_bid_diff, high_bid_diff, low_bid_diff, bid_diff,
                         open_ask_diff, high_ask_diff, low_ask_diff, ask_diff])
    
    def scale_stack_states(self, stack_states):
        entry_actions  = np.array([x[0] for x in stack_states])
        close_actions  = np.array([x[1] for x in stack_states])
        pip_changes    = np.array([x[2] for x in stack_states])
        open_bid_diffs = np.array([x[3] for x in stack_states])
        high_bid_diffs = np.array([x[4] for x in stack_states])
        low_bid_diffs  = np.array([x[5] for x in stack_states])
        bid_diffs      = np.array([x[6] for x in stack_states])
        open_ask_diffs = np.array([x[7] for x in stack_states])
        high_ask_diffs = np.array([x[8] for x in stack_states])
        low_ask_diffs  = np.array([x[9] for x in stack_states])
        ask_diffs      = np.array([x[10] for x in stack_states])
        
        scaled_entry_actions = scaler.min_max_scale(entry_actions, -1, 1) # -1 = Default, 0 = Buy, 1 = Sell
        scaled_close_actions = scaler.min_max_scale(close_actions, -1, 1) # -1 = Default, 0 = Buy, 1 = Sell
        
        return np.vstack([scaled_entry_actions, scaled_close_actions, pip_changes,
                          open_bid_diffs, high_bid_diffs, low_bid_diffs, bid_diffs,
                          open_ask_diffs, high_ask_diffs, low_ask_diffs, ask_diffs]).transpose()
    
    def reset(self, random=True):
        # Timestep
        index = np.random.choice(self.indexes[:self.random_range]) if random else 0
        index = self.stack_size -1 if index < self.stack_size else index
        
        # Stacked states
        self.default_action = self.constant_values()['TRADE_ACTION']['DEFAULT']
        entry_action = self.default_action
        close_action = self.default_action
        pip_change   = 0
        
        self.stack_states = np.empty((self.stack_size, self.state_size()))
        for stack_index, x in enumerate(range(index +1 -self.stack_size, index +1)):
            if stack_index == 0:
                self.update_timestep(x)
                self.observe_open_bid = self.timestep['open_bid']
                self.observe_high_bid = self.timestep['high_bid']
                self.observe_low_bid  = self.timestep['low_bid']
                self.observe_bid      = self.timestep['bid']
                self.observe_open_ask = self.timestep['open_ask']
                self.observe_high_ask = self.timestep['high_ask']
                self.observe_low_ask  = self.timestep['low_ask']
                self.observe_ask      = self.timestep['ask']
            
            open_bid_diff = self.open_bids[x] - self.observe_open_bid
            high_bid_diff = self.high_bids[x] - self.observe_high_bid
            low_bid_diff  = self.low_bids[x] - self.observe_low_bid
            bid_diff      = self.bids[x] - self.observe_bid
            open_ask_diff = self.open_asks[x] - self.observe_open_ask
            high_ask_diff = self.high_asks[x] - self.observe_high_ask
            low_ask_diff  = self.low_asks[x] - self.observe_low_ask
            ask_diff      = self.asks[x] - self.observe_ask
            
            stack_state = np.array([entry_action, close_action, pip_change,
                                    open_bid_diff, high_bid_diff, low_bid_diff, bid_diff,
                                    open_ask_diff, high_ask_diff, low_ask_diff, ask_diff])
            
            # stack_state = np.round(stack_state, 5)
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
    
    def step(self, action):
        const_action_dict = self.constant_values()['TRADE_ACTION']
        const_status_dict = self.constant_values()['TRADE_STATUS']
        
        trade_dict = self.trading_params_dict['trade_dict']
        
        # Get entry action & price
        # - if there's no entry action, treat current action as action to open a trade
        # - if there's entry action, treat current action as action to close a trade
        try:
            # NOTE: not to use pd.DataFrame() to convert trade_dict to dataframe, as it is slower
            open_index      = trade_dict['status'].index(const_status_dict['OPEN'])
            trade_actions   = trade_dict['action'][open_index:]
            trade_prices    = trade_dict['price'][open_index:]
            trade_datetimes = trade_dict['datetime'][open_index:]
            
            entry_action = trade_actions[0]
            
            # Not allowed to close open trades with same entry action
            if entry_action == action:
                trade_actions  = []
                trade_prices   = []
                trade_datetime = []
            
        except:
            trade_actions  = []
            trade_prices   = []
            trade_datetime = []

            entry_action = self.default_action
        
        
        profit = 0.
        closed_trade = False
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
            
            
            trade_dict['action'].append(action)
            trade_dict['datetime'].append(self.timestep['datetime'])
            trade_dict['price'].append(price)
            trade_dict['status'].append(const_status_dict['CLOSE_TRADE'] if closed_trade else const_status_dict['OPEN'])
            trade_dict['profits'].append(profit)
            trade_dict['acct_bal'].append(self.trading_params_dict['acct_bal'])
        
        
        # Calculate floating P/L
        float_profit = 0.
        if (entry_action != self.default_action) & (not closed_trade):
            for trade_index, trade_price in enumerate(trade_prices):
                float_profit += self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                
            float_profit *= self.trading_params_dict['unit']
            float_profit = round(float_profit, 5)
          
        # Pip change
        pip_change = profit if closed_trade else float_profit
        
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
                
        # Observe the price at current timestemp if open or closed trades, else observe the entry price
        if not done and (closed_trade or action != const_action_dict['HOLD']):
            self.observe_open_bid = self.timestep['open_bid']
            self.observe_high_bid = self.timestep['high_bid']
            self.observe_low_bid  = self.timestep['low_bid']
            self.observe_bid      = self.timestep['bid']
            self.observe_open_ask = self.timestep['open_ask']
            self.observe_high_ask = self.timestep['high_ask']
            self.observe_low_ask  = self.timestep['low_ask']
            self.observe_ask      = self.timestep['ask']
        
        # State
        if done:
            next_state = np.array([self.default_action, self.default_action, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            state_entry_action = self.default_action if closed_trade else self.state[0] if action == const_action_dict['HOLD'] else action
            state_close_action = action if closed_trade else self.default_action

            open_bid_diff = self.timestep['open_bid'] - self.observe_open_bid
            high_bid_diff = self.timestep['high_bid'] - self.observe_high_bid
            low_bid_diff  = self.timestep['low_bid'] - self.observe_low_bid
            bid_diff      = self.timestep['bid'] - self.observe_bid
            open_ask_diff = self.timestep['open_ask'] - self.observe_open_ask
            high_ask_diff = self.timestep['high_ask'] - self.observe_high_ask
            low_ask_diff  = self.timestep['low_ask'] - self.observe_low_ask
            ask_diff      = self.timestep['ask'] - self.observe_ask
            
            next_state = np.array([state_entry_action, state_close_action, pip_change,
                                   open_bid_diff, high_bid_diff, low_bid_diff, bid_diff,
                                   open_ask_diff, high_ask_diff, low_ask_diff, ask_diff])
            # next_state = np.round(next_state, 5)
            
        self.state = next_state
        
        # Stacked states
        self.stack_states = np.vstack([self.stack_states[1:], self.state])
        
        # Reward
        reward = profit
        
        # Additional information
        info_dict = {
            'closed_trade': closed_trade,
            'sufficient_margin': sufficient_margin,
            'pip_change': pip_change
        }
        return (self.state, reward, done, info_dict)