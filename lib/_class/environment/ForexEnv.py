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
        return np.array(['default_entry', 'buy_entry', 'sell_entry', 'avg_bid_velocity', 'avg_ask_velocity'])
        
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
        
        except IndexError:
            self.timestep = {}
            return True
    
    def update_observe_timestep(self):
        self.observe_timestep = self.timestep.copy()

    # TODO
    def scale_state(self, state):
        return state
    
    # TODO
    def scale_stack_states(self, stack_states):
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

            # Reference: https://www.wikihow.com/Calculate-Velocity
            # Modified bid velocity to ensure it's having positive correlation with sell profit
            openbid_velocity = (self.observe_timestep['open_bid'] - self.open_bids[x]) / 2
            highbid_velocity = (self.observe_timestep['high_bid'] - self.high_bids[x]) / 2
            lowbid_velocity  = (self.observe_timestep['low_bid'] - self.low_bids[x]) / 2
            bid_velocity     = (self.observe_timestep['bid'] - self.bids[x]) / 2

            openask_velocity = (self.open_asks[x] - self.observe_timestep['open_ask']) / 2
            highask_velocity = (self.high_asks[x] - self.observe_timestep['high_ask']) / 2
            lowask_velocity  = (self.low_asks[x] - self.observe_timestep['low_ask']) / 2
            ask_velocity     = (self.asks[x] - self.observe_timestep['ask']) / 2

            avg_bid_velocity = (openbid_velocity + highbid_velocity + lowbid_velocity + bid_velocity) / 4
            avg_ask_velocity = (openask_velocity + highask_velocity + lowask_velocity + ask_velocity) / 4

            stack_state = np.array([1, 0, 0, avg_bid_velocity, avg_ask_velocity])
            stack_state = np.round(stack_state, 5)
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
            next_state = np.array([1, 0, 0, 0, 0])
        else:
            # Calculate floating P/L
            float_profit = 0.
            if (entry_action != self.default_action) & (not closed_trade):
                for trade_index, trade_price in enumerate(trade_prices):
                    float_profit += self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                
                float_profit *= self.trading_params_dict['unit']
                float_profit = round(float_profit, 5)


            # Velocity
            openbid_velocity = (self.observe_timestep['open_bid'] - self.timestep['open_bid']) / 2
            highbid_velocity = (self.observe_timestep['high_bid'] - self.timestep['high_bid']) / 2
            lowbid_velocity  = (self.observe_timestep['low_bid'] - self.timestep['low_bid']) / 2
            bid_velocity     = (self.observe_timestep['bid'] - self.timestep['bid']) / 2

            openask_velocity = (self.timestep['open_ask'] - self.observe_timestep['open_ask']) / 2
            highask_velocity = (self.timestep['high_ask'] - self.observe_timestep['high_ask']) / 2
            lowask_velocity  = (self.timestep['low_ask'] - self.observe_timestep['low_ask']) / 2
            ask_velocity     = (self.timestep['ask'] - self.observe_timestep['ask']) / 2

            avg_bid_velocity = (openbid_velocity + highbid_velocity + lowbid_velocity + bid_velocity) / 4
            avg_ask_velocity = (openask_velocity + highask_velocity + lowask_velocity + ask_velocity) / 4

            state_actions  = [-1, 0, 1]
            if closed_trade:
                default_entry, buy_entry, sell_entry = 1, 0, 0

            elif action == const_action_dict['HOLD']:
                default_entry, buy_entry, sell_entry = self.state[:len(state_actions)]

            else:
                onehot_actions = np.zeros(len(state_actions), dtype=np.int8)
                onehot_actions[state_actions.index(action)] = 1

                default_entry, buy_entry, sell_entry = onehot_actions

            next_state = np.array([default_entry, buy_entry, sell_entry, avg_bid_velocity, avg_ask_velocity])
            next_state = np.round(next_state, 5)
            
        self.state = next_state
        
        # Stacked states
        self.stack_states = np.vstack([self.stack_states[1:], self.state])
        
        # Reward
        reward = profit
        
        # Additional information
        info_dict = {
            'closed_trade': closed_trade,
            'sufficient_margin': sufficient_margin,
            'float_profit': float_profit
        }
        return (self.state, reward, done, info_dict)