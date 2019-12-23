import lib._util.scaler as scaler

import numpy as np
import pandas as pd

'''
Reference:
- https://www.babypips.com/learn/forex/margin-call-exemplified
- https://www.babypips.com/learn/forex/pips-and-pipettes
- https://www.babypips.com/learn/forex/lots-leverage-and-profit-and-loss
- https://www.youtube.com/watch?v=Amt7foVw5YE
- https://www.youtube.com/watch?v=RgmDywzNlZA
'''
class ForexEnv:
    def __init__(self, source_path, filename, nrows=None, train_size=.7, train=True, random_size=.5, stack_size=5, timeframe=5):
        self.measure_unit = 10_000 if 'JPY' not in filename else 100
        self.leverage     = 10
        self.trade_unit   = 100_000

        self.timeframe = timeframe
        self.__train_test_split(source_path, filename, nrows=nrows, train_size=train_size, train=train)

        self.random_range = int(len(self.indexes) * random_size)
        self.stack_size   = stack_size
        
    def __train_test_split(self, source_path, filename, chunk_size=50_000, nrows=None, train_size=.7, train=True):
        source_file = f'{source_path}{filename}'
        df_chunks = pd.read_csv(source_file, sep=',',
                                parse_dates=['datetime'],
                                date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'),
                                chunksize=chunk_size, nrows=nrows)
        timeseries_df = pd.concat(df_chunks)
        
        row_count  = len(timeseries_df) if nrows is None else nrows
        split_size = round(row_count * train_size)
        
        if train:
            timeseries_df = timeseries_df[:split_size]
        else:
            timeseries_df = timeseries_df[split_size:]

        self.indexes   = timeseries_df.index.values
        self.datetimes = timeseries_df['datetime'].values
        
        self.bids      = np.round(timeseries_df['bid'].values, 5)
        self.bids_rsi  = np.round(timeseries_df['bid_rsi'].values, 0)
        
        self.asks      = np.round(timeseries_df['ask'].values, 5)
        self.asks_rsi  = np.round(timeseries_df['ask_rsi'].values, 0)

        # Calculate estimated max. profit
        # const_action_dict = self.constant_values()['TRADE_ACTION']
        # min_ask           = self.asks.min()
        # max_bid           = self.bids.max()
        # max_buy_reward    = self.__profit_by_action(const_action_dict['BUY'], min_ask, max_bid, None)
        # max_sell_reward   = self.__profit_by_action(const_action_dict['SELL'], max_bid, None, min_ask)
        # self.estimate_max_reward = max(max_buy_reward, max_sell_reward)

        # Calculate estimated max. pip change
        self.estimate_max_reward  = round((np.max(self.bids) - np.min(self.asks)) * self.measure_unit, 1)
        
    def constant_values(self):
        return {
            'TRADE_STATUS': {
                'OPEN': 'OPEN',
                'CLOSE': 'CLOSED',
                'CLOSE_TRADE': 'CLOSE_TRADE',
                'MARGIN_CALL': 'MARGIN_CALL'
            },
            'TRADE_ACTION': {
                'DEFAULT': -1,
                'BUY': 0,
                'SELL': 1,
                'HOLD': 2
            }
        }
        
    def state_space(self):
        return np.array(['usable_margin_percentage', 'buy_float_reward', 'ask_rsi', 'sell_float_reward', 'bid_rsi'])
        
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
        if const_status_dict['OPEN'] in self.trade_dict['status']:
            open_index  = self.trade_dict['status'].index(const_status_dict['OPEN'])
            open_action = self.trade_dict['action'][open_index]

            # Ensure agent is able to have only 1 open trade while trading
            actions.remove(open_action)
        return actions
    
    def __price_by_action(self, action, bid, ask):
        const_action_dict = self.constant_values()['TRADE_ACTION']

        # Buy at ask price, Sell at bid price
        return bid if action == const_action_dict['SELL'] else ask if action == const_action_dict['BUY'] else 0
    
    def __profit_by_action(self, entry_action, entry_price, curr_bid, curr_ask):
        const_action_dict = self.constant_values()['TRADE_ACTION']

        if entry_action == const_action_dict['BUY']:
            close_price = curr_bid
            movement    = close_price - entry_price

        elif entry_action == const_action_dict['SELL']:
            close_price = curr_ask
            movement    = entry_price - close_price

        else:
            movement = 0

        pip_change = round(movement * self.measure_unit, 1)
        profit     = ((1 / self.measure_unit) / close_price) * self.trade_unit * pip_change

        return round(profit, 2), pip_change
    
    def update_timestep(self, index):
        try:
            self.timestep = {
                'index':    self.indexes[index],
                'datetime': self.datetimes[index],
                
                'bid':      self.bids[index],
                'bid_rsi':  self.bids_rsi[index],
                
                'ask':      self.asks[index],
                'ask_rsi':  self.asks_rsi[index]
            }
            return False
        
        except IndexError:
            self.timestep = {}
            return True
    
    def normalize_reward(self, reward):
        return round(reward / self.estimate_max_reward, 2)

    def normalize_state(self, state):
        usable_margin_percentage, buy_float_reward, ask_rsi, sell_float_reward, bid_rsi = state

        norm_ump     = scaler.clipping(usable_margin_percentage, 100)
        norm_buy_fp  = scaler.clipping(buy_float_reward, self.estimate_max_reward)
        norm_sell_fp = scaler.clipping(sell_float_reward, self.estimate_max_reward)
        norm_bid_rsi = round(bid_rsi / 100, 2)
        norm_ask_rsi = round(ask_rsi / 100, 2)

        return np.array([norm_ump, norm_buy_fp, norm_ask_rsi, norm_sell_fp, norm_bid_rsi])
    
    # TODO
    def normalize_stack_states(self, stack_states):
        return stack_states
    
    def reset(self, random=False):
        self.default_action = self.constant_values()['TRADE_ACTION']['DEFAULT']

        # Timestep
        index = np.random.choice(self.indexes[:self.random_range]) if random else 0
        index = self.stack_size -1 if index < self.stack_size else index
        
        # Stacked states
        self.stack_states = np.empty((self.stack_size, self.state_size()))

        for stack_index, x in enumerate(range(index +1 -self.stack_size, index +1)):
            stack_state = np.array([100, 0, self.asks_rsi[x], 0, self.bids_rsi[x]])
            self.stack_states[stack_index] = stack_state
            
        self.update_timestep(x)
        
        # State
        self.state = stack_state

        # Done
        self.done  = False
        
        # Trading
        self.acct_bal              = 10_500.
        self.equity                = self.acct_bal
        self.usable_margin         = self.equity
        self.observe_usable_margin = self.usable_margin
        self.used_margin           = 0.

        self.trade_dict = {
            'action':     [],
            'datetime':   [],
            'price':      [],
            'status':     [],
            'pip_change': [],
            'profits':    [],
            'acct_bal':   []
        }
        return self.state
    
    def __trade_vars(self, action=None):
        # Get entry action & price
        # - if there's no entry action, treat current action as action to open a trade
        # - if there's entry action, treat current action as action to close a trade
        try:
            # NOTE: not to use pd.DataFrame() to convert trade_dict to dataframe, as it is slower
            open_index      = self.trade_dict['status'].index(self.constant_values()['TRADE_STATUS']['OPEN'])
            trade_actions   = self.trade_dict['action'][open_index:]
            trade_prices    = self.trade_dict['price'][open_index:]
            trade_datetimes = self.trade_dict['datetime'][open_index:]
            
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

    def __update_equity(self, float_profit):
        self.equity = self.acct_bal + float_profit
        self.__update_usable_margin()

    def __update_used_margin(self, used_margin=None, required_margin=0):
        self.used_margin   = self.used_margin if used_margin is None else used_margin
        self.used_margin   += required_margin
        self.__update_usable_margin()

    def __update_usable_margin(self):
        self.usable_margin = round(self.equity - self.used_margin, 2)

    def __add_transaction(self, action, pip_change, profit, closed_trade=False, margin_call=False):
        const_status_dict = self.constant_values()['TRADE_STATUS']
        price             = self.__price_by_action(action, self.timestep['bid'], self.timestep['ask'])

        self.trade_dict['action'].append(action)
        self.trade_dict['datetime'].append(self.timestep['datetime'])
        self.trade_dict['price'].append(price)
        self.trade_dict['status'].append(const_status_dict['MARGIN_CALL'] if margin_call else \
                                         const_status_dict['CLOSE_TRADE'] if closed_trade else \
                                         const_status_dict['OPEN'])
        self.trade_dict['pip_change'].append(pip_change)
        self.trade_dict['profits'].append(profit)
        self.trade_dict['acct_bal'].append(self.acct_bal)

    def close_trade(self, profit):
        self.__update_used_margin(used_margin=0)
        self.__update_equity(profit)
        self.acct_bal = self.equity

    def step(self, action):
        assert not self.done, 'Environment reaches terminal, please reset the environment.'

        const_action_dict = self.constant_values()['TRADE_ACTION']
        const_status_dict = self.constant_values()['TRADE_STATUS']
        
        entry_action, trade_prices, trade_datetimes = self.__trade_vars(action)
        
        
        profit            = 0.
        float_profit      = 0.
        pip_change        = 0.
        float_pip_change  = 0.
        closed_trade      = False
        sufficient_margin = True
        margin_call       = False

        # Open / Close trade
        if action in [const_action_dict['BUY'], const_action_dict['SELL']]:
            # Close open trades
            for trade_index, trade_price in enumerate(trade_prices):
                trade_profit, trade_pip = self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                profit     += trade_profit
                pip_change += trade_pip
                
                self.trade_dict['status'][self.trade_dict['datetime'].index(trade_datetimes[trade_index])] = const_status_dict['CLOSE']
                closed_trade = True

            # Add back used margin upon close trade
            if closed_trade:
                self.close_trade(profit)
                
            # Deduct required margin upon opening trade
            else:
                required_margin = self.trade_unit / self.leverage
                if self.usable_margin < required_margin:
                    sufficient_margin = False
                
                self.__update_used_margin(required_margin=required_margin)

            # Observe usable margin upon open / close trade
            self.observe_usable_margin = self.usable_margin
            
            if sufficient_margin:
                # Add trade transaction
                self.__add_transaction(action, pip_change, profit, closed_trade=closed_trade)
                
                # Update trade variables
                entry_action, trade_prices, trade_datetimes = self.__trade_vars()


        # Done
        self.done = self.update_timestep(self.timestep['index'] +1)
        if not self.done:
            # Stop trading if do not have enough usable margin to pay for required margin
            if not sufficient_margin:
                self.done = True

            # Consider closing trade as end of episode
            # elif closed_trade:
            #     self.done = True

        # State
        if self.done:
            next_state = np.array([0, 0, 0, 0, 0])
        else:
            # Calculate floating P/L
            float_profit = 0.
            if (entry_action != self.default_action) & (not closed_trade):
                for trade_index, trade_price in enumerate(trade_prices):
                    trade_profit, trade_pip = self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                    float_profit += trade_profit
                    float_pip_change += trade_pip
                
            self.__update_equity(float_profit)

            # Observe profit
            # buy_float_reward  = float_profit if entry_action == const_action_dict['BUY'] else 0
            # sell_float_reward = float_profit if entry_action == const_action_dict['SELL'] else 0

            # Observe pip change
            buy_float_reward  = float_pip_change if entry_action == const_action_dict['BUY'] else 0
            sell_float_reward = float_pip_change if entry_action == const_action_dict['SELL'] else 0

            # Usable margin %
            usable_margin_percentage = round(self.usable_margin / self.observe_usable_margin * 100, 0)

            # Next State
            next_state = np.array([usable_margin_percentage, buy_float_reward, self.timestep['ask_rsi'], sell_float_reward, self.timestep['bid_rsi']])

            # Margin call
            if self.equity <= self.used_margin:
                margin_call  = True
                closed_trade = True
                self.done    = True
                next_state   = np.array([0, 0, 0, 0, 0])

                self.__update_used_margin(used_margin=0)
                self.acct_bal = self.equity

                # Close open trades
                for trade_index, trade_price in enumerate(trade_prices):
                    self.trade_dict['status'][self.trade_dict['datetime'].index(trade_datetimes[trade_index])] = const_status_dict['CLOSE']

                close_action = const_action_dict['SELL'] if entry_action == const_action_dict['BUY'] else \
                               const_action_dict['BUY'] if entry_action == const_action_dict['SELL'] else \
                               const_action_dict['HOLD']

                profit           = float_profit
                float_profit     = 0.
                pip_change       = float_pip_change
                float_pip_change = 0.

                self.close_trade(profit)
                self.__add_transaction(close_action, pip_change, profit, margin_call=margin_call)

                # Update trade variables
                entry_action, trade_prices, trade_datetimes = self.__trade_vars()

                # Observe usable margin upon margin call
                self.observe_usable_margin = self.usable_margin
            
        self.state = next_state
        
        # Stacked states
        # NOTE: not to use np.vstack() as it's much slower
        # self.stack_states = np.vstack([self.stack_states[1:], self.state])
        self.stack_states = np.append(self.stack_states[1:], [self.state], axis=0)
        
        # Reward
        # reward = profit
        reward = pip_change
        
        # Additional information
        info_dict = {
            'closed_trade': closed_trade,
            'sufficient_margin': sufficient_margin,
            'margin_call': margin_call,

            'float_profit': float_profit,
            'float_pip_change': float_pip_change,

            'have_open': len(trade_prices) > 0
        }
        return (self.state, reward, self.done, info_dict)