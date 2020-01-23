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
    def __init__(self, source_path, filename, nrows=None, train_size=.7, train=True, random_size=.8):
        self.measure_unit = 10_000 if 'JPY' not in filename else 100
        self.leverage     = 10
        self.trade_unit   = 100_000

        self.__train_test_split(source_path, filename, nrows=nrows, train_size=train_size, train=train)

        self.random_range = int(len(self.indexes) * random_size)
        
    def __train_test_split(self, source_path, filename, chunk_size=50_000, nrows=None, train_size=.7, train=True):
        source_file = f'{source_path}{filename}'
        df_chunks = pd.read_csv(source_file, sep=';',
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
        self.asks      = np.round(timeseries_df['ask'].values, 5)

        self.pca_1     = np.round(timeseries_df['pca_1'].values, 5)
        self.pca_2     = np.round(timeseries_df['pca_2'].values, 5)
        self.pca_3     = np.round(timeseries_df['pca_3'].values, 5)
        self.pca_4     = np.round(timeseries_df['pca_4'].values, 5)
        self.pca_5     = np.round(timeseries_df['pca_5'].values, 5)
        self.pca_6     = np.round(timeseries_df['pca_6'].values, 5)
        self.pca_7     = np.round(timeseries_df['pca_7'].values, 5)
        self.pca_8     = np.round(timeseries_df['pca_8'].values, 5)
        self.pca_9     = np.round(timeseries_df['pca_9'].values, 5)
        self.pca_10    = np.round(timeseries_df['pca_10'].values, 5)
        self.pca_11    = np.round(timeseries_df['pca_11'].values, 5)
        self.pca_12    = np.round(timeseries_df['pca_12'].values, 5)
        self.pca_13    = np.round(timeseries_df['pca_13'].values, 5)
        self.pca_14    = np.round(timeseries_df['pca_14'].values, 5)
        self.pca_15    = np.round(timeseries_df['pca_15'].values, 5)
        
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
        
    def terminal_state(self):
        return np.array([0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0])
        
    def state_space(self):
        return np.array(['buy_roi', 'sell_roi', 'enter_trade',
                         'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5',
                         'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10',
                         'pca_11', 'pca_12', 'pca_13', 'pca_14', 'pca_15'])
        
    def state_size(self):
        return len(self.state_space())
        
    def action_space(self):
        const_action_dict = self.constant_values()['TRADE_ACTION']
        return [const_action_dict['BUY'], const_action_dict['SELL'], const_action_dict['HOLD']]
        
    def action_size(self):
        return len(self.action_space())
        
    def available_actions(self):
        actions = self.action_space()
        
        # NOTE: not to use .index() as it's much slower
        # Ensure agent is able to have only 1 open trade while trading
        try:
            open_action = self.open_dict['action'][0]
            actions.remove(open_action)

        except IndexError:
            pass

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
        profit     = 0 if close_price == 0 else ((1 / self.measure_unit) / close_price) * self.trade_unit * pip_change

        return round(profit, 2), pip_change
    
    def update_timestep(self, index):
        try:
            self.timestep = {
                'index':    self.indexes[index],
                'datetime': self.datetimes[index],
                'bid':      self.bids[index],
                'ask':      self.asks[index],

                'pca_1':  self.pca_1[index],
                'pca_2':  self.pca_2[index],
                'pca_3':  self.pca_3[index],
                'pca_4':  self.pca_4[index],
                'pca_5':  self.pca_5[index],
                'pca_6':  self.pca_6[index],
                'pca_7':  self.pca_7[index],
                'pca_8':  self.pca_8[index],
                'pca_9':  self.pca_9[index],
                'pca_10': self.pca_10[index],
                'pca_11': self.pca_11[index],
                'pca_12': self.pca_12[index],
                'pca_13': self.pca_13[index],
                'pca_14': self.pca_14[index],
                'pca_15': self.pca_15[index]
            }
            return False
        
        except IndexError:
            self.timestep = {
                'index':    -1,
                'datetime': None,
                'bid':      0,
                'ask':      0,

                'pca_1':  0,
                'pca_2':  0,
                'pca_3':  0,
                'pca_4':  0,
                'pca_5':  0,
                'pca_6':  0,
                'pca_7':  0,
                'pca_8':  0,
                'pca_9':  0,
                'pca_10': 0,
                'pca_11': 0,
                'pca_12': 0,
                'pca_13': 0,
                'pca_14': 0,
                'pca_15': 0
            }
            return True
    
    # TODO
    def normalize_reward(self, reward):
        return reward

    # TODO
    def normalize_state(self, state):
        return state
    
    def __reset_tradestats(self):
        self.acct_bal              = 15_000
        self.equity                = self.acct_bal
        self.usable_margin         = self.equity
        self.observe_usable_margin = self.usable_margin
        self.used_margin           = 0.

    def reset(self, random=False):
        self.default_action = self.constant_values()['TRADE_ACTION']['DEFAULT']

        # Timestep
        index = np.random.choice(self.indexes[:self.random_range]) if random else 0
        self.update_timestep(index)
        
        # State
        self.state = np.array([0, 0, 0,
                               self.timestep['pca_1'], self.timestep['pca_2'], self.timestep['pca_3'], self.timestep['pca_4'], self.timestep['pca_5'],
                               self.timestep['pca_6'], self.timestep['pca_7'], self.timestep['pca_8'], self.timestep['pca_9'], self.timestep['pca_10'],
                               self.timestep['pca_11'], self.timestep['pca_12'], self.timestep['pca_13'], self.timestep['pca_14'], self.timestep['pca_15']])

        # Done
        self.done  = False
        
        # Trading
        self.__reset_tradestats()
        self.trade_dict = {
            'action':     [],
            'datetime':   [],
            'price':      [],
            'status':     [],
            'pip_change': [],
            'profits':    [],
            'acct_bal':   []
        }
        self.open_dict = {
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
            trade_actions   = self.open_dict['action'][0:]
            trade_prices    = self.open_dict['price'][0:]
            
            entry_action = trade_actions[0]
            
            # Not allowed to close open trades with same entry action
            if entry_action == action:
                trade_actions   = []
                trade_prices    = []
            
        except IndexError:
            trade_actions   = []
            trade_prices    = []

            entry_action = self.default_action

        return entry_action, trade_prices

    def __update_equity(self, float_profit):
        self.equity = self.acct_bal + float_profit
        self.__update_usable_margin()

    def __update_used_margin(self, used_margin=None, required_margin=0):
        self.used_margin   = self.used_margin if used_margin is None else used_margin
        self.used_margin   += required_margin
        self.__update_usable_margin()

    def __update_usable_margin(self):
        self.usable_margin = round(self.equity - self.used_margin, 2)

    def __add_transaction(self, action, pip_change, profit, closed_trade=False, margin_call=False, done=False):
        const_status_dict = self.constant_values()['TRADE_STATUS']
        price             = self.__price_by_action(action, self.timestep['bid'], self.timestep['ask'])

        # Stop trade
        if closed_trade or margin_call or done:
            self.trade_dict['action'].extend(self.open_dict['action'])
            self.trade_dict['datetime'].extend(self.open_dict['datetime'])
            self.trade_dict['price'].extend(self.open_dict['price'])
            self.trade_dict['status'].extend([const_status_dict['CLOSE'] if closed_trade or margin_call else x
                                              for x in self.open_dict['status']])
            self.trade_dict['pip_change'].extend(self.open_dict['pip_change'])
            self.trade_dict['profits'].extend(self.open_dict['profits'])
            self.trade_dict['acct_bal'].extend(self.open_dict['acct_bal'])
            self.open_dict = {
                'action':     [],
                'datetime':   [],
                'price':      [],
                'status':     [],
                'pip_change': [],
                'profits':    [],
                'acct_bal':   []
            }

            if closed_trade or margin_call:
                self.trade_dict['action'].append(action)
                self.trade_dict['datetime'].append(self.timestep['datetime'])
                self.trade_dict['price'].append(price)
                self.trade_dict['status'].append(const_status_dict['MARGIN_CALL'] if margin_call else const_status_dict['CLOSE_TRADE'])
                self.trade_dict['pip_change'].append(pip_change)
                self.trade_dict['profits'].append(profit)
                self.trade_dict['acct_bal'].append(self.acct_bal)

        # Open trade
        else:
            self.open_dict['action'].append(action)
            self.open_dict['datetime'].append(self.timestep['datetime'])
            self.open_dict['price'].append(price)
            self.open_dict['status'].append(const_status_dict['OPEN'])
            self.open_dict['pip_change'].append(pip_change)
            self.open_dict['profits'].append(profit)
            self.open_dict['acct_bal'].append(self.acct_bal)

    def close_trade(self, profit):
        self.__update_used_margin(used_margin=0)
        self.__update_equity(profit)
        self.acct_bal = self.equity

    def step(self, action):
        assert not self.done, 'Environment reaches terminal, please reset the environment.'

        const_action_dict          = self.constant_values()['TRADE_ACTION']
        entry_action, trade_prices = self.__trade_vars(action)
        
        
        roi               = 0.
        float_roi         = 0.
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
            for trade_price in trade_prices:
                trade_profit, trade_pip = self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                profit       += trade_profit
                pip_change   += trade_pip
                closed_trade = True

            # Calculate ROI
            if self.used_margin != 0:
                roi = round(profit / self.used_margin, 5)

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
            
            if sufficient_margin or closed_trade:
                # Add trade transaction
                self.__add_transaction(action, pip_change, profit, closed_trade=closed_trade)
                
                # Update trade variables
                entry_action, trade_prices = self.__trade_vars()


        # Next timetep
        self.done = self.update_timestep(self.timestep['index'] +1)

        # Calculate floating P/L
        float_profit = 0.
        if (entry_action != self.default_action) & (not closed_trade):
            for trade_price in trade_prices:
                trade_profit, trade_pip = self.__profit_by_action(entry_action, trade_price, self.timestep['bid'], self.timestep['ask'])
                float_profit     += trade_profit
                float_pip_change += trade_pip
            
        self.__update_equity(float_profit)

        # Calculate floating ROI
        if self.used_margin:
            float_roi = round(float_profit / self.used_margin, 5)

        # Next State
        if entry_action == const_action_dict['BUY']:
        	buy_roi  = float_roi
        	sell_roi = 0
        elif entry_action == const_action_dict['SELL']:
        	buy_roi  = 0
        	sell_roi = float_roi
        else:
        	buy_roi  = 0
        	sell_roi = 0
        next_state = np.array([buy_roi, sell_roi, entry_action != self.default_action,
                               self.timestep['pca_1'], self.timestep['pca_2'], self.timestep['pca_3'], self.timestep['pca_4'], self.timestep['pca_5'],
                               self.timestep['pca_6'], self.timestep['pca_7'], self.timestep['pca_8'], self.timestep['pca_9'], self.timestep['pca_10'],
                               self.timestep['pca_11'], self.timestep['pca_12'], self.timestep['pca_13'], self.timestep['pca_14'], self.timestep['pca_15']])

        # Margin call
        if self.equity <= self.used_margin:
            margin_call       = True
            closed_trade      = True
            sufficient_margin = False

            self.__update_used_margin(used_margin=0)
            self.acct_bal = self.equity

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
            entry_action, trade_prices = self.__trade_vars()

            # Observe usable margin upon margin call
            self.observe_usable_margin = self.usable_margin


        # Done
        if not self.done:
            # Stop trading if hit margin call
            if margin_call:
                self.done = True

            # Stop trading if do not have enough usable margin to pay for required margin
            elif not sufficient_margin:
                self.done = True

        # State
        if self.done:
            self.__add_transaction(entry_action, pip_change, profit, done=True)
            next_state = self.terminal_state()
        self.state = next_state
        
        # Reward
        # reward = profit
        reward = pip_change
        
        # Trade Done & State
        trade_done       = (self.done or closed_trade)
        trade_next_state = self.terminal_state() if trade_done else next_state
        if trade_done:
            self.__reset_tradestats()

        # Additional information
        info_dict = {
            # 'closed_trade': closed_trade,
            # 'sufficient_margin': sufficient_margin,
            # 'margin_call': margin_call,

            # 'float_profit': float_profit,
            # 'float_pip_change': float_pip_change,

            'roi': roi,
            'float_roi': float_roi,

            'entry_action': entry_action,
            'have_open': len(trade_prices) > 0,
            'trade_done': trade_done,
            'trade_next_state': trade_next_state
        }
        return (self.state, reward, self.done, info_dict)