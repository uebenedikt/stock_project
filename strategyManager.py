import re
import multiprocessing as mp
from multiprocessing import Process, Manager, Lock
import pandas as pd
import json
import datetime
from datetime import date
import math
import pathlib
from scipy import stats
import os

start_date = date(2020, 1, 1)
base_balance = 30000.0  # also in logic_module.py
max_no_of_investments = 30
path = pathlib.Path('.').joinpath('strategyUtil')


# noinspection SpellCheckingInspection
def get_strategy(predictions, sell_prices, today):
    """
    Requires the most recent 14:00 price.
    Generates the strategies and chooses the best performing strategy.
    :param today: the day of the execution
    :param sell_prices: the price for which the stocks are sold, 14:00, dict: ticker->float
    :param predictions: dictionary of form: stock ticker -> discrete distribution of the next price
    :return: strategy: dictionary of form: stock ticker -> percentage that should be invested
    """

    try:
        os.mkdir('strategyUtil')
    except FileExistsError:
        pass

    """
    Removed
    """

    with open(str(path.joinpath('strategies.json')), 'w') as strat:
        json.dump(strategy_dict, strat)

    return chosen_strategy


def strategy_null():
    """Never Invests."""
    return {'remainder': 1.0}


def create_strategy(transformed_predictions, count=0, percentage=0.0, min_growth=0.0, weight_exponent=1.0):
    """
    Returns a buy-strategy as a dict of form: stock-ticker -> percentage to invest.
    :param transformed_predictions: needs to be a descending sorted list.
    :param count: How many different stocks to buy.
    :param percentage: How much of the budget is allowed to go into a single stock.
    :param min_growth: Cap for the predicted growth. Everything below is discarded.
    :param weight_exponent: Weight exponent.
    :return: dict of form: stock-ticker -> percentage
    """
    # first, check if there are enough stocks that pass the min_growth requirement.
    strategy = {}

    """
    Removed
    """

    strategy['remainder'] = remainder

    return strategy


def create_dataframe(today, buy_prices, sell_prices):
    """
    Creates or edits the DataFrame that contains a time series of balances for each strategy.
    :raises: FileNotFoundError
    :return: The DataFrame.
    """

    needs_initializing = False

    # read all the pre-calculated strategy splits. If there are none, raise a FileNotFoundError
    with open(str(path.joinpath('strategies.json')), 'r') as strategy:
        strategy_dict = json.load(strategy)

    # attempt to read the dataframe. contains the progression for each of the strategies over time
    df = {}
    try:
        df = pd.read_csv(str(path.joinpath('time_series.csv')), index_col=0)
    except FileNotFoundError:
        print('Initializing the first day.')
        needs_initializing = True

    progression = {}  # maps the strategies to the balance they generate

    list_of_param_dicts = []

    # compute the next value for interval based strategies
    for key, strategy in strategy_dict.items():
        if needs_initializing:
            balance = base_balance
        else:
            # get the last balance for the current strategy
            helper = list(df.columns)
            balance = df.at[key, helper[-1]]

        param_dict = {'name': key, 'balance': balance, 'strategy': strategy, 'buy_prices': buy_prices,
                      'sell_prices': sell_prices}

        list_of_param_dicts += [param_dict]

    if needs_initializing:
        # if the dataframe doesn't already exist, create a series containing the base balance for each strategy as an
        # initialization, and put that series into a dataframe
        yesterday = today - datetime.timedelta(days=1)
        start = pd.Series(base_balance, range(0, len(list_of_param_dicts)))
        start.index = [d['name'] for d in list_of_param_dicts]

        df = pd.DataFrame({f'{yesterday.year}-{yesterday.month}-{yesterday.day}': start})

    # evaluate each strategy, to get the balance after selling.

    with Manager() as manager:
        helper_list = manager.list()
        processes = []
        lock = Lock()
        number_of_cpus = mp.cpu_count()

        for i in range(0, number_of_cpus):
            # create a partition of the already reduced indices
            dict_slice = [d for j, d in enumerate(list_of_param_dicts) if j % number_of_cpus == i]
            p = Process(target=helper_create_dataframe, args=(dict_slice, helper_list, lock))
            p.start()
            processes += [p]

        for p in processes:
            p.join()

        helper_list = list(helper_list)

    for element in helper_list:
        progression[element[0]] = element[1]

    # add the newly calculated balances to the dataframe
    for key, value in progression.items():
        df.at[key, f'{today.year}-{today.month}-{today.day}'] = round(value, 2)

    # when dataFrame becomes to big, drop the first (oldest) column
    while len(df.columns) > 66:
        df = df.drop(df.columns[0], axis=1)

    return df


def helper_create_dataframe(list_of_dicts, results_list, lock):
    """

    :param list_of_dicts: list of parameter dictionaries, each representing a strategy
    :param results_list:
    :param lock:
    :return:
    """

    local_results = []

    for dictionary in list_of_dicts:
        value = calculate_balance_per_strategy(balance=dictionary['balance'], strategy=dictionary['strategy'],
                                               buy_prices=dictionary['buy_prices'],
                                               sell_prices=dictionary['sell_prices'])

        local_results += [(dictionary['name'], value)]

    lock.acquire()
    results_list += local_results
    lock.release()


def calculate_balance_per_strategy(balance, strategy, buy_prices, sell_prices):
    """
    Calculates the balance resulting from the execution of a given strategy.
    :param balance: the last balance for the strategy (yesterday)
    :param strategy: dict: ticker -> percentage
    :param buy_prices: dict: ticker -> price per share (yesterday, 14:00)
    :param sell_prices: dict: ticker -> price (today, 14:00)
    :return:
    """
    ####################################################################################################################
    # calculate what was bought according to the strategy on the previous day

    # to map the tickers to an amount of shares
    shares_per_stock = {}

    leftover = balance  # will contain unused funds after the execution of a strategy

    # calculate for each stock that is part of the strategy how many stocks were bought
    for key, value in strategy.items():
        if key == 'remainder':
            continue
        # number of shares (uncapped)
        shares_per_stock[key] = balance * value / buy_prices[key]
        leftover -= balance * value
    shares_per_stock['remainder'] = leftover

    ####################################################################################################################
    # calculate the balance after selling all stocks today

    # map the stocks to the dollar amount after selling the corresponding stock
    returns = {}
    for amount_key, amount_value in shares_per_stock.items():
        if amount_key == 'remainder':
            returns[amount_key] = shares_per_stock[amount_key]
            continue
        # it is possible that there is missing data for today. in that case, use the last known price.
        try:
            price = sell_prices[amount_key]
        except KeyError:
            # when on the following day there is no data for this stock, because it is no longer a selected stock
            with open('last_known_prices.json', 'r') as file:
                price_dict = json.load(file)
                price = price_dict[amount_key]

        # calculate the returns
        returns[amount_key] = price * shares_per_stock[amount_key]

    # calculate the new balance from the individual returns
    new_balance = 0

    for value in returns.values():
        new_balance += value

    return new_balance


def choose_strategy_save_df(today, df, is_null_dict_updated):
    """
    Find the strategy that realized the steepest growth for a given time span.
    :param today: to allow for a clean start after a given date, i.e. start of a year
    :param df: The dataframe containing the time series of balances for each strategy
    :param is_null_dict_updated: Identifies strategies, that result in no investments. Calculated previously.
    :return: list containing name of the strategy, its parameter as dict, and the sell strategy
    """
    # too little data to choose a good strategy
    if len(df.columns) <= 60 or today < start_date:
        df.to_csv(str(path.joinpath('time_series.csv')))
        return ['strategy_null', {}]
    else:
        ################################################################################################################
        # remove strategies that would result in no investments and constant strategies
        rows = len(df.T.columns)  # number of strategies
        last_column = len(df.columns)
        counter_no_investment = 0
        counter_null = 0
        index_list = []
        for index in range(0, rows):
            # remove null-strategies
            if df.iat[index, 0] == df.iat[index, last_column - 1]:
                # fist balance the same as the last -> null-strategy
                counter_null += 1
                continue
            if df.iat[index, last_column - 60] == df.iat[index, last_column - 1]:
                # no investments in the last 60 days -> null strategy
                counter_null += 1
                continue
            if is_null_dict_updated[df.index[index]]:
                # would result in no investments today
                counter_no_investment += 1
                continue

            # to allow second iteration (for example, to remove riskiest strategies)
            index_list += [index]

        print(f'Removed {counter_null} strategies that were null-strategies')
        print(f'Removed {counter_no_investment} strategies that would have resulted in no investments.')

        # if no strategy is suitable, return a null-strategy
        if len(index_list) == 0:
            df.to_csv(str(path.joinpath('time_series.csv')))
            return ['strategy_null', {}]

        print(f'Remaining strategies: {len(index_list)}.')

        # calculate the regression values in parallel
        with Manager() as manager:
            helper_list = manager.list()
            processes = []
            lock = Lock()
            number_of_cpus = mp.cpu_count()

            for i in range(0, number_of_cpus):
                # create a partition of the already reduced indices
                index_slice = [j for j in index_list if j % number_of_cpus == i]
                p = Process(target=helper_choose_strategy_save_df, args=(df, index_slice, helper_list, lock))
                p.start()
                processes += [p]

            for p in processes:
                p.join()

            results = list(helper_list)

        print(f'Regressions calculated: {len(results)}.')

        results_sorted = sorted(results, reverse=True)

        # if there is no promising strategy, default back to null_strategy
        if results_sorted[0][0] < 1.0:
            df.to_csv(str(path.joinpath('time_series.csv')))
            return ['strategy_null', {}, 1400]

        # a strategy was found. translate the name of the strategy back into its parameters
        chosen_strategy = results_sorted[0][1]

        function_name = re.search(r'\*(\w+)\(', chosen_strategy)
        function_name = function_name.groups()[0] if function_name else 'null'

        """
        Removed.
        """

        param_dict = {'p': p, 'g': g, 'm': m, 'w': w, 'c': c, 's': s}

        with open('r_val.json', 'w+') as json_file:
            json.dump(results_sorted[0][0], json_file)

        df.to_csv(str(path.joinpath('time_series.csv')))
        return [function_name, param_dict]


def helper_choose_strategy_save_df(data, indices, results_list, lock):
    """
    Calculates a set of regression slopes. Intended to be used in parallel.
    :param data: the dataframe containing the time series for each strategy
    :param indices: a list of indices, for which the regressions need to be computed
    :param results_list: a shared list in which the results are written
    :param lock: to avoid parallelization errors
    """
    start_points = []  # removed
    last_column = len(data.columns)
    local_results = []

    for row in indices:  # the row in the dataframe
        helper = []  # collects the slopes calculated
        for start_value in start_points:
            # we expect exponential growth. find a linear regression for the logarithmic (linear) data
            y_values_prev = data.iloc[row, last_column - start_value:last_column].tolist()
            x_values = list(range(0, len(y_values_prev)))
            y_values = [math.log(z) for z in y_values_prev]

            linear_regression = stats.linregress(x=x_values, y=y_values)
            log_growth = linear_regression.slope

            helper += [log_growth]

        # calculate the average regression slope
        avg_of_logs = sum(helper) / len(helper)

        growth = math.exp(avg_of_logs)
        row_name = data.index[row]
        local_results += [(growth, row_name)]

    # share the generated data with the main process
    lock.acquire()
    results_list += local_results
    lock.release()


# noinspection SpellCheckingInspection
def generate_strategy():
    """
    Removed
    """
