from StockSnippet import StockSnippet
import ensembleEstimator
import strategyManager
import polygonKey
import alpacaKEY

import numpy as np
import time
import datetime
from datetime import date
from datetime import datetime as dt

import threading
import os
import pathlib
import json

import math
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

base_balance = strategyManager.base_balance


def run(today, last_initialization, balance_in_dollar, simulation):
    """
    Meant to be used daily at 14:00 (New York). Updates, edits and generates data, coordinates tasks to submodules.
    :param today:
    :param balance_in_dollar:
    :param simulation:
    :return:
    """
    start_time_complete = datetime.datetime.now()
    ####################################################################################################################
    # check if the program already ran for today
    start_time = datetime.datetime.now()
    print('Check if day was already processed...')
    done_dict = {}
    try:
        with open('last_day.json', 'r') as json_file:
            done_dict = json.load(json_file)
            year, month, day = done_dict['execute']
            this_day = date(year, month, day)

            if this_day >= today:
                print(f'The calculations for {str(today)} were already performed, program will go back to main.')
                return {}
    except (FileNotFoundError, KeyError):
        pass

    print(f'Check if day was already processed: done. Duration: {datetime.datetime.now() - start_time}')
    ####################################################################################################################
    # step 1: update the stocks
    start_time = datetime.datetime.now()
    print('Load, update and edit data...')
    list_of_stocks, list_of_missing_days = get_updated_stocks(today, simulation)
    print(f'Load, update and edit data: done. Duration: {datetime.datetime.now() - start_time}')

    ####################################################################################################################
    # step 2: get prices for stocks bought (for the future) and sold today (already sold at time of execution!)
    start_time = datetime.datetime.now()
    print('Get prices...')
    # since the execution of this function takes some time, it is not possible to buy the shares at 14:00. The real
    # price per share can be assumed to be somewhere between the 14:00 and 15:00 price
    buy_prices_today = {}  # the ideal case, used for the generation of the training data
    real_buy_prices = {}  # the more realistic case, used for the evaluation of the strategies

    to_remove = []  # if there is no buy price available, do not consider the stock for buying
    for stock in list_of_stocks:
        # this is the price that was just downloaded
        try:
            buy_prices_today[stock.abbreviation] = stock.get_price(year=today.year, month=today.month, day=today.day,
                                                                   hour=14)
        except IndexError:
            to_remove += [stock]
            continue

        try:
            next_hour = stock.get_price(year=today.year, month=today.month, day=today.day, hour=15)

            real_buy_prices[stock.abbreviation] = (buy_prices_today[stock.abbreviation] + next_hour) / 2
        except IndexError:
            real_buy_prices[stock.abbreviation] = buy_prices_today[stock.abbreviation]

    # remove the stocks with missing buy prices
    list_of_stocks = [stock for stock in list_of_stocks if stock not in to_remove]

    # get prices between 15:00 (yesterday) and 14:00 (today).
    # This data is used to evaluate strategies that sell price dependent, not time dependent
    sell_prices = {}
    for stock in list_of_stocks:
        # find the stock price for today
        tup = stock.edited_data[dt(year=today.year, month=today.month, day=today.day, hour=14)]
        # remove any potential 'missing data'-markers
        price = tup[0]

        sell_prices[stock.abbreviation] = price  # most recent last

    print(f'Get prices: done. Duration: {datetime.datetime.now() - start_time}')
    ####################################################################################################################
    start_time = datetime.datetime.now()
    # create the data for today, without a response value
    print('Create data for predictions...')

    stocks_to_remove = []
    helper_list = []

    for stock in list_of_stocks:
        temp = stock.generate_training_data(date_list=[today], number_of_categories=23, get_response=False, save=False)
        # one datapoint found?
        if len(temp) > 0:
            helper_list += [temp[0]]
        else:
            stocks_to_remove += [stock.abbreviation]

    if len(list_of_stocks) == len(stocks_to_remove):
        print(f'There is no data for {today.year}-{today.month}-{today.day}. Return to main.')
        return {}

    prediction_x = np.zeros((len(helper_list), len(helper_list[0][4])), dtype=float)
    prediction_link = {}

    for row in range(0, len(helper_list)):
        for column in range(0, len(helper_list[0][4])):
            prediction_x[row, column] = helper_list[row][4][column]
            prediction_link[helper_list[row][0]] = row

    print(f'Create data for predictions: done. Duration: {datetime.datetime.now() - start_time}')
    ####################################################################################################################
    start_time = datetime.datetime.now()
    print('Create training data...')

    # remove stocks for which there is no data today, as these will not be considered for investments
    list_of_stocks = [stock for stock in list_of_stocks if stock.abbreviation not in stocks_to_remove]

    # generate a training-set consisting of the data of the last trade-week
    current = today - datetime.timedelta(days=1)  # first day on which all data could be available (response!)
    training_data_list = []

    # first, find data and put it in a list
    day_list = []

    while len(day_list) < 5:
        if current.weekday() < 5 and current not in list_of_missing_days:
            day_list += [current]
        current = current - datetime.timedelta(days=1)

    for stock in list_of_stocks:
        # only spend time saving the data during simulations
        data = stock.generate_training_data(date_list=day_list, number_of_categories=23, get_response=True,
                                            save=simulation)
        if len(data) > 0:
            training_data_list += data

    training_x = np.zeros((len(training_data_list), len(training_data_list[0][4])), dtype=float)

    for row in range(0, len(training_data_list)):
        for column in range(0, len(training_data_list[0][4])):
            training_x[row, column] = training_data_list[row][4][column]

    training_y = np.zeros((len(training_data_list), len(training_data_list[0][5])), dtype=float)

    for row in range(0, len(training_data_list)):
        for column in range(0, len(training_data_list[0][5])):
            training_y[row, column] = training_data_list[row][5][column]

    print(f'Create training data: done. Duration: {datetime.datetime.now() - start_time}')
    ####################################################################################################################
    start_time = datetime.datetime.now()
    print('Load and re-train the ensemble estimator...')
    ens = ensembleEstimator.EnsembleEstimator(today=last_initialization)
    ens.fit(x_input=training_x, y_input=training_y)
    print(f'Load and re-train the ensemble estimator: done. Duration: {datetime.datetime.now() - start_time}')
    ####################################################################################################################
    start_time = datetime.datetime.now()
    print('Calculate predictions using the ensemble estimator...')
    predictions = ens.predict(prediction_x)
    predictions_dict = {}

    for key, value in prediction_link.items():
        predictions_dict[key] = predictions[value]

    print(f'Calculate predictions using the ensemble estimator: done. Duration: {datetime.datetime.now() - start_time}')
    ####################################################################################################################
    # choose an investment strategy
    start_time = datetime.datetime.now()
    print('Choose a strategy...')
    buy_strategy = strategyManager.get_strategy(predictions_dict, sell_prices, today)

    print(f'Choose a strategy: done. Duration: {datetime.datetime.now() - start_time}')
    ####################################################################################################################
    start_time = datetime.datetime.now()
    print('Create the investment recommendation...')

    # to map the ticker to the full stock name
    with open('stock_dictionary.json', 'r') as file:
        stock_dict = json.load(file)

    # create the text file instructions for the real-money investments
    investments_real = {}
    remaining_money = balance_in_dollar

    # for each stock in the strategy, calculate how many shares to buy
    # buy strategy is a dict: ticker -> value, where the values form a partition of 1
    for key, value in buy_strategy.items():
        if key == 'remainder':
            continue
        else:
            to_invest = balance_in_dollar * value
            temp = to_invest / buy_prices_today[key]
            temp = math.floor(100 * temp) / 100
            investments_real[key] = temp
            remaining_money -= temp * buy_prices_today[key]

    # how long is the longest stock name?
    len_max = 10
    for key in investments_real.keys():
        if len(stock_dict[key]) > len_max:
            len_max = len(stock_dict[key])

    investments_real['remainder'] = remaining_money

    with open('STRATEGY.txt', 'w') as file:
        file.write(f'{today.year}/{today.month}/{today.day}\n\n')
        file.write(f'{"Name":<{len_max}}   {"Abbr.":>5}   {"Shares":>6}   {"sell at":<8}\n')
        for key, value in investments_real.items():
            if value > 0:
                if key != 'remainder':
                    string = f'{stock_dict[key]:<{len_max}}   {key:>5}   {value:>6.2f}   {"14:00":>8}\n'

                    file.write(string)
                else:
                    file.write(f'{"remainder":<{len_max}}          {value:>7.2f}$\n')

    print(f'Create the investment recommendation: done. Duration: {datetime.datetime.now() - start_time}')
    print(f'Total elapsed time to reach the recommendation: {datetime.datetime.now() - start_time_complete}')
    ####################################################################################################################
    # this part is not necessary for the functionality of the routine, but provides useful insights
    if True:
        start_time = datetime.datetime.now()
        # calculate simulated balance
        print('Calculate simulated balance...')
        try:
            progress_simulated, current_balance_simulated = calculate_balance(today, sell_prices)
        except FileNotFoundError:
            print('There was a file missing, calculating the progress failed.')
            progress_simulated = []
            current_balance_simulated = 5000
        print(f'Calculate simulated balance: done.')
        ################################################################################################################
        print('Create the simulation...')
        try:
            investments_simulated = {}

            leftover = current_balance_simulated
            # calculate how many shares per stock to buy
            for key, value in buy_strategy.items():
                if key == 'remainder':
                    continue
                else:
                    to_invest = current_balance_simulated * value
                    temp = to_invest / buy_prices_today[key]
                    temp = math.floor(100*temp)/100
                    investments_simulated[key] = temp
                    leftover -= temp*buy_prices_today[key]

            investments_simulated['remainder'] = leftover

            # save the sell strategy for later, save the sell-thresholds or the time at which to sell
            with open('investments_simulated.json', 'w') as invest:
                json.dump(investments_simulated, invest)

            with open('progress.json', 'w+') as file:
                json.dump(progress_simulated, file)
        except FileNotFoundError:
            pass
        print('Create the simulation: done.')
        ################################################################################################################
        print('Plot simulation data...')
        if len(progress_simulated) > 0:
            data = [point for index, point in enumerate(progress_simulated) if int(point[3]) != int(base_balance)]

            # add one point with the base balance, right before the first data point
            first_day = date(year=3000, month=1, day=1)

            for point in data:
                dd = date(year=point[0], month=point[1], day=point[2])
                if dd < first_day:
                    first_day = dd

            day_before = first_day - datetime.timedelta(days=1)

            data = [[day_before.year, day_before.month, day_before.day, base_balance]] + data

            if len(data) > 2:
                image_name = 'complete'
                plot_data(today, image_name, data)

            data_current_year = [point for point in data if point[0] == today.year]

            if len(data_current_year) > 2:
                image_name = 'current'
                plot_data(today, image_name, data_current_year)

        print('Plot simulation data: done.')
        print(f'Time for the simulation: {datetime.datetime.now() - start_time}')
    ####################################################################################################################

    with open('buy_prices.json', 'w+') as json_file:
        json.dump(real_buy_prices, json_file)

    # save the last known price in case a stock gets removed for the next day (ie because of missing data)
    last_known_prices = {}
    for stock in list_of_stocks:
        max_key = max(stock.edited_data.keys())
        last_known_prices[stock.abbreviation] = stock.edited_data[max_key][0]

    with open('last_known_prices.json', 'w+') as file:
        json.dump(last_known_prices, file)

    with open('last_day.json', 'w+') as json_file:
        done_dict['execute'] = (today.year, today.month, today.day)
        json.dump(done_dict, json_file)

    print(f'Time total: {datetime.datetime.now() - start_time_complete}')

    return buy_strategy


def get_updated_stocks(today, simulation):
    """

    :param today:
    :param simulation:
    :return:
    """
    ####################################################################################################################
    # create a list of stocks from saved data
    path = pathlib.Path('.')
    with open(str(path.joinpath('selected_stocks')) + '.json', 'r') as file:
        list_of_names = json.load(file)

    list_of_stocks = [StockSnippet(name=n[0], abbreviation=n[1], category_=n[2]) for n in list_of_names]

    ####################################################################################################################
    # update the stocks

    if polygonKey.PREMIUM:
        threads = []
        for stock in list_of_stocks:
            # don't save the data, for speedup
            thread = threading.Thread(target=stock.update, args=(False, None, None, False))
            thread.start()
            threads += [thread]

        for i in range(len(threads)):
            threads[i].join()

    ####################################################################################################################
    # edit data and remove bad data
    for stock in list_of_stocks:
        stock.edit_data(today - datetime.timedelta(days=50), today)

    # remove empty data
    list_of_stocks = [stock for stock in list_of_stocks if not stock.isEmpty]

    # remove days without trades
    try:
        with open('missing_days.json', 'r') as file:
            loaded_list = json.load(file)
            # the first two elements represent the first and last day that were already checked.
            # the third element is a list of all known missing days
            first_day = datetime.date(year=loaded_list[0][0], month=loaded_list[0][1], day=loaded_list[0][2])
            last_day = datetime.date(year=loaded_list[1][0], month=loaded_list[1][1], day=loaded_list[1][2])
            missing_days_all = [datetime.date(year=i[0], month=i[1], day=i[2]) for i in loaded_list[2]]
            missing_days_in_scope = [i for i in missing_days_all if today - datetime.timedelta(days=60) <= i <= today]
    except FileNotFoundError:
        first_day = today - datetime.timedelta(days=60)
        last_day = first_day - datetime.timedelta(days=1)
        missing_days_all = []
        missing_days_in_scope = []

    # try to find new no-trade days
    helper_list = []
    helper_set = set()
    no_trade_days = []
    # don't search in the range that was already covered
    starting_date_for_search = last_day + datetime.timedelta(days=1)

    for stock in list_of_stocks:
        # temp is a list of missing days for this stock
        temp = stock.possible_no_trade_days(starting_date_for_search, today)
        helper_list += [temp]  # list of lists
        helper_set.update(temp)

    # check if a suspicious day is in many of the lists
    for day_obj in helper_set:
        if len([lists for lists in helper_list if day_obj in lists]) >= 0.8 * len(helper_list):
            no_trade_days += [day_obj]

    # remove data that might be available for a found no-trade day
    no_trade_days_in_scope = no_trade_days + missing_days_in_scope
    for stock in list_of_stocks:
        stock.remove_no_trade_days(today=today, no_trade_days=no_trade_days_in_scope, save=simulation)

    # save the findings to a file for later use
    no_trade_days_complete = missing_days_all + no_trade_days
    list_of_tuples = [(i.year, i.month, i.day) for i in no_trade_days_complete]

    to_dump = [[first_day.year, first_day.month, first_day.day], [today.year, today.month, today.day],
               list_of_tuples]

    with open('missing_days.json', 'w+') as file:
        json.dump(to_dump, file)

    return list_of_stocks, no_trade_days_complete


def calculate_balance(today, sell_prices, regression_value):
    """
    Returns the simulated balance for the current day after selling the stocks.
    :param today:
    :param sell_prices:
    :param regression_value: the regression value of the strategy that led to the decision
    :return:
    """
    ####################################################################################################################
    # load the necessary data

    # try to load the progression. If this fails, initialize the progress.
    try:
        with open('progress.json', 'r') as file:
            progress = json.load(file)
        with open('buy_prices.json', 'r') as json_file:
            buy_prices_yesterday = json.load(json_file)
    except FileNotFoundError:
        print('There is no data containing the overall progress.')
        progress = [(today.year, today.month, today.day, base_balance)]

        with open('progress.json', 'w+') as file:
            json.dump(progress, file)

        return progress, base_balance

    # day already in the list for some reason?
    for point in progress:
        if today.year == point[0] and today.month == point[1] and today.day == point[2]:
            return progress, point[3]

    # load the investments that were made yesterday in the simulation
    try:
        with open('investments_simulated.json', 'r') as file:
            investments = json.load(file)
    except FileNotFoundError:
        print('There is no data containing the investments.')
        raise FileNotFoundError

    # load the last known prices. Used as a backup in case the recent data is not available
    try:
        with open('last_known_prices.json', 'r') as file:
            last_price_dict = json.load(file)
    except FileNotFoundError:
        print('There is no data containing the last known prices.')
        raise FileNotFoundError

    ####################################################################################################################
    # calculate the new balance after executing the sell strategies for the last day
    balance = investments['remainder']
    chosen_sell_prices = {}

    # sell for the first matching price
    for key, amount in investments.items():
        if key == 'remainder':
            continue
        try:
            # in case there is no current sell price available, default back to last known price
            sell_price = sell_prices[key]
        except KeyError:
            sell_price = last_price_dict[key]
        # initialize as the 14:00 price
        chosen_sell_prices[key] = sell_price
        balance += amount * sell_price

    progress += [(today.year, today.month, today.day, balance)]

    ####################################################################################################################
    # put the regression value and the result in a list

    if regression_value != -1:
        try:
            with open('regression_list.json', 'r') as json_file:
                reg_list = json.load(json_file)
        except FileNotFoundError:
            reg_list = []

        reg_list += [(regression_value, balance / progress[-2][3])]

        with open('regression_list.json', 'w+') as json_file:
            json.dump(reg_list, json_file)

    ####################################################################################################################
    # print summary
    if len(progress) > 1:
        try:
            with open(f'{today.year}_summary.txt', 'r') as txt:
                temp = [row for row in txt]
        except FileNotFoundError:
            temp = []

        with open(f'{today.year}_summary.txt', 'w+') as txt:
            txt.write('{\n')

            txt.write(f'########## Summary for {today.year}/{today.month}/{today.day}: ##########\n')
            txt.write(f'Not Invested: {investments["remainder"]:.2f}$\n')
            for key, value in chosen_sell_prices.items():
                gains = investments[key]*(round(chosen_sell_prices[key] - buy_prices_yesterday[key], 4))

                txt.write(f'{key:<5}: Bought {investments[key]:>9.2f} shares at price ' +
                          f'{buy_prices_yesterday[key]:>9.4f}$, sold for the price of ' +
                          f'{chosen_sell_prices[key]:>9.4f}$. Gain/Loss: {gains:>9.2f}$\n')

            txt.write(f'Balance before: {progress[-2][3]:.2f}$. Balance after: {balance:.2f}$. ' +
                      f'Gain/Loss: {balance - progress[-2][3]:.2f}$\n')

            txt.write('}\n\n')

            for row in temp:
                txt.write(row)

    return progress, balance


def plot_data(today, plot_name, time_value_data):
    data = [x[3] for x in time_value_data]
    time_data = [date(year=x[0], month=x[1], day=x[2]) for x in time_value_data]
    x = list(range(0, len(data)))
    log_data = [math.log(d) for d in data]
    log_reg = stats.linregress(x, log_data)
    regression = lambda z: math.e ** (log_reg.intercept + z * log_reg.slope)
    data_bar = [regression(z) for z in x]
    growth = math.e ** log_reg.slope

    # calculate the first day
    first_day = min(time_data)

    helper_list = []
    _max_ind = 0
    _end_ind = 0
    _max_val = 0
    for index, value in enumerate(data):
        counter = 0
        for i in range(index+1, len(data)):
            if data[i] < value:
                counter += 1
            else:
                break
        if counter > _max_val:
            _max_val = counter
            _max_ind = index
            _end_ind = i
        helper_list += [counter]

    _mean = statistics.mean(helper_list)
    _median = statistics.median(helper_list)
    _max1 = max(helper_list)
    _max2 = data[_max_ind]

    data_expanded = [('Portfolio value', time_data[i], data[i]) for i in range(0, len(x))] + [
        (f'Growth per day: {(100*growth)-100:.3f}%', time_data[i], data_bar[i]) for i in range(0, len(x))] + [
        (f'Recovery: max: {_max1}, mean: {_mean:.2f}', i, _max2) for i in [xx for index, xx in enumerate(time_data) if
                                                                           index in range(_max_ind, _end_ind)]]

    df = pd.DataFrame(data_expanded, columns=['type', 'x', 'y'])
    df = df.pivot(index='x', columns='type', values='y')
    plot = df.plot()
    sns.set()

    plot.set(xlabel='date', ylabel='balance ('+'$'+')',
             title=f'Portfolio value from {first_day.year}/{first_day.month:02}/{first_day.day:02} to ' +
                   f'{today.year}/{today.month:02}/{today.day:02}\n' +
                   f'Start: {data[0]:,.2f}'+'$\$$' + f', End: {data[len(data)-1]:,.2f}'+'$\$$')

    plt.gcf().autofmt_xdate(rotation=90)

    try:
        os.mkdir(f'{today.year}_plots')
    except FileExistsError:
        pass

    path = pathlib.Path('.').joinpath(f'{today.year}_plots')

    fig = plot.get_figure()
    fig.savefig(str(path.joinpath(f'{plot_name}_{today.year}-{today.month:02}-{today.day:02}')) + '.png', dpi=300)

    plt.close()
