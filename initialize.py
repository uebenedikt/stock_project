import trading212
from StockSnippet import StockSnippet
import polygonKey
import ensembleEstimator

from datetime import date
import datetime
import time
import pathlib
import json
import threading


def initialize(start_date, end_date, simulation, reset=False, update=False):
    """
    tasks:
    - create a list of stocks
    - update the data for these stocks
    - remove unused data points, fill missing ones by interpolation
    - remove days with overall sparse data ('no-trade days')
    - reduce the stock list to those stocks that meet certain quality criteria (completeness of data)
    - generate the data for training, and plot a summary of the response values, to see their distribution
    - call the method that is responsible for the creation of the estimators, that all together form the ensemble
    estimator

    :param start_date: start_date and end_date define the time-span from which the data to train the estimators is taken
    :param end_date: usually this represents the current date, but in simulations it could be any date
    :param simulation: whether the current execution is part of a simulation or not
    """

    # check whether day was already executed
    try:
        with open('last_day.json', 'r') as json_file:
            done_dict = json.load(json_file)

            year, month, day = done_dict['initialize']
            this_day = date(int(year), int(month), int(day))
            if this_day >= end_date:
                print('The initializations for this day were already performed, program will go back to main.')
                return None
    except FileNotFoundError:
        done_dict = {}
    except KeyError:
        pass

    print('Creating data...')
    list_of_names = trading212.get_stock_list()

    stock_dict = {}

    # create a mapping ticker -> name, ie AAPL -> Apple
    for n in list_of_names:
        stock_dict[n[1]] = n[0]

    # for convenience
    with open('stock_dictionary.json', 'w') as file:
        json.dump(stock_dict, file)

    # create a list of the stocks, elements of format: (name, ticker, category)
    list_of_stocks = [StockSnippet(n[0], n[1], n[2]) for n in list_of_names]
    print('Creating data: done.')

    # download the most recent data (2 years back by default)
    if update or polygonKey.PREMIUM:
        print('Updating data...')
        counter = 0

        threads = []
        for stock in list_of_stocks:
            stock.clean_splits()
            stock.detect_stock_split()
            thread = threading.Thread(target=stock.update, args=(False, None, None, True))
            if len(threads) < 10:
                thread.start()
                threads += [thread]
            else:
                threads[0].join()
                counter += 1
                del threads[0]
                thread.start()
                threads += [thread]
                print(f'\rUpdating data... {100 * counter / len(list_of_stocks):.2f}%', end='')

        for i in range(len(threads)):
            threads[i].join()
            counter += 1
            print(f'\rUpdating data... {100 * counter / len(list_of_stocks):.2f}%', end='')

        print('Updating data: done.')

    # edit data: discard unused data points, interpolate missing data points
    print('Editing data...')
    for stock in list_of_stocks:
        stock.edit_data(start_date, end_date)
    print('Editing data: done.')

    print('Remove no-trade days from the data...')
    # first, try to load a list of already found no-trade days
    try:
        with open('missing_days.json', 'r') as file:
            loaded_list = json.load(file)
            # first two points define the already covered time span
            first_day = datetime.date(year=loaded_list[0][0], month=loaded_list[0][1], day=loaded_list[0][2])
            last_day = datetime.date(year=loaded_list[1][0], month=loaded_list[1][1], day=loaded_list[1][2])
            # third point is a list containing the missing days
            missing_days_all = [datetime.date(year=i[0], month=i[1], day=i[2]) for i in loaded_list[2]]
            missing_days_in_scope = [i for i in missing_days_all if start_date <= i <= end_date]
    except FileNotFoundError:
        print('There is no data containing the missing days.')
        first_day = start_date
        last_day = start_date - datetime.timedelta(days=1)
        missing_days_all = []
        missing_days_in_scope = []

    # remove stocks that contain not data
    list_of_stocks = [stock for stock in list_of_stocks if not stock.isEmpty]

    # remove no-trade days
    helper_list = []
    helper_set = set()
    no_trade_days = []
    starting_date_for_search = last_day + datetime.timedelta(days=1)

    for stock in list_of_stocks:
        temp = stock.possible_no_trade_days(starting_date_for_search, end_date)
        helper_list += [temp]
        helper_set.update(temp)

    for day_obj in helper_set:
        # sparse data for at least 80% of the stocks?
        # do more than 80% of the stocks mark a specific day as missing?
        if len([lists for lists in helper_list if day_obj in lists]) >= 0.8*len(helper_list):
            print('Found a no trade day! ' + str(day_obj))
            no_trade_days += [day_obj]

    no_trade_days_in_scope = no_trade_days + missing_days_in_scope
    for stock in list_of_stocks:
        stock.remove_no_trade_days(today=end_date, no_trade_days=no_trade_days_in_scope, save=True)

    no_trade_days_complete = missing_days_all + no_trade_days
    list_of_tuples = [(i.year, i.month, i.day) for i in no_trade_days_complete]

    to_dump = [[first_day.year, first_day.month, first_day.day], [end_date.year, end_date.month, end_date.day],
               list_of_tuples]

    with open('missing_days.json', 'w+') as file:
        json.dump(to_dump, file)

    print('Remove no-trade days from the data: done.')

    # select stocks that meet the quality criteria
    print('Selecting data...')
    # check the quality criteria for each stock
    selected_stocks = [stock for stock in list_of_stocks if (stock.missing_data < 0.3 and stock.missing_data_month < 0.2
                       and stock.missing_data_week < 0.1)]

    counter = 0
    for stock in list_of_stocks:
        if stock not in selected_stocks:
            counter += 1

    print(f'Removed {counter} stocks because of missing data.')

    # save a list of the selected stocks as .json for later use
    path = pathlib.Path('.')

    helper_list = [[stock.name, stock.abbreviation, stock.category] for stock in selected_stocks]
    with open(str(path.joinpath('selected_stocks')) + '.json', 'w+') as file:
        json.dump(helper_list, file)

    print('Selecting data: done.')
    print(f'Number of selected stocks: {len(selected_stocks)}')

    if not reset and ensembleEstimator.exists(end_date):
        # implies that training data already exists
        with open('last_day.json', 'w+') as json_file:
            done_dict['initialize'] = (end_date.year, end_date.month, end_date.day)
            json.dump(done_dict, json_file)
        return None

    print(f'Generate training data...')
    # how many different data categories are there?
    number_of_categories = -1

    for index in range(0, len(list_of_names)):
        if list_of_names[index][2] > number_of_categories:
            number_of_categories = list_of_names[index][2]

    number_of_categories += 1

    # create training data for each day within the time span
    # the training data for a day contains information about the previous 30 days, therefore we need an offset here
    current = start_date + datetime.timedelta(days=30)
    list_of_all_days = []

    while current < end_date:
        # a training point consists of data for the current day, plus a response from the next day
        # the most recent training point that we can get is therefore the day before the end day, including the
        # response from the end day.
        if current.weekday() < 5:
            list_of_all_days += [current]
        current = current + datetime.timedelta(days=1)

    counter = 0

    for stock in selected_stocks:
        stock.generate_training_data(date_list=list_of_all_days, number_of_categories=number_of_categories,
                                     save=True)
        counter += 1
        print(f'\rGenerating training data... {100 * counter / len(selected_stocks):.2f}%', end='')

    list_of_all_days = sorted(list_of_all_days)

    print(f'\nGenerate training data: done.')

    # with everything set up, create the ensemble estimator
    print('Initialize the ensemble estimator...')
    ens = ensembleEstimator.EnsembleEstimator(today=end_date)
    ens.initialize(list_of_days=list_of_all_days, stocks=selected_stocks, categories=number_of_categories)

    print('Initialize the ensemble estimator: done.')

    # to make the check possible, whether this routine was already executed
    with open('last_day.json', 'w+') as json_file:
        done_dict['initialize'] = (end_date.year, end_date.month, end_date.day)
        json.dump(done_dict, json_file)
