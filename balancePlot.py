"""
Creates a plot in the sub-folder 'balancePlot' that summarizes the results of the program and gives an estimate for
the amount of taxes that have to be paid.
"""

import json
import os
import pathlib

import datetime
from datetime import date
from datetime import datetime as dt
import time
import pytz

import matplotlib.pyplot as plt
import seaborn as sns


def plot(type, balance, deposits=0):
    """
    The function that generates the plot and writes it to the disk.
    :param type: string, used to put the plots in different folders
    :param balance: Balance for the day in €
    :param deposits: Today's new deposits (>0) or withdrawals (<0)
    """

    east = pytz.timezone('US/Eastern')
    now = dt.fromtimestamp(time.time()).astimezone(east)
    today = date(year=now.year, month=now.month, day=now.day)

    try:
        os.mkdir(type+'balancePlot')
    except FileExistsError:
        pass

    path = pathlib.Path('.').joinpath(type+'balancePlot')

    # read and write data containing the balances for every day and the deposits for every day
    balance_list = []
    try:
        with open(str(path.joinpath('balance.json')), 'r') as json_file:
            temp = json.load(json_file)
            for point in temp:
                balance_list += [(date(year=point[0][0], month=point[0][1], day=point[0][2]), point[1])]
    except FileNotFoundError:
        pass

    # make sure there are no duplicates
    helper = [point for point in balance_list if point[0] == today]
    if len(helper) == 0:
        balance_list += [(today, balance)]

    with open(str(path.joinpath('balance.json')), 'w') as json_file:
        helper_list = []
        for point in balance_list:
            helper_list += [((point[0].year, point[0].month, point[0].day), point[1])]

        json.dump(helper_list, json_file)

    deposits_list = []
    try:
        with open(str(path.joinpath('deposits.json')), 'r') as json_file:
            temp = json.load(json_file)
            for point in temp:
                deposits_list += [(date(year=point[0][0], month=point[0][1], day=point[0][2]), point[1])]
    except FileNotFoundError:
        pass

    balance_list = sorted(balance_list)
    deposits_list = sorted(deposits_list)

    helper = [point for point in deposits_list if point[0] == today]

    if deposits != 0 and len(helper) == 0:
        deposits_list += [(today, deposits)]

    with open(str(path.joinpath('deposits.json')), 'w') as json_file:
        helper_list = []
        for point in deposits_list:
            helper_list += [((point[0].year, point[0].month, point[0].day), point[1])]

        json.dump(helper_list, json_file)

    # only attempt to plot the progress if there are at least 3 data points
    if len(balance_list) > 3:
        # generate the data for the plot
        daily_earnings_list = []
        daily_earnings_cumulated = []
        tax_list = []
        deposits_cumulated = []

        for index, point in enumerate(balance_list):
            # no earnings on the first day
            if len(daily_earnings_list) == 0:
                daily_earnings_list += [(point[0], 0)]
                continue

            # calculate the earnings for the given day with respect to the deposits
            deposits_this_day = [p[1] for p in deposits_list if p[0] == point[0]]

            if len(deposits_this_day) == 0:
                daily_earnings_list += [(point[0], point[1] - balance_list[index - 1][1])]
            else:
                daily_earnings_list += [(point[0], point[1] - deposits_this_day[0] - balance_list[index - 1][1])]

            # fill the other lists
            temp = get_earnings_cumulated(daily_earnings_list, point[0])
            daily_earnings_cumulated += [(point[0], temp)]
            tax_list += [(point[0], get_tax(temp))]
            deposits_cumulated += [(point[0], cumulated_deposits(deposits_list, point[0]))]

        # create the plot
        sns.set_style('whitegrid')
        sns.set_context('paper')
        # select a range for the data
        last_day = today

        if tax_list[-1][0] - datetime.timedelta(days=730) < tax_list[0][0]:
            # if there is less than two year worth of data, take it all
            first_day = tax_list[0][0]
        else:
            # if there is more than two years worth of data, just take the last two years
            first_day = tax_list[-1][0] - datetime.timedelta(days=730)

        # make four lineplots
        axes = sns.lineplot(x=[point[0] for point in deposits_cumulated if first_day <= point[0] <= last_day],
                            y=[point[1] for point in deposits_cumulated if first_day <= point[0] <= last_day],
                            color='blue')
        plt.text(deposits_cumulated[-1][0], deposits_cumulated[-1][1], f'{deposits_cumulated[-1][1]:>,.2f}€',
                 color='blue')

        sns.lineplot(x=[point[0] for point in balance_list if first_day <= point[0] <= last_day],
                     y=[point[1] for point in balance_list if first_day <= point[0] <= last_day], color='black')
        plt.text(balance_list[-1][0], balance_list[-1][1], f'{balance_list[-1][1]:>,.2f}€', color='black')

        sns.lineplot(x=[point[0] for point in tax_list if first_day <= point[0] <= last_day],
                     y=[point[1] for point in tax_list if first_day <= point[0] <= last_day], color='red')
        plt.text(tax_list[-1][0], tax_list[-1][1], f'{tax_list[-1][1]:>,.2f}€', color='red')

        sns.lineplot(x=[point[0] for point in daily_earnings_cumulated if first_day <= point[0] <= last_day],
                     y=[point[1] for point in daily_earnings_cumulated if first_day <= point[0] <= last_day],
                     color='green')
        plt.text(daily_earnings_cumulated[-1][0], daily_earnings_cumulated[-1][1],
                 f'{daily_earnings_cumulated[-1][1]:>,.2f}€', color='green')

        # decorate the plot
        axes.set(ylabel='€')
        plt.axhline(y=0, color='black', linestyle='-')
        plt.legend(labels=('Investments', 'Balance', 'Tax', 'Earnings'), loc='upper left')
        sns.despine()

        # calculate the taxes for each year for the title-string
        helper_string = 'Taxes: '
        day_current = first_day
        # find all year-end points in the data and and read their tax-value. Put the results in the title string
        while day_current <= last_day:
            day_current += datetime.timedelta(days=1)
            if day_current.month == 12 and day_current.day == 31:
                helper = [point for point in tax_list if point[0] == day_current]
                helper_string += f'{str(day_current)}: {helper[0][1]:,.2f}€   '

        if helper_string == 'Taxes: ':
            axes.set_title(f'Investments and Balance')
        else:
            axes.set_title(f'Investments and Balance\n{helper_string}')

        # set the x-ticks to the first day of each month
        if len(balance_list) > 200:
            helper = []
            helper_day = first_day
            while helper_day <= last_day:
                if helper_day.day == 1:
                    helper += [helper_day]
                helper_month = helper_day.month
                while helper_day.month == helper_month:
                    helper_day += datetime.timedelta(days=1)
            plt.xticks(helper)

        # add text to the plot indicating deposits (grey) and withdrawals (red)
        val_max = max([point for point in balance_list if first_day <= point[0] <= last_day], key=lambda x: x[1])

        for point in deposits_list:
            if not first_day <= point[0] <= last_day:
                continue
            if point[1] > 0:
                plt.axvline(x=point[0], color='grey', linestyle=':')
                plt.text(point[0], 0.80 * val_max[1], f'{point[1]:>+,.2f}€', rotation=90,
                         color='grey')
            elif point[1] < 0:
                plt.axvline(x=point[0], color='red', linestyle=':')
                plt.text(point[0], 0.60 * val_max[1], f' {point[1]:>+,.2f}€', rotation=90, color='red')

        # + datetime.timedelta(days=2)

        # set the x-labels to the date, rotated by 90 degrees
        plt.gcf().autofmt_xdate(rotation=90)

        fig = plt.gcf()
        fig.savefig(str(path.joinpath(f'balancePlot_{today.year}-{today.month:02}-{today.day:02}.png')), dpi=500)

        plt.close()


def cumulated_deposits(deposits_list, today):
    """
    Calculates the cumulated deposits on the whole date up until the most recent day.
    :param deposits_list: list containing the all the deposits and the date they were made.
    :param today: datetime date containing the most recent day
    :return: cumulated sum
    """
    cumsum = 0
    for point in deposits_list:
        # withdrawals
        if point[0] <= today and point[1] > 0:
            cumsum += point[1]

    return cumsum


def get_earnings_cumulated(earnings, today):
    """
    Calculates the cumulated earnings on the year the date 'today' falls in, up until the most recent day. The reasoning
    is that only these earnings are relevant for the tax ('Kapitalertragssteuer').
    :param earnings: list containing the all the earnings and the date they were made.
    :param today: datetime date containing the most recent day
    :return: cumulated sum
    """
    cumsum = 0
    for point in earnings:
        if point[0].year == today.year and point[0] <= today:
            cumsum += point[1]
    return cumsum


def get_tax(earnings_cumulated):
    """
    Calculates the tax.
    """
    to_tax = earnings_cumulated - 801

    return max([0, to_tax * 0.25 * 1.055])
