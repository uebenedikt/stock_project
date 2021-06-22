import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import statistics
import math
import pandas as pd
import numpy as np

import pathlib
import json
import os

from datetime import date
import datetime


def create_plots(initialize_year=2012, training_year=2013, start_year=2014):
    first_day, last_day, progress_list = _plot_progress(start=training_year)
    if first_day is not None:
        alpha_selection_dict = plot_alpha_bots(last_day, start_year, progress_list)
        _plot_significance(last_day, training_year, start_year, alpha_selection_dict, progress_list)


def _plot_progress(start):
    try:
        with open('progress.json', 'r') as json_file:
            progress = json.load(json_file)
    except FileNotFoundError:
        return None

    base_value = min(progress)[3]
    progress_adj = [p for p in progress if date(p[0], p[1], p[2]) >= date(start, 1, 1)]
    if len(progress_adj) > 0:
        p_min = min(progress_adj)
        progress_adj = [(p[0], p[1], p[2], base_value * p[3] / p_min[3]) for p in progress_adj]

        if len(progress_adj) > 3:
            progress = progress_adj

    first_day, today = plot_raw_progress(progress)

    return first_day, today, progress


def plot_raw_progress(progress):
    if len(progress) == 0:
        return None, None

    first_point = min(progress)

    # remove days in the beginning, when no investments occurred
    data = [point for point in progress if point[3] != first_point[3]]

    if len(data) == 0:
        return None, None

    second_point = min(data)
    last_point = max(data)

    # add one point with the base balance, right before the first data point

    first_day = date(year=second_point[0], month=second_point[1], day=second_point[2])
    day_before = first_day - datetime.timedelta(days=1)

    data = [[day_before.year, day_before.month, day_before.day, first_point[3]]] + data

    # load the list of # stocks considered
    try:
        with open('number_of_selected_stocks.json', 'r') as file:
            list_of_length = json.load(file)

            new_list = []

            for tup in list_of_length:
                d = date(year=tup[0], month=tup[1], day=tup[2])
                if d >= day_before:
                    new_list += [(d, tup[3])]
    except FileNotFoundError:
        new_list = []

    # create the plots
    today = date(year=last_point[0], month=last_point[1], day=last_point[2])
    if len(data) > 2:
        plot_data(today, 'complete', data, new_list, daily=False)

    data_current_year = [point for point in data if point[0] == today.year]
    length_list_year = [point for point in new_list if point[0].year == today.year]

    if len(data_current_year) > 2:
        plot_data(today, 'current', data_current_year, length_list_year, daily=False)

    return day_before, today


def plot_alpha_bots(today, start_year, progress):
    progress_dict = generate_truncated_progresses(start_year, progress)

    plot_truncated(progress, progress_dict, start_year, today)

    progress_list = []
    for key, progress in progress_dict.items():
        progress_before = [p for p in progress if p[0] < today.year]
        progress_current = [p for p in progress if p[0] == today.year]
        annual = '    NA    '
        if len(progress_before) > 0:
            p_max = max(progress_before)
            p_min = min(progress_before)
            year_max = p_max[0]
            year_min = p_min[0]
            growth = (p_max[3] / p_min[3]) ** (1.0 / (year_max - year_min + 1))
            annual = f'{growth:>10.6f}'

        current = '    NA    '
        if len(progress_current) > 0:
            growth = max(progress_current)[3] / min(progress_current)[3]
            current = f'{growth:>10.6f}'

        complete = '    NA    '
        if len(progress) > 0:
            growth = max(progress)[3] / min(progress)[3]
            complete = f'{growth:>10.6f}'

        worst = '    NA    '
        if len(progress_before) > 0:
            p_max = max(progress_before)
            p_min = min(progress_before)
            year_max = p_max[0]
            year_min = p_min[0]
            minimum = 1000.0
            for y in range(year_min, year_max + 1):
                year_data = [p for p in progress_before if p[0] == y]
                if len(year_data) > 0:
                    factor = max(year_data)[3] / min(year_data)[3]
                    if factor < minimum:
                        minimum = factor
            worst = f'{minimum:>10.6f}'

        progress_list += [(progress[-1][-1], progress, key, annual, current, complete, worst)]

    progress_list = sorted(progress_list, reverse=True)

    path = pathlib.Path('.').joinpath(f'{today.year}_plots')

    with open(path.joinpath('choice.txt'), 'w+') as txt_file:
        txt_file.write(f'{"Name":<20}  {"Predicted":>10}  {"Current":>10}  {"Complete":>10}  {"Worst Year":>10}\n')
        for bot in progress_list:
            txt_file.write(f'{bot[2]:<20}  {bot[3]}  {bot[4]}  {bot[5]}  {bot[6]}\n')

    alpha_dict = {}
    helper = ['bot $\\alpha$', 'bot $\\beta$', 'bot $\\gamma$']

    for i in range(0, min(3, len(progress_list))):
        alpha_dict[helper[i]] = progress_list[i][1]

    return alpha_dict



def generate_truncated_progresses(start_year, progress):
    # removed


def plot_truncated(progress, careful_dict, start_year, today):
    first_day = date(start_year, 1, 1)
    raw = [point for point in progress if date(point[0], point[1], point[2]) >= first_day]

    for key, data_list in careful_dict.items():
        data = [point for point in data_list if date(point[0], point[1], point[2]) >= first_day]

        if data[0][3] == data[-1][3]:
            continue

        if len(data) > 2:
            plot_data(today, key, data, [], additional_data=raw, daily=False, show=False)


def _plot_significance(last_day, training_year, start_year, alpha_dict, progress):
    progress_dict = {'bot': progress}

    for key, values in alpha_dict.items():
        progress_dict[key] = values

    with open('random_progress.json', 'r') as json_file:
        temp_dict = json.load(json_file)

        for key, progress in temp_dict.items():
            progress_dict[key] = temp_dict[key]

    helper_list = ['full'] + list(range(training_year, last_day.year+1))

    for instruction in helper_list:
        progress_dict_cut = {}
        skip = False
        for key, progress in progress_dict.items():
            skip = False

            if instruction == 'full':
                new_list = [(date(p[0], p[1], p[2]), p[3]) for p in progress if
                            date(p[0], p[1], p[2]) >= date(training_year, 1, 1)]
            else:
                new_list = [(date(p[0], p[1], p[2]), p[3]) for p in progress if p[0] == instruction]
            if len(new_list) < 3 and key == 'bot':
                skip = True
                break
            elif len(new_list) < 3:
                continue

            new_list_adj = [p for p in new_list if p[0] >= date(training_year, 1, 1)]
            if len(new_list_adj) > 3:
                new_list = new_list_adj

            start_value = min(new_list)[1]

            if instruction == 'full' and 'bot' in key and key != 'bot':
                base_value = min(progress_dict['bot'])[3]
                new_list = [(p[0], p[1] / base_value) for p in new_list]
            else:
                new_list = [(p[0], p[1] / start_value) for p in new_list]

            if len(new_list) > 3:
                progress_dict_cut[key] = new_list

        if skip:
            continue

        try:
            x_bar = [p[0] for p in progress_dict_cut['bot']]
        except KeyError:
            break

        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(4, 1)

        ax1 = fig.add_subplot(gs[0:3, 0])

        for key, progress in progress_dict_cut.items():
            x = [p[0] for p in progress]
            y = [p[1] for p in progress]

            if instruction == 'full' and 'bot' in key and key != 'bot':
                y_bar = [p[1] for p in progress_dict_cut['bot']]
                growth = (y[-1] / y_bar[0]) ** (1 / (len(y_bar) - 1)) if len(y_bar) > 1 else 1.0
            else:
                growth = (y[-1] / y[0]) ** (1 / (len(y) - 1)) if len(y) > 1 else 1.0
            ls = '-'  # if key == 'bot' else ':'
            lw = 2 if key == 'bot' else 1
            m = 'o' if 'bot' in key else 'x'
            ms = 3 if key == 'bot' else 2
            ms = 4 if m == 'x' else ms
            lab = f'{growth:.5f}: {key}' if 'bot' in key else f'{growth:.5f}: $n = {key}$'
            al = 1.0  # if key == 'bot' else 0.5
            pl = plt.plot(x, y, label=lab, linestyle=ls, marker=m, markersize=ms, linewidth=lw, alpha=al)

            color = pl[-1].get_color()
            le = int(0.03 * len(x_bar))
            plt.plot((x[-1], x[-1] + datetime.timedelta(days=le)), (y[-1], y[-1]), linestyle='-', color=color, linewidth=lw)

        plt.ylabel('growth factor')
        plt.axhline(y=1.0, linestyle=':')
        if instruction == 'full':
            plt.axvline(x=date(start_year, 1, 1), linestyle=':', color='red', linewidth=3)
        # plt.legend(fontsize='small')
        plt.legend()

        plt.grid()
        if instruction == 'full':
            plt.suptitle('bots vs choosing $n$ stocks randomly each day\nred dotted line marks end of warm-up', fontsize=14)
        else:
            plt.suptitle('bots vs choosing $n$ stocks randomly each day', fontsize=14)

        fig.set_size_inches((12, 6.75), forward=False)

        ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
        if instruction == 'full':
            plt.axvline(x=date(start_year, 1, 1), linestyle=':', color='red', linewidth=3)
        plt.grid()

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(wspace=0, hspace=0)

        try:
            with open('number_of_selected_stocks.json', 'r') as file:
                list_of_length = json.load(file)

                x_val = []
                y_val = []

                for tup in list_of_length:
                    d = date(year=tup[0], month=tup[1], day=tup[2])
                    if instruction == 'full':
                        if d >= date(training_year, 1, 1):
                            x_val += [d]
                            y_val += [tup[3]]
                    else:
                        if d.year == instruction:
                            x_val += [d]
                            y_val += [tup[3]]

                plt.plot(x_val, y_val)
                plt.ylabel('stocks considered')

        except FileNotFoundError:
            pass

        plt.xlabel('\nlegend shows the geometric mean' +
                   '\nbots $\\alpha, \\beta, \\gamma$ reject some of the decisions of the main bot')

        plt.gcf().autofmt_xdate(rotation=90)

        try:
            os.mkdir(f'{last_day.year}_plots')
        except FileExistsError:
            pass

        path = pathlib.Path('.').joinpath(f'{last_day.year}_plots')
        fig.savefig(str(path.joinpath(f'significance_plot_{instruction}.png')), dpi=500)

        plt.close()


def plot_data(today, plot_name, time_value_data, stock_counts, additional_data=[], daily=False, show=True):
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

    # remove duplicate values, we are only interested in actual investments
    helper_list = [data[0]]

    for index in range(1, len(data)):
        if data[index] != helper_list[-1]:
            helper_list += [data[index]]

    growth_list = []

    for index in range(0, len(helper_list) - 1):
        growth_list += [helper_list[index + 1] / helper_list[index]]

    up = len([point for point in growth_list if point > 1])
    down = len([point for point in growth_list if point < 1])
    flat = len(data) - 1 - up - down

    if len(growth_list) == 0:
        growth_list = [1, 1]
    elif len(growth_list) == 1:
        growth_list = growth_list * 2

    helper_list = []
    _max_ind = 0
    _end_ind = 0
    _max_val = 0
    for index, value in enumerate(data):
        counter = 0
        for i in range(index + 1, len(data)):
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
        (f'Growth per day: {(100 * growth) - 100:.3f}%', time_data[i], data_bar[i]) for i in range(0, len(x))] + [
                        (f'Recovery: max: {_max1}, mean: {_mean:.2f}', i, _max2) for i in
                        [xx for index, xx in enumerate(time_data) if
                         index in range(_max_ind, _end_ind)]]

    df = pd.DataFrame(data_expanded, columns=['type', 'x', 'y'])
    df = df.pivot(index='x', columns='type', values='y')
    plot = df.plot()
    plt.grid()

    if len(additional_data) > 3:
        xx = [date(p[0], p[1], p[2]) for p in additional_data]
        yy = [p[3] for p in additional_data]
        plt.plot(xx, yy, label='raw', linestyle=':')
        plt.legend()

    __mean = statistics.mean(growth_list)
    __median = statistics.median(growth_list)
    __stdev = statistics.stdev(growth_list)
    __max = max(growth_list)
    __min = min(growth_list)
    # plot.set(xlabel='date', ylabel='balance',
    #          title=f'Fitted  growth: {growth:.6f}. Start: {data[0]:.2f}USD, '+f'End: {data[len(data)-1]:.2f}USD' +
    #                f'\nGrowth: Mean: {__mean:.6f}, Median: {__median:.6f}, sigma: {__stdev:.6f},' +
    #                f'\nUp: {up}, Flat: {flat}, Down: {down}, MaxUp: {__max:.6f}, MaxDown: {__min:.6f}'
    #          )

    plot.set(xlabel='date', ylabel='balance (' + '$' + ')')

    if up + down > 0:
        fac = 100 * up / (up + down)
    else:
        fac = 0.0

    if data[0] > 0:
        total_growth = data[len(data) - 1] / data[0]
    else:
        total_growth = 0.0

    plt.gcf().autofmt_xdate(rotation=90)

    path = pathlib.Path('.').joinpath(f'{today.year}_plots')

    if show:
        plot.set(  # xlabel='date', ylabel='balance (' + '$' + ')',
            title=f'Portfolio value from {first_day.year}/{first_day.month:02}/{first_day.day:02} to ' +
                  f'{today.year}/{today.month:02}/{today.day:02}\n' +
                  f'Start: {data[0]:,.2f}' + '$\$$' + f', End: {data[len(data) - 1]:,.2f}' + '$\$$')

        fig = plot.get_figure()
        fig.set_size_inches((12, 6.75), forward=False)
        if daily:
            fig.savefig(str(path.joinpath(f'show_{plot_name}_{today.year}-{today.month:02}-{today.day:02}')) + '.png',
                        dpi=300)
        else:
            fig.savefig(str(path.joinpath(f'show_{plot_name}_{today.year}')) + '.png',
                        dpi=300)

    if len(stock_counts) > 0:
        plot.set(title='')
        plot2 = plot.twinx()
        value_list = [point[1] for point in stock_counts]
        plot2.plot([point[0] for point in stock_counts], value_list, color='c')
        plot2.set_ylabel('stocks considered', color='c')
        plot2.set_ylim(top=3 * max(value_list), bottom=0)

        plot2.set(  # xlabel='date', ylabel='balance (' + '$' + ')',
            #         title=f'Portfolio value from {first_day.year}/{first_day.month:02}/{first_day.day:02} to ' +
            #               f'{today.year}/{today.month:02}/{today.day:02}\n' +
            #               f'Start: {data[0]:,.2f}'+'$\$$' + f', End: {data[len(data)-1]:,.2f}'+'$\$$')
            title=f'Est. growth: {growth:.6f}. %up: {fac:.2f}%, ' + f'Fac: {total_growth:.4f}' + f'End: {data[len(data) - 1]:.2f}$\$$' +
                  f'\nGrowth: Mean: {__mean:.6f}, Median: {__median:.6f}, sigma: {__stdev:.6f},' +
                  f'\nUp: {up}, Flat: {flat}, Down: {down}, MaxUp: {__max:.6f}, MaxDown: {__min:.6f}'
        )

        fig = plot2.get_figure()
    else:
        plot.set(  # xlabel='date', ylabel='balance (' + '$' + ')',
            #         title=f'Portfolio value from {first_day.year}/{first_day.month:02}/{first_day.day:02} to ' +
            #               f'{today.year}/{today.month:02}/{today.day:02}\n' +
            #               f'Start: {data[0]:,.2f}'+'$\$$' + f', End: {data[len(data)-1]:,.2f}'+'$\$$')
            title=f'Est. growth: {growth:.6f}. %up: {fac:.2f}%, ' + f'Fac: {total_growth:.4f}' + f'End: {data[len(data) - 1]:.2f}$\$$' +
                  f'\nGrowth: Mean: {__mean:.6f}, Median: {__median:.6f}, sigma: {__stdev:.6f},' +
                  f'\nUp: {up}, Flat: {flat}, Down: {down}, MaxUp: {__max:.6f}, MaxDown: {__min:.6f}'
        )
        fig = plt.gcf()

    fig.set_size_inches((12, 6.75), forward=False)

    if daily:
        fig.savefig(str(path.joinpath(f'{plot_name}_{today.year}-{today.month:02}-{today.day:02}')) + '.png', dpi=300)
    else:
        fig.savefig(str(path.joinpath(f'{plot_name}_{today.year}')) + '.png', dpi=300)

    plt.close()
