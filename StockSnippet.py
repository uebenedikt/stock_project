"""
Contains the class StockSnippet and the necessary functionality, which includes:
- downloading and updating the stock data
- processing the stock data to generate training data
"""
import polygonKey

import time
import pytz
import datetime
from datetime import date
from datetime import datetime as dt
import os
import pathlib
import urllib.request
import json
import random

import statistics as stats
import re


def get_next_time(current):
    """
    Returns the next date and time. For example: input = datetime(2020,2,11,16), output: datetime(2020,2,12,9)
    :param current: the datetime object that needs to be updated
    :return: datetime-object containing the next timestamp
    """
    current += datetime.timedelta(hours=1)

    if current.hour > 16:
        current += datetime.timedelta(days=1)
        current += datetime.timedelta(hours=9 - current.hour)

    while current.weekday() >= 5:
        current += datetime.timedelta(days=1)

    return current


def get_previous_time(current):
    """
    Returns the previous date and time. For example: input = datetime(2020,2,12,9), output: datetime(2020,2,11,16)
    :param current: the datetime object that needs to be updated
    :return: datetime-object containing the next timestamp
    """
    current += datetime.timedelta(hours=-1)

    if current.hour < 9:
        current += datetime.timedelta(days=-1)
        current += datetime.timedelta(hours=16 - current.hour)  # sets the time to 16:00

    while current.weekday() >= 5:
        current += datetime.timedelta(days=-1)

    return current


def get_next_day(current):
    """
    Returns the next date. For example: input = date(2020,2,11), output: date(2020,2,12)
    :param current: the date object that needs to be updated
    :return: date-object containing the next timestamp
    """
    current += datetime.timedelta(days=1)

    while current.weekday() >= 5:
        current += datetime.timedelta(days=1)

    return current


def get_previous_day(current):
    """
    Returns the precious date. For example: input = date(2020,2,12), output: date(2020,2,11)
    :param current: the date object that needs to be updated
    :return: date-object containing the previous timestamp
    """
    current -= datetime.timedelta(days=1)

    while current.weekday() >= 5:
        current -= datetime.timedelta(days=1)

    return current


def get_datetime_from_date(date_obj, start=9, end=16):
    """
    For a given date-object returns a list of datetime objects with the range of specified hour-entries
    :param date_obj: The date-object
    :param start: The first hour to be included
    :param end: The last hour to be included
    :return:
    """
    return [dt(year=date_obj.year, month=date_obj.month, day=date_obj.day, hour=t) for t in range(start, end+1)]


def is_match(t1, t2):
    """
    Checks whether two time stamps in tuple form, ie (2021, 2, 21, 14), are identical.
    The tuples given are NOT checked for equality, only the first 4 dimensions are compared!
    :param t1: First tuple
    :param t2: Second tuple
    :return: True/False
    """
    return t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2] and t1[3] == t2[3]


def is_in(this, t1, list_of_tuples):
    """
    Checks whether the time data of one tuple matches the time date of another tuple within a list.
    This method is used to check for discrepancies between older data an freshly downloaded data.
    If matches are found, the input list will be altered!
    :param this: the stock object calling the method
    :param t1: the new tuple
    :param list_of_tuples: list of older tuples
    :return: True, if the tuple matches, False, if not
    """
    for index, t2 in enumerate(list_of_tuples):
        if is_match(t1, t2):
            if t1[4] != t2[4]:
                # print(f'Fixed one data point with a distance of {abs(t1[4] - t2[4])} for {this.name}.')
                list_of_tuples[index] = t1
            return True
    return False


def calculate_sstd(base_list, period_length=8, weighted=True):
    # removed


def calculate_stictoc(base_list, period=8, weighted=True):
    # removed


def calculate_sgrowth(base_list, period=8, weighted=True):
    # removed


def calculate_smean(base_list, weighted=True):
    # removed


class StockSnippet:
    # this dictionary contains the last day an update was performed for each instance of the class
    last_day_updated_dict = {}

    def __init__(self, name, abbreviation, category_, recent):
        self.__name = name
        self.__abbreviation = abbreviation
        self.__raw_data = []
        self.__edited_data = {}
        self.__training_data = {}
        self.missing_data = 0
        self.missing_data_month = 0
        self.missing_data_week = 0
        self.category = category_
        self.__load_data(recent=recent)
        self.isEmpty = False
        StockSnippet.last_day_updated_dict[abbreviation] = date(1970, 1, 1)

    def is_empty(self):
        return len(self.__raw_data) == 0

    def __load_data(self, recent):
        """
        Reads saved raw data from the folder 'stocks_raw'.
        """
        try:
            os.mkdir('stocks_recent')
        except FileExistsError:
            pass

        try:
            os.mkdir('stocks_raw')
        except FileExistsError:
            pass

        if recent:
            suffix = '_recent.json'
            folder = 'stocks_recent'
        else:
            suffix = '_raw.json'
            folder = 'stocks_raw'

        path = pathlib.Path('.').joinpath(folder)
        try:
            with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + suffix, 'r') as file:
                list_of_lists = json.load(file)
                self.__raw_data = sorted([tuple(list_) for list_ in list_of_lists], reverse=True)
        except FileNotFoundError:
            with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + suffix, 'x') as file:
                json.dump([], file)

    def clean_splits(self):
        """
        Checks for stock-splits, and deletes data that was downloaded before the split.
        """
        if polygonKey.PREMIUM:
            url = f'https://api.polygon.io/v2/reference/splits/{self.__abbreviation}?&apiKey={polygonKey.KEY}'

            # read from URL
            try:
                with urllib.request.urlopen(url) as url:
                    data = json.loads(url.read().decode())
            except:  # HTTPError?
                # a lot can go wrong here, but we don't want that to lead to termination of the whole program
                print(f'Something went wrong with the stock split URL to {self.__abbreviation}')
                return None

            # are there any stock splits?
            if data['count'] > 0:
                day_list = []

                try:
                    for point in data['results']:
                        # parse the date
                        result = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(point['exDate']))
                        helper_list = [int(i) for i in result.groups()]
                        day_list += [date(year=helper_list[0], month=helper_list[1], day=helper_list[2])]
                except KeyError:
                    print(f'Something went wrong with the stock split data for {self.__abbreviation}')

                today = date.today()
                recent_split = False
                split_day = date(1970, 1, 1)

                # check if any of the splits happened recently. cover more days to reduce the risk of missing a split
                for day in day_list:
                    if (today - day).days < 7:
                        recent_split = True
                        split_day = day
                        break

                if recent_split:
                    print(f'Found a recent stock split for {self.abbreviation} on {str(split_day)}. Erasing all data.')
                    # overwrite the raw data
                    path = pathlib.Path('.').joinpath('stocks_raw')
                    with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + '_raw.json', 'w') as file:
                        json.dump([], file)

    def detect_stock_split(self):
        """

        :return:
        """
        if date.today == StockSnippet.last_day_updated_dict[self.__abbreviation]:
            # since the update is performed after the check for anomalies the check has already been performed
            return None

        if len(self.__raw_data) == 0:
            return None

        lower_index = int(0.5*len(self.__raw_data))
        upper_index = len(self.__raw_data) - 1
        target = random.randint(lower_index, upper_index)

        datapoint = self.__raw_data[target]

        data_day = date(year=datapoint[0], month=datapoint[1], day=datapoint[2])
        start = data_day - datetime.timedelta(days=4)
        end = data_day + datetime.timedelta(days=4)

        data_to_check = [point for point in self.__raw_data if
                         start <= date(year=point[0], month=point[1], day=point[2]) <= end]

        stock_url = f'https://api.polygon.io/v2/aggs/ticker/{self.abbreviation}/range/1/hour/' + \
                    f'{start.year}-{start.month:02}-{start.day:02}/' + \
                    f'{end.year}-{end.month:02}-{end.day:02}' + \
                    f'?unadjusted=false&sort=asc&apiKey={polygonKey.KEY}'

        # read from URL
        try:
            if not polygonKey.PREMIUM:
                time.sleep(12)
            with urllib.request.urlopen(stock_url) as url:
                data = json.loads(url.read().decode())
        except:  # HTTPError?
            # a lot can go wrong here, but we don't want that to lead to termination of the whole program
            print(f'Something went wrong with the URL to {self.abbreviation}')
            return None

        # for time conversion. the data contains a UTC timestamp in milliseconds
        east = pytz.timezone('US/Eastern')

        found_list = []

        try:
            for point in data['results']:
                try:
                    # try to get a mean value
                    value = float(point['vw'])
                except KeyError:
                    try:
                        # if you can't get the mean value, take the open-price
                        value = float(point['o'])
                    except KeyError:
                        # no usable data in this point
                        continue
                try:
                    # converse the timestamp to New-York time
                    date_of_value = dt.fromtimestamp(point['t'] / 1000).astimezone(east)
                except KeyError:
                    continue

                # extract YEAR-MONTH-DAY-HOUR from the timestamp
                result = re.search(r'(\d{4})-(\d{2})-(\d{2}) (\d{2})', str(date_of_value))
                helper_list = [int(i) for i in result.groups()]
                # put the data in a list containing the results
                found_list += [tuple(helper_list + [value])]
        except KeyError:
            return None

        max_deviation = 0
        max_day = start

        for point in found_list:
            match = [p for p in data_to_check if
                     p[0] == point[0] and p[1] == point[1] and p[2] == point[2] and p[3] == point[3]]

            if len(match) > 0:
                match = match[0]
            else:
                continue

            deviation = abs(1 - point[4]/match[4])

            if deviation > max_deviation:
                max_deviation = deviation
                max_day = date(year=point[0], month=point[1], day=point[2])

        if max_deviation >= 0.1:
            print(f'{self.abbreviation}: Found a deviation of {100*max_deviation:.2f}% on ' +
                  f'{max_day.year}/{max_day.month}/{max_day.day}.')
            print(f'{self.abbreviation}: Stock anomaly detected, delete old data.')

            self.__raw_data = []
            path = pathlib.Path('.').joinpath('stocks_raw')
            with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + '_raw.json', 'w') as file:
                json.dump([], file)

            path = pathlib.Path('.').joinpath('stocks_recent')
            with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + '_recent.json', 'w') as file:
                json.dump([], file)
        else:
            pass
            # print(f'{self.abbreviation}: No stock anomaly detected.')

    def update(self, save=False, save_recent=True, old_data=False):
        """
        Loads data from polygon.io.
        :param use_dates: if True, the given dates are used as start- and end-points. If not, 'end' is today, and
        'start' is set to the date two years back.
        :param start: specifies a start date
        :param end: specifies a end date
        """

        today = date.today()

        if len(self.__raw_data) > 1:
            self.__raw_data = sorted(self.__raw_data, reverse=True)

        # we want the data of the interval (start, end)
        if old_data:
            if len(self.__raw_data) == 0:
                end = today
            else:
                end = date(self.__raw_data[-1][0], self.__raw_data[-1][1], self.__raw_data[-1][2])
            start = date(today.year - 60, 1, 1)
        else:
            end = today
            if len(self.__raw_data) == 0:
                start = date(today.year - 60, 1, 1)
            else:
                start = date(self.__raw_data[0][0], self.__raw_data[0][1], self.__raw_data[0][2])

            if end <= StockSnippet.last_day_updated_dict[self.__abbreviation]:
                # in this case the update for this class was already performed today
                return None

        end_iter = end
        step_size = 10
        attempt_count = 0
        empty_response_count = 0

        while end_iter >= start:
            # two empty responses in a row -> stop
            if empty_response_count > 1:
                break

            # to get a 10-day window
            start_iter = end_iter - datetime.timedelta(days=step_size)

            stock_url = f'https://api.polygon.io/v2/aggs/ticker/{self.abbreviation}/range/1/hour/' + \
                        f'{start_iter.year}-{start_iter.month:02}-{start_iter.day:02}/' + \
                        f'{end_iter.year}-{end_iter.month:02}-{end_iter.day:02}' + \
                        f'?unadjusted=false&sort=asc&apiKey={polygonKey.KEY}'

            # read from URL
            try:
                if not polygonKey.PREMIUM:
                    time.sleep(12)
                with urllib.request.urlopen(stock_url, timeout=5) as url:
                    data = json.loads(url.read().decode())
            except:  # HTTPError?
                # a lot can go wrong here, but we don't want that to lead to termination of the whole program
                # print(f'Something went wrong with the URL to {self.abbreviation}')
                attempt_count += 1
                if attempt_count > 4:
                    print(f"Couldn't resolve an issue with {self.abbreviation}")
                    break
                else:
                    continue

            attempt_count = 0

            # for time conversion. the data contains a UTC timestamp in milliseconds
            east = pytz.timezone('US/Eastern')

            new_list = []

            try:
                for point in data['results']:
                    try:
                        # try to get a mean value
                        value = float(point['vw'])
                    except KeyError:
                        try:
                            # if you can't get the mean value, take the open-price
                            value = float(point['o'])
                        except KeyError:
                            # no usable data in this point
                            continue
                    try:
                        # converse the timestamp to New-York time
                        date_of_value = dt.fromtimestamp(point['t'] / 1000).astimezone(east)
                    except KeyError:
                        continue

                    # extract YEAR-MONTH-DAY-HOUR from the timestamp
                    result = re.search(r'(\d{4})-(\d{2})-(\d{2}) (\d{2})', str(date_of_value))
                    helper_list = [int(i) for i in result.groups()]
                    # put the data in a list containing the results
                    new_list += [tuple(helper_list + [value])]
            except KeyError:
                # happens, if there is no data at all in the downloaded data
                # print(f'KeyError: No data for {self.name} in the time from {str(start_iter)} to {str(end_iter)}')
                # print(data)
                end_iter -= datetime.timedelta(days=step_size)
                empty_response_count += 1
                continue

            if len(new_list) == 0:
                # make sure there really is data to process
                print(f'No usable data for {self.name} in the time from {str(start_iter)} to {str(end_iter)}')
                end_iter -= datetime.timedelta(days=step_size)
                empty_response_count += 1
                continue

            # print(f'Added {counter} data points for ' + self.name + ' from Polygon.')

            self.__raw_data += new_list
            empty_response_count = 0
            end_iter -= datetime.timedelta(days=step_size)

        # always keep the data sorted
        self.__raw_data = sorted(self.__raw_data, reverse=True)

        # check for duplicates and remove them
        remove = []
        for i in range(0, len(self.__raw_data)):
            t1 = self.__raw_data[i]
            for j in range(i + 1, len(self.__raw_data)):
                t2 = self.__raw_data[j]
                if t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2] and t1[3] == t2[3]:
                    remove += [j]
                else:
                    break

        if len(remove) > 0:
            self.__raw_data = [tup for index, tup in enumerate(self.__raw_data) if index not in remove]

        if save:
            # important: it's all or nothing, saving incomplete results might lead to gaps in the data,
            # if an interrpution occurs
            # save the raw_data list as .json file
            path = pathlib.Path('.').joinpath('stocks_raw')
            try:
                os.mkdir('stocks_raw')
            except FileExistsError:
                pass
            with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + '_raw.json', 'w') as file:
                json.dump(self.__raw_data, file)

            if save_recent:
                path = pathlib.Path('.').joinpath('stocks_recent')
                try:
                    os.mkdir('stocks_recent')
                except FileExistsError:
                    pass
                with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + '_recent.json', 'w') as file:
                    json.dump(self.__raw_data[0:500], file)

            StockSnippet.last_day_updated_dict[self.__abbreviation] = end

    def simulate_update(self, today):
        start_date = today - datetime.timedelta(days=60)
        end_date = today + datetime.timedelta(days=7)  # call of logicModule on the next day expects to find the data
        raw_short = [point for point in self.__raw_data
                     if start_date <= date(year=point[0], month=point[1], day=point[2]) <= end_date]
        raw_short = sorted(raw_short, reverse=True)
        path = pathlib.Path('.').joinpath('stocks_recent')
        try:
            os.mkdir('stocks_recent')
        except FileExistsError:
            pass
        with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + '_recent.json', 'w') as file:
            json.dump(raw_short, file)

    def edit_data(self, start_date, end_date):
        """
        Calls functions that remove unwanted data points from the raw data and try to fill gaps in the data.
        :param start_date: indicate the time-frame the operation is applied to.
        :param end_date: See above.
        """
        # apply filter
        filtered_list = self.filter_list(start_date, end_date)

        if len(filtered_list) == 0:
            self.isEmpty = True

        # fill gaps in the data
        try:
            self.fill(filtered_list, start_date, end_date)
        except IndexError:
            pass

    def filter_list(self, start_date, end_date):
        """
        Returns only those data points that fall in the defined time span and have a time-value between 9 and 16
        (trading hours)
        :param start_date: Define the time span.
        :param end_date: See above.
        :return: filtered list
        """
        filtered_list = []

        for point in self.__raw_data:
            date_of_observation = date(point[0], point[1], point[2])

            if start_date <= date_of_observation <= end_date and date_of_observation.weekday() < 5:
                if 9 <= point[3] <= 16:
                    filtered_list += [point]

        return filtered_list

    def fill(self, filtered_list, start_date, end_date):
        """
        Finds missing data points and fills them with plausible values. Result is in self.edited_data.
        :param filtered_list: The filtered list generated by the function filter_list
        :param start_date: Defines the time span
        :param end_date: See above.
        """
        filled_list = []  # will contain the results
        filtered_list = sorted(filtered_list)

        # define the exact start- and end-points
        start_time = dt(year=start_date.year, month=start_date.month, day=start_date.day, hour=9)
        end_time = dt(year=end_date.year, month=end_date.month, day=end_date.day, hour=16)

        # fill edges first, filtered_list contains oldest data first!
        first = filtered_list[0]
        last = filtered_list[-1]

        first_time = dt(year=first[0], month=first[1], day=first[2], hour=first[3])
        last_time = dt(year=last[0], month=last[1], day=last[2], hour=last[3])

        # fill the gaps with the first/last value available. mark added data points by '**'
        while start_time < first_time:
            filled_list += [(start_time.year, start_time.month, start_time.day, start_time.hour, first[4], '**')]
            start_time = get_next_time(start_time)

        while last_time < end_time:
            filled_list += [(end_time.year, end_time.month, end_time.day, end_time.hour, last[4], '**')]
            end_time = get_previous_time(end_time)

        # now, fill the gaps in between data points. Only makes sense if there are at least two data points
        if len(filtered_list) >= 2:
            for index in range(0, len(filtered_list) - 1):
                # count number of time points in between
                a = filtered_list[index]
                b = filtered_list[index+1]
                time_a = dt(year=a[0], month=a[1], day=a[2], hour=a[3])
                time_b = dt(year=b[0], month=b[1], day=b[2], hour=b[3])

                current = time_a
                count = -1

                while current < time_b:
                    current = get_next_time(current)
                    count += 1

                # is there a gap?
                if count > 0:
                    # interpolate the two values on the edge of the gap
                    base_value = a[4]
                    value_delta = (b[4] - a[4])/(count+1)

                    current = time_a
                    count = 0
                    maxi = get_previous_time(time_b)

                    while current < maxi:
                        current = get_next_time(current)
                        count += 1

                        filled_list += [(current.year, current.month, current.day, current.hour, base_value +
                                         count*value_delta, '**')]

        # add the filtered list to the list with filler points
        filled_list += filtered_list

        # transform into a dictionary
        for point in filled_list:
            if len(point) == 5:
                self.__edited_data[dt(year=point[0], month=point[1], day=point[2], hour=point[3])] = (point[4],)
            elif len(point) == 6:
                self.__edited_data[dt(year=point[0], month=point[1], day=point[2], hour=point[3])] = (point[4], '**')

    def possible_no_trade_days(self, start_date, end_date):
        """
        Attempts to identify possible days on which very little to no trades occurred.
        :param start_date: defines the time frame for the operation
        :param end_date: see above
        :return: list of potential days on which the markets were closed
        """
        current = start_date
        result = []

        while current <= end_date:
            # for each day, count the number of points with filler values
            dt_list = get_datetime_from_date(current)  # hour-steps between 9 and 16
            day_list = [self.__edited_data.get(t, []) for t in dt_list]  # elements are tuples

            count = len([point for point in day_list if len(point) == 2])  # a tuple of length 2 indicates added data

            # more than half missing? possible no-trade day!
            if count > 4:
                result += [current]

            current = get_next_day(current)

        return result

    def remove_no_trade_days(self, today, no_trade_days, save=False):
        """
        Given a list of no-trade days, these days are removed from the data, and the remaining number of missing points
        is counted on three levels: all time, last month and last week. After that, the remaining data is written to a
        .txt file, to allow the user to have a look at the data.
        :param today: marks the end-day for this search. Important for simulations, in case the data exceeds the current
        day
        :param no_trade_days: list of no trade days, compiled from all the stocks
        """
        for day in no_trade_days:
            dt_list = get_datetime_from_date(day)
            for dt_obj in dt_list:
                if len(self.__edited_data.get(dt_obj, [])) > 0:
                    del self.__edited_data[dt_obj]

        end_time = dt(year=today.year, month=today.month, day=today.day, hour=23)

        start_day_month = today - datetime.timedelta(days=30)
        start_time_month = dt(year=start_day_month.year, month=start_day_month.month, day=start_day_month.day, hour=0)
        start_day_week = today - datetime.timedelta(days=7)
        start_time_week = dt(year=start_day_week.year, month=start_day_week.month, day=start_day_week.day, hour=0)

        total = 0
        missing_temp = 0
        total_month = 0
        missing_month_temp = 0
        total_week = 0
        missing_week_temp = 0
        for key, value in self.__edited_data.items():
            # count the added data points in the whole remaining data
            total += 1
            if len(value) == 2:
                missing_temp += 1

            # count the added data in the last month
            if start_time_month <= key <= end_time:
                total_month += 1
                if len(value) == 2:
                    missing_month_temp += 1

            # count the added data in the last week
            if start_time_week <= key <= end_time:
                total_week += 1
                if len(value) == 2:
                    missing_week_temp += 1

        self.missing_data = 1.0 if total == 0 else missing_temp / total
        self.missing_data_month = 1.0 if total_month == 0 else missing_month_temp / total_month
        self.missing_data_week = 1.0 if total_week == 0 else missing_week_temp / total_week

        if save:
            # transform the dictionary into a list to get sorted data
            helper = []
            for key, value in self.__edited_data.items():
                helper += [(key.year, key.month, key.day, key.hour, *value)]
            helper = sorted(helper, reverse=True)

            # write the data to a .txt file
            try:
                os.mkdir('edited_data')
            except FileExistsError:
                pass
            path = pathlib.Path('.').joinpath('edited_data')
            with open(str(path.joinpath(self.abbreviation.replace('.', '0')))+'.txt', 'w+') as txtfile:
                for point in helper:
                    txtfile.write(str(point)+'\n')

    def load_training_data(self, recent=False):
        """
        Loads the already generated training data from a file and removes data older than 3 years.
        """
        if not recent:
            folder = 'training_data'
            suffix = '_t.json'
        else:
            folder = 'training_data_recent'
            suffix = '_t_recent.json'

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        path = pathlib.Path('.').joinpath(folder)
        try:
            with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + suffix, 'r') as file:
                helper_list = json.load(file)
                helper_dict = {}
                for point in helper_list:
                    date_obj = date(year=point[1], month=point[2], day=point[3])
                    helper_dict[date_obj] = point

                self.__training_data = helper_dict
        except:
            # data could be damaged because of an interruption
            # print('No training data for ' + self.name + '.')
            with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + suffix, 'x') as file:
                json.dump([], file)

    def return_training_data(self, date_obj):
        """
        Returns the training data for a specified date.
        :param date_obj: The specified date
        :return: List of matching data points
        """
        if len(self.__training_data) == 0:
            self.load_training_data()
        return self.__training_data.get(date_obj, [])

    def simulate_generation_of_trainingdata(self, today):
        if len(self.__training_data) == 0:
            try:
                os.mkdir('training_data_recent')
            except FileExistsError:
                pass
            # if it is an simulation, we want to reuse precalculated data
            self.load_training_data(recent=False)

        center_day = today
        left = center_day - datetime.timedelta(days=15)
        right = center_day + datetime.timedelta(days=7)

        data_list = list(self.__training_data.values())
        data_list = [p for p in data_list if left <= date(year=p[1], month=p[2], day=p[3]) <= right]

        path = pathlib.Path('.').joinpath('training_data_recent')

        with open(str(path.joinpath(self.abbreviation.replace('.', '0'))) + '_t_recent.json', 'w') as file:
            json.dump(data_list, file)


    def generate_training_data(self, date_list, number_of_categories, get_response=True, save=False, recent=False):
        # removed

    def get_price(self, year, month, day, hour):
        tup = self.__edited_data.get(dt(year=year, month=month, day=day, hour=hour), [])

        if len(tup) > 0:
            return tup[0]
        else:
            raise IndexError

    @property
    def name(self, ):
        return self.__name

    @property
    def abbreviation(self):
        return self.__abbreviation

    @property
    def edited_data(self):
        return self.__edited_data
