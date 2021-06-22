"""
Monitors the current time and starts the appropriate parts of the routine OR executes simulations up until the current
date.

The schedule is as follows (in New York time):

16:15: retrieve the current data for the past trading day. Use it to generate training data and set up an
ensemble Neural Network Estimator, trained on data from the whole last year.

14:00: call for action from the user: sell all stocks, and then enter the current balance into the program. It will be
automatically converted into USD. Then generate a plot to visualize the overall success of the investment strategy

14:15: gather the data from the current day (9:00 - 14:00), and use it to retrain the ensemble estimator. Select an
investment strategy, and calculate which and how many stocks the user should buy. The output is written to the
STRATEGY.txt file.
"""

if __name__ == "__main__":
    import initialize
    import logicModule
    import balancePlot
    import polygonKey
    import alpaca

    import datetime
    from datetime import date
    from datetime import datetime as dt
    import time
    import pytz
    import urllib.request
    import json
    import playsound

    long_term = True
    today = date.today()

    if long_term:
        # to get simulations for the whole last year, 3 months of warm up for strategy management
        current = date(2012, 10, 1)
    else:
        # to get the predictions running as quickly as possible (60 trading days required to set up the strategies)
        current = today - datetime.timedelta(days=90)

    list_of_days = []
    last_day = date(1970, 1, 1)

    initialization_frequency = 1
    last_initialization_day = date(1970, 1, 1)

    # check were the program left off
    try:
        with open('last_day.json', 'r') as json_file:
            done_dict = json.load(json_file)

            year, month, day = done_dict['execute']
            done_day = date(year, month, day)

            if current <= done_day:
                # it is possible, that 'execute' is finished for a day, but 'initialize' is not. allow double-check
                current = done_day

            year, month, day = done_dict['initialize']
            done_day = date(year, month, day)

            if last_initialization_day <= done_day:
                last_initialization_day = done_day
    except (FileNotFoundError, KeyError, ValueError):
        pass

    alpaca_enabled = False
    trading212_enabled = False
    now = dt(year=current.year, month=current.month, day=current.day, hour=0)
    simulation = True
    initialized = False
    balance_requested = False
    balance = 0
    strategy_dict = {}

    end_date = dt(year=2021, month=1, day=1, hour=0)

    while True:
        # important: in the workflow, on a given date, the call to the logic module happens FIRST, THEN the
        # initialization for the next day is performed
        east = pytz.timezone('US/Eastern')
        if not simulation:
            time.sleep(1)
            now = dt.fromtimestamp(time.time()).astimezone(east)
            print('\r' + now.strftime("%Y/%m/%d %H:%M:%S"), end='')
        else:
            real_now = dt.fromtimestamp(time.time()).astimezone(east)
            if now.year == real_now.year and now.month == real_now.month and now.day == real_now.day:
                simulation = False
                continue
            else:
                now += datetime.timedelta(hours=1)

        if now.weekday() in (5, 6):
            continue

        if simulation and now > end_date:
            break

        # happens on the same day as the call to the logic module, but AFTER the call to the logic module
        if not initialized and (now.hour, now.minute) >= (16, 15):
            # important: does not trigger after midnight, to avoid the violations of assumptions made about now_date
            # in relation to the execution of the logicModule, which is set for the next day.
            if not simulation:
                print('')
            print(now.strftime("%Y/%m/%d %H:%M:%S")+' Initialization')
            now_date = date(year=now.year, month=now.month, day=now.day)
            if not simulation or (now_date - last_initialization_day).days >= initialization_frequency:
                initialize.initialize(now_date - datetime.timedelta(days=365), now_date, simulation)
                last_initialization_day = now_date
            initialized = True

        if not simulation and initialized and not balance_requested and (now.hour, now.minute) >= (13, 58):
            if alpaca_enabled:
                alpaca_value = alpaca.sell_all()

            if trading212_enabled:
                print('\nSELL ALL STOCKS NOW!')
                playsound.playsound('gong.wav')

                # balance_eur = 1.0
                # deposits_eur = 0.0
                while True:
                    try:
                        balance_eur = round(float(input('Please enter the current balance in €: ')), 2)
                        break
                    except ValueError:
                        print('Please enter a valid number.')

                while True:
                    try:
                        deposits_eur = round(
                            float(input('Recent changes? (positive: deposits, negative: withdrawal)? ')),
                            2)
                        break
                    except ValueError:
                        print('Please enter a valid amount.')

                balancePlot.plot(broker='trading212', balance=balance_eur, deposits=deposits_eur)

            if alpaca_enabled or trading212_enabled:
                y = now - datetime.timedelta(days=1)
                conv_url = 'https://api.polygon.io/v2/aggs/grouped/locale/global/market/fx/' + \
                           f'{y.year}-{y.month:02}-{y.day:02}?unadjusted=true&apiKey={polygonKey.KEY}'

                try:
                    with urllib.request.urlopen(conv_url) as url:
                        data = json.loads(url.read().decode())

                    results = data['results']
                    for dic in results:
                        if dic['T'] == 'C:EURUSD':
                            if trading212_enabled:
                                balance = balance_eur * dic['c']
                            else:
                                balance = 0
                            exchange_rate = dic['c']
                            break

                    if alpaca_enabled:
                        balancePlot.plot(broker='alpaca', balance=alpaca_value / exchange_rate, deposits=0)
                except:  # HTTPError?
                    print(f'Error: automatic conversion failed. Please enter the balance converted to USD:')

                    while True:
                        try:
                            balance = round(float(input('Please enter the current balance in $USD: ')), 2)
                            break
                        except ValueError:
                            print('Please enter a valid number.')

                balance_requested = True

        minute = 0 if polygonKey.PLAN == 'Developer' else 15
        now_tup = (now.hour, now.minute)
        if initialized and (simulation or balance_requested) and (14, minute) <= now_tup <= (16, 0):
            if not simulation:
                print('')
            print(now.strftime("%Y/%m/%d %H:%M:%S")+f' Call to LogicModule.' +
                  f' Initialization: {str(last_initialization_day)}')
            now_date = date(year=now.year, month=now.month, day=now.day)
            temp = 5000 if simulation else balance

            start_time_complete_ = datetime.datetime.now()
            lm = logicModule.LogicModule()
            buy_strategy = lm.run(today=now_date, last_initialization=last_initialization_day, balance_in_dollar=temp,
                                  simulation=simulation)
            print(f'Time total: {datetime.datetime.now() - start_time_complete_}')

            if not simulation and alpaca_enabled and 14 <= now.hour <= 15 and len(strategy_dict) > 0:
                try:
                    alpaca.buy(buy_strategy)
                except:
                    print('An error occurred trying to use the alpaca API')
                    pass

            lm.save_state(simulation)

            initialized = False
            balance_requested = False
            balance = 0

