import alpacaKEY
import os
from portfolio_manager import PortfolioManager
import alpaca_trade_api as trade_api

# establish the connection to the alpaca API
os.environ['APCA_API_KEY_ID'] = alpacaKEY.KEY[0]
os.environ['APCA_API_SECRET_KEY'] = alpacaKEY.SECRET_KEY[0]

# a paper account allows to test alpaca without involving real money
paper = True
if paper:
    api = trade_api.REST(base_url='https://paper-api.alpaca.markets')
    manager = PortfolioManager(paper_account=True)
else:
    api = trade_api.REST()
    manager = PortfolioManager(paper_account=False)


def sell_all():
    """
    Close all open positions / sell all shares currently in the portfolio.
    """
    api.close_all_positions()
    account = api.get_account()
    return float(account.portfolio_value)


def buy(strategy_dict):
    """
    Buy stocks according to the given strategy.
    :param strategy_dict: dictionary: ticker -> [0, 1], where the images of the mappings form a 1 partition
    """

    # make sure the strategy is in the expected form
    s = 0
    for value in strategy_dict.values():
        if not (0 <= value <= 1):
            raise ValueError
        s += value

    if abs(1 - s) >= 0.0001:
        raise ValueError

    rem = strategy_dict.get('remainder', 0)
    if rem < 1.0:
        items_list = []

        for key, value in strategy_dict.items():
            if key == 'remainder':
                continue
            items_list += [[key, value]]

        # rescale to remove any remainder
        if rem > 0:
            for pair in items_list:
                pair[1] = pair[1] / (1-rem)

        manager.add_items(items_list)
        manager.percent_rebalance('block')
    else:
        print('No investments today.')

