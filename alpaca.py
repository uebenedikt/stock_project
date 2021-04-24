import alpacaKEY
import os
from portfolio_manager import PortfolioManager
import alpaca_trade_api as trade_api

os.environ['APCA_API_KEY_ID'] = alpacaKEY.KEY[0]  # 'API_KEY_ID'
os.environ['APCA_API_SECRET_KEY'] = alpacaKEY.SECRET_KEY[0]  # 'API_SECRET_KEY'
paper = True

if paper:
    api = trade_api.REST(base_url='https://paper-api.alpaca.markets')
    manager = PortfolioManager(paper_account=True)
else:
    api = trade_api.REST()
    manager = PortfolioManager(paper_account=False)


def sell_all():
    """
    Sell all shares of all stocks.
    """
    api.close_all_positions()
    account = api.get_account()
    return float(account.portfolio_value)


def buy(strategy_dict):
    """
    Buy stocks according to the given strategy.
    :param strategy_dict: map: ticker -> 1 partition
    """

    items_list = []

    for key, value in strategy_dict.items():
        if key == 'remainder':
            continue
        items_list += [[key, value]]

    rem = strategy_dict.get('remainder', 0)

    if rem > 0:
        for pair in items_list:
            pair[1] = pair[1] / (1-rem)

    manager.add_items(items_list)
    manager.percent_rebalance('block')

