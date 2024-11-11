# http://developers.xstore.pro/documentation/

import logging

# from utils.xtb_api.api import XTB
from utils.xtb_api.xAPIConnector import (
    APIClient,
    APIStreamClient,
    procBalanceExample,
    procProfitExample,
    procTickExample,
    procTradeExample,
    procTradeStatusExample,
)

xtb_key_path = "./data/xtb.key"
logger = logging.getLogger("jsonSocket")
xtb_assets_filename = './data/xtb_assets.csv'

# xtb_api = XTB("./data/xtb.key")
# print(f'Server time: {xtb_api.get_ServerTime()}')
# print(f'Balance: {xtb_api.get_Balance()}')
# xtb_api.logout()

# df_assets = pd.read_csv(xtb_assets_filename)
# df_assets.columns = [x.upper() for x in df_assets.columns]
# df_assets.to_csv(xtb_assets_filename, index=False)
# print(df_assets[df_assets.CURRENCY == 'EUR'][['SYMBOL', 'TIME', 'PRECISION', 'GROUPNAME','DESCRIPTION']].to_string())

# create & connect to RR socket
client = APIClient()

# connect to RR socket, login
loginResponse = client.login(xtb_key_path)
logger.info(str(loginResponse))

# check if user logged in correctly
if not loginResponse['status']:
    print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))
    exit()

# assets = client.getAssets()
# df_assets = pd.DataFrame.from_dict(assets['returnData'])
# df_assets.to_csv(xtb_assets_filename, index=False)

# create & connect to Streaming socket with given ssID
# and functions for processing ticks, trades, profit and tradeStatus
sclient = APIStreamClient(
    ssId=loginResponse['streamSessionId'],
    tickFun=procTickExample,
    balanceFun=procBalanceExample,
    tradeFun=procTradeExample,
    profitFun=procProfitExample,
    tradeStatusFun=procTradeStatusExample,
)

sclient.subscribeBalance()

# gracefully close streaming socket
sclient.disconnect()

# gracefully close RR socket
client.disconnect()


# get ssId from login response
# ssid = loginResponse['streamSessionId']

# def main():
#     # enter your login credentials here
#     userId = 12345
#     password = "password"

#     # create & connect to RR socket
#     client = APIClient()

#     # connect to RR socket, login
#     loginResponse = client.execute(loginCommand(userId=userId, password=password))
#     logger.info(str(loginResponse))

#     # check if user logged in correctly
#     if loginResponse['status'] == False:
#         print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))
#         return

#     # get ssId from login response
#     ssid = loginResponse['streamSessionId']

#     # second method of invoking commands
#     resp = client.commandExecute('getAllSymbols')

#     # create & connect to Streaming socket with given ssID
#     # and functions for processing ticks, trades, profit and tradeStatus
#     sclient = APIStreamClient(
#         ssId=ssid,
#         tickFun=procTickExample,
#         tradeFun=procTradeExample,
#         profitFun=procProfitExample,
#         tradeStatusFun=procTradeStatusExample,
#     )

#     # subscribe for trades
#     sclient.subscribeTrades()

#     # subscribe for prices
#     sclient.subscribePrices(['EURUSD', 'EURGBP', 'EURJPY'])

#     # subscribe for profits
#     sclient.subscribeProfits()

#     # this is an example, make it run for 5 seconds
#     time.sleep(5)

#     # gracefully close streaming socket
#     sclient.disconnect()

#     # gracefully close RR socket
#     client.disconnect()
