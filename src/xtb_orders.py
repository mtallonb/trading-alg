from xtbapi.api import *

# http://developers.xstore.pro/documentation/

xtb_api = XTB("../data/xtb.key")
print(f'Balance: {xtb_api.get_Balance()}')
xtb_api.logout()