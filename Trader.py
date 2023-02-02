from binance.client import Client

# Replace with your own API keys
api_key = 'API_KEY'
api_secret = 'API_SECRET'

# Initialize the Binance client
client = Client(api_key, api_secret)

# Define the symbol you want to trade
symbol = "BTCUSDT"

# Define the quantity of fake Bitcoin to trade
quantity = 1

# Buy fake Bitcoin
order = client.order_market_buy(
    symbol=symbol,
    quantity=quantity
)

# Confirm the order was executed successfully
print("Bought {} {}".format(quantity, symbol))

# Sell the fake Bitcoin
order = client.order_market_sell(
    symbol=symbol,
    quantity=quantity
)

# Confirm the order was executed successfully
print("Sold {} {}".format(quantity, symbol))