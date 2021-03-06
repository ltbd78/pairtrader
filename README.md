# Finding Pair
```
for each pair:
  fit Y~X obtain residuals
  do ADF test on residuals
  if p-value < 0.05: can trade pair
```

# Fitting Model
```
fit Y~X obtain b1
get spread = Y - b1*X  (spread is same as residuals)
get rolling moving average of spread (ideally window=60)
get rolling standard deviation of spread (ideally window=60)
get z = (spread - rolling_ma)/rolling_sd
```

# Trading Pair
```
for t in timeframe:
  if no open positions:
    if z < -z_crit : long spread (buy Y, sell b1 * X)
    if z > z_crit : short spread (sell Y, buy b1 * X)
  else:
    if long : close position if z outside -z_crit + (-z_stoploss, z_takeprofit)
    if short : close position if z outside z_crit + (-z_takeprofit, z_stoploss)
```