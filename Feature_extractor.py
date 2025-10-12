def add_basicfeatures(df):
    ret1 = df['Close'].pct_change()
    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    ma_diff = ma5 - ma20
    vol_20 = ret1.rolling(20).std()
    return ret1, ma5, ma20, ma_diff, vol_20


def add_ema(df):
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    ema_cross = ema_fast - ema_slow
    return ema_fast, ema_slow, ema_cross


def add_macd(df):
    ema_fast, ema_slow, _ = add_ema(df)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def add_ichimoku(df):
    period9_high = df['High'].rolling(window=9).max()
    period9_low = df['Low'].rolling(window=9).min()
    tenkan = (period9_high + period9_low) / 2

    period26_high = df['High'].rolling(window=26).max()
    period26_low = df['Low'].rolling(window=26).min()
    kijun = (period26_high + period26_low) / 2

    senkou_a = ((tenkan + kijun) / 2).shift(26)
    period52_high = df['High'].rolling(window=52).max()
    period52_low = df['Low'].rolling(window=52).min()
    senkou_b = ((period52_high + period52_low) / 2).shift(26)

    chikou = df['Close'].shift(-26)
    cloud_thickness = (senkou_a - senkou_b).abs()
    close_above_cloud = (df['Close'] > pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)).astype(int)
    tenkan_above_kijun = (tenkan > kijun).astype(int)

    return tenkan, kijun, senkou_a, senkou_b, chikou, cloud_thickness, close_above_cloud, tenkan_above_kijun


def add_ATR(df, period=14):
    H_L = df['High'] - df['Low']
    H_PC = (df['High'] - df['Close'].shift(1)).abs()
    L_PC = (df['Low'] - df['Close'].shift(1)).abs()
    TR = pd.concat([H_L, H_PC, L_PC], axis=1).max(axis=1)
    ATR = TR.rolling(window=period).mean()
    return ATR


def add_RSI(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    stochastic_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    stochastic_d = stochastic_k.rolling(window=d_period).mean()
    return stochastic_k, stochastic_d


def add_obv(df):
    delta = df['Close'].diff()
    direction = delta.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * df['Volume']).cumsum()
    return obv


def add_ROC(df, period=12):
    prev_close = df['Close'].shift(period)
    roc = ((df['Close'] - prev_close) / prev_close) * 100
    return roc


def add_price_action(df):
    candle_body = df['Close'] - df['Open']
    candle_range = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    body_to_range = candle_body.abs() / (candle_range + 1e-9)
    upper_to_lower_shadow = upper_shadow / (lower_shadow + 1e-9)
    direction = (df['Close'] > df['Open']).astype(int)

    return (candle_body, candle_range, upper_shadow, lower_shadow,
            body_to_range, upper_to_lower_shadow, direction)


def add_bollinger(df, period=20):
    ma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    bb_upper = ma + 2 * std
    bb_lower = ma - 2 * std
    bb_width = bb_upper - bb_lower
    bb_percent = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    return bb_upper, bb_lower, bb_width, bb_percent


def add_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mfi = 100 * (pos_sum / (pos_sum + neg_sum + 1e-9))
    return mfi


def add_williams_r(df, period=14):
    highest_high = df['High'].rolling(period).max()
    lowest_low = df['Low'].rolling(period).min()
    williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low + 1e-9)
    return williams_r


def add_pvt(df):
    pvt = ((df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)) * df['Volume']
    pvt = pvt.cumsum()
    return pvt


def add_rolling_stats(df, window=10):
    roll_mean = df['Close'].rolling(window).mean()
    roll_std = df['Close'].rolling(window).std()
    roll_skew = df['Close'].rolling(window).skew()
    roll_kurt = df['Close'].rolling(window).kurt()
    return roll_mean, roll_std, roll_skew, roll_kurt


def rolling_slope(series, window=5):
    slopes = [np.nan] * len(series)
    for i in range(window, len(series)):
        y = series[i - window:i]
        x = np.arange(window)
        slope = np.polyfit(x, y, 1)[0]
        slopes[i] = slope
    return pd.Series(slopes, index=series.index)


def add_slopes(df):
    slope_5 = rolling_slope(df['Close'], 5)
    slope_20 = rolling_slope(df['Close'], 20)
    return slope_5, slope_20


def add_feature_interactions(df):
    rsi_macd = add_RSI(df) * add_macd(df)[0]
    atr_vol_ratio = add_ATR(df) / (add_basicfeatures(df)[4] + 1e-9)
    ema_diff_ratio = add_ema(df)[0] / (add_ema(df)[1] + 1e-9)
    return rsi_macd, atr_vol_ratio, ema_diff_ratio


def feature_extraction(df):
    ret1, ma5, ma20, ma_diff, vol_20 = add_basicfeatures(df)
    ema_fast, ema_slow, ema_cross = add_ema(df)
    macd, macd_signal, macd_hist = add_macd(df)
    tenkan, kijun, senkou_a, senkou_b, chikou, cloud_thick, close_above_cloud, tenkan_above_kijun = add_ichimoku(df)
    atr = add_ATR(df)
    rsi = add_RSI(df)
    stochastic_k, stochastic_d = add_stochastic(df)
    obv = add_obv(df)
    roc = add_ROC(df)
    candle_body, candle_range, upper_shadow, lower_shadow, body_to_range, upper_to_lower_shadow, direction = add_price_action(
        df)
    bb_upper, bb_lower, bb_width, bb_percent = add_bollinger(df)
    mfi = add_mfi(df)
    williams_r = add_williams_r(df)
    pvt = add_pvt(df)
    roll_mean, roll_std, roll_skew, roll_kurt = add_rolling_stats(df)
    slope_5, slope_20 = add_slopes(df)
    rsi_macd, atr_vol_ratio, ema_diff_ratio = add_feature_interactions(df)

    df = df.copy()
    df['ret1'] = ret1
    df['ma_5'] = ma5
    df['ma_20'] = ma20
    df['ma_diff'] = ma_diff
    df['vol_20'] = vol_20
    df['ema_fast'] = ema_fast
    df['ema_slow'] = ema_slow
    df['ema_cross'] = ema_cross
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    df['tenkan'] = tenkan
    df['kijun'] = kijun
    df['senkou_a'] = senkou_a
    df['senkou_b'] = senkou_b
    df['chikou'] = chikou
    df['cloud_thick'] = cloud_thick
    df['close_above_cloud'] = close_above_cloud
    df['tenkan_above_kijun'] = tenkan_above_kijun
    df['ATR'] = atr
    df['rsi'] = rsi
    df['stochastic_k'] = stochastic_k
    df['stochastic_d'] = stochastic_d
    df['obv'] = obv
    df['roc'] = roc
    df["candle_body"] = candle_body
    df["candle_range"] = candle_range
    df["upper_shadow"] = upper_shadow
    df["lower_shadow"] = lower_shadow
    df["body_to_range"] = body_to_range
    df["upper_to_lower_shadow"] = upper_to_lower_shadow
    df["direction"] = direction
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width
    df["bb_percent"] = bb_percent
    df["mfi"] = mfi
    df["williams_r"] = williams_r
    df["pvt"] = pvt
    df["roll_mean"] = roll_mean
    df["roll_std"] = roll_std
    df["roll_skew"] = roll_skew
    df["roll_kurt"] = roll_kurt
    df["slope_5"] = slope_5
    df["slope_20"] = slope_20
    df["rsi_macd"] = rsi_macd
    df["atr_vol_ratio"] = atr_vol_ratio
    df["ema_diff_ratio"] = ema_diff_ratio

    return df