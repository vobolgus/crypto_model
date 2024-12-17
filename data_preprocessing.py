import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_signals.log',
    filemode='a'
)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df):
    # Задаём точный формат дат для парсинга.
    # Если ваши даты выглядят так: 2024-02-03 00:00:00.000000+00:00
    # тогда используем следующий формат:
    date_format = '%Y-%m-%d %H:%M:%S'
    # liquidation_bar_df['open_timestamp'] = liquidation_bar_df['timestamp']

    for df_name, df in zip(['trade_bar_df', 'derivative_ticker_bar_df'],
                           [trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df]):
        if 'open_timestamp' in df.columns:
            df['open_timestamp'] = pd.to_datetime(df['open_timestamp'], format=date_format, errors='raise')
            # df['open_timestamp'] = df['open_timestamp'].astype('int64') // 10 ** 9
        else:
            logging.warning(f"В {df_name} отсутствует колонка 'open_timestamp'.")

    merge_keys = ['open_timestamp', 'symbol', 'exchange']
    combined = pd.merge(trade_bar_df, derivative_ticker_bar_df, on=merge_keys, how='outer', suffixes=('_tb', '_dt'))
    # combined = pd.merge(combined, liquidation_bar_df, on=merge_keys, how='outer', suffixes=('', '_liq'))
    logging.info('Данные объединены (outer join) по open_timestamp, symbol, exchange.')

    # Сортируем по open_timestamp для корректного временного порядка
    if 'open_timestamp' in combined.columns:
        combined.sort_values(by='open_timestamp')
        logging.info('Данные отсортированы по open_timestamp.')
    else:
        logging.warning('Отсутствует open_timestamp после объединения, сортировка невозможна.')

    # Заполняем пропущенные значения ближайшими известными по времени
    combined.ffill(inplace=True)

    for col in ['close_timestamp', 'timestamp']:
        if col in combined.columns:
            combined.drop(columns=[col], inplace=True)
            logging.info(f"Столбец {col} удалён из итоговых данных.")

    # Рассчитываем CVD
    if 'buy_volume' in combined.columns and 'sell_volume' in combined.columns:
        combined['delta_volume'] = combined['buy_volume'] - combined['sell_volume']
        combined['cvd'] = combined['delta_volume'].cumsum()
        logging.info('CVD рассчитан.')
    else:
        combined['cvd'] = np.nan
        logging.warning('Отсутствуют buy_volume или sell_volume для расчёта CVD.')

    # Изменение OI
    if 'open_interest' in combined.columns:
        combined['oi_change'] = combined['open_interest'].diff().fillna(0)
        logging.info('Изменение Open Interest рассчитано.')
    else:
        combined['oi_change'] = 0

    # Изменение Funding Rate
    if 'funding_rate' in combined.columns:
        combined['funding_rate_change'] = combined['funding_rate'].diff().fillna(0)
        logging.info('Изменение Funding Rates рассчитано.')
    else:
        combined['funding_rate_change'] = 0

    # Скользящие средние и RSI
    if 'close' in combined.columns:
        combined['ma_10'] = combined['close'].rolling(window=10, min_periods=1).mean()
        combined['ma_50'] = combined['close'].rolling(window=50, min_periods=1).mean()
        combined['rsi'] = calculate_rsi(combined['close'])
        logging.info('MA и RSI рассчитаны.')
    else:
        combined['ma_10'] = np.nan
        combined['ma_50'] = np.nan
        combined['rsi'] = np.nan

    return combined[combined['oi_change'] != 0]

if __name__ == "__main__":
    # Загрузка данных
    trade_bar_df = pd.read_csv('trade_bar_data.csv')
    derivative_ticker_bar_df = pd.read_csv('derivative_ticker_bar_data.csv')
    liquidation_bar_df = pd.read_csv('liquidation_bar_data.csv')

    data = preprocess_data(trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df)

    data.to_csv('preprocessed_data.csv', index=False)
    logging.info('Предобработанные данные сохранены в preprocessed_data.csv.')
    print(data.columns)