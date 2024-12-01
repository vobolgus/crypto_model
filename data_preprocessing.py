import pandas as pd
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_signals.log',
    filemode='a'  # Добавлять логи к существующему файлу
)

def fill_missing_values(df):
    """
    Функция для заполнения пропущенных значений методом forward fill и backward fill.
    """
    missing_values = df.isnull().sum()
    if missing_values.any():
        logging.warning(f'Пропущенные значения в данных:\n{missing_values}')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        logging.info('Пропущенные значения заполнены методом forward/backward fill.')
    else:
        logging.info('Пропущенных значений в данных не обнаружено.')
    return df

def calculate_rsi(series, period=14):
    """
    Функция для вычисления RSI.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df):
    """
    Предобработка данных: преобразование времени, заполнение пропущенных значений, объединение DataFrame,
    вычисление индикаторов.
    """
    # Преобразование столбца 'timestamp' в datetime
    trade_bar_df['timestamp'] = pd.to_datetime(trade_bar_df['timestamp'])
    derivative_ticker_bar_df['timestamp'] = pd.to_datetime(derivative_ticker_bar_df['timestamp'])
    liquidation_bar_df['timestamp'] = pd.to_datetime(liquidation_bar_df['timestamp'])
    logging.info('Столбцы timestamp преобразованы в datetime.')

    # Заполнение пропущенных значений
    trade_bar_df = fill_missing_values(trade_bar_df)
    derivative_ticker_bar_df = fill_missing_values(derivative_ticker_bar_df)
    liquidation_bar_df = fill_missing_values(liquidation_bar_df)

    # Объединение данных по столбцу 'timestamp', 'symbol', 'exchange'
    data = trade_bar_df.merge(
        derivative_ticker_bar_df,
        on=['timestamp', 'symbol', 'exchange'],
        how='left',
        suffixes=('_tb', '_dt')
    )
    data = data.merge(
        liquidation_bar_df,
        on=['timestamp', 'symbol', 'exchange'],
        how='left',
        suffixes=('', '_liq')  # оставшиеся столбцы из liquidation_bar_df будут без суффикса
    )
    logging.info('Данные объединены по столбцам timestamp, symbol, exchange.')

    # Заполнение пропущенных значений после объединения
    data = fill_missing_values(data)

    # Вычисление CVD
    data['delta_volume'] = data['buy_volume'] - data['sell_volume']
    data['cvd'] = data['delta_volume'].cumsum()
    logging.info('CVD рассчитан.')

    # Вычисление изменения Open Interest (OI)
    if 'oi' in data.columns:
        data['oi_change'] = data['oi'].diff().fillna(0)
    else:
        data['oi_change'] = 0
    logging.info('Изменение Open Interest рассчитано.')

    # Вычисление Ликвидаций
    data['total_liquidations'] = data['buy_liqs'] + data['sell_liqs']
    data['liquidation_volume'] = data['buy_volume_liq'] + data['sell_volume_liq']
    logging.info('Ликвидации рассчитаны.')

    # Вычисление изменения Funding Rates
    if 'fr' in data.columns:
        data['funding_rate_change'] = data['fr'].diff().fillna(0)
    else:
        data['funding_rate_change'] = 0
    logging.info('Изменение Funding Rates рассчитано.')

    # Дополнительные Индикаторы
    data['ma_10'] = data['close'].rolling(window=10).mean()
    data['ma_50'] = data['close'].rolling(window=50).mean()
    data['rsi'] = calculate_rsi(data['close'])
    logging.info('Дополнительные индикаторы (MA и RSI) рассчитаны.')

    # Переименование столбцов для удобства
    data.rename(columns={
        'close': 'close',
        'volume': 'volume',
        'buy_volume': 'buy_volume',
        'sell_volume': 'sell_volume',
        'vwap': 'vwap',
    }, inplace=True)

    return data

if __name__ == "__main__":
    # Загрузка данных из CSV-файлов
    trade_bar_df = pd.read_csv('trade_bar_data.csv')
    derivative_ticker_bar_df = pd.read_csv('derivative_ticker_bar_data.csv')
    liquidation_bar_df = pd.read_csv('liquidation_bar_data.csv')

    # Предобработка данных
    data = preprocess_data(trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df)

    # Сохранение предобработанных данных (опционально)
    data.to_csv('preprocessed_data.csv', index=False)
    logging.info('Предобработанные данные сохранены в файл preprocessed_data.csv.')