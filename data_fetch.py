import os
from dotenv import load_dotenv
import clickhouse_connect
import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_signals.log',
    filemode='w'
)

load_dotenv()


def fetch_trade_bars(symbol='btc', exchange='binance', timeframe='5min', start_date='2024-11-01',
                     end_date='2024-12-08'):
    try:
        host = os.getenv('CLICKHOUSE_HOST', '136.243.21.210')
        # Предположим, что сервер доступен по HTTP на порту 8123 (стандартный)
        port = int(os.getenv('CLICKHOUSE_PORT', '8123'))
        user = os.getenv('CLICKHOUSE_USER', 'oshu')
        password = os.getenv('CLICKHOUSE_PASSWORD', 'oshu')

        # Подключение без TLS
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=user,
            password=password,
            secure=False,
            verify=False
        )

        query = f"""
        SELECT
            symbol,
            exchange,
            open_timestamp,
            countMerge(trades) as trades,
            argMinMerge(open) as open,
            high, low,
            argMaxMerge(close) as close,
            volume, sell_volume, buy_volume,
            volume_usdt, buy_volume_usdt, sell_volume_usdt
        FROM trades_aggregated
        WHERE
            tf = '{timeframe}'
            AND symbol = '{symbol}'
            AND exchange = '{exchange}'
            AND open_timestamp BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY symbol, exchange, high, low, volume, sell_volume, buy_volume, volume_usdt, buy_volume_usdt, sell_volume_usdt, open_timestamp
        ORDER BY open_timestamp
        """

        result = client.query(query)
        df = pd.DataFrame(result.result_set, columns=result.column_names)
        logging.info(f"Успешно получены трейд бары: {len(df)} записей.")
        return df

    except Exception as e:
        logging.error("Ошибка при получении трейд баров из ClickHouse: %s", e)
        return pd.DataFrame()


def fetch_derivative_bars(symbol='btc', exchange='binance-futures', timeframe='5min', start_date='2024-02-02',
                          end_date='2024-02-03'):
    try:
        host = os.getenv('CLICKHOUSE_HOST', '136.243.21.210')
        port = int(os.getenv('CLICKHOUSE_PORT', '8123'))
        user = os.getenv('CLICKHOUSE_USER', 'oshu')
        password = os.getenv('CLICKHOUSE_PASSWORD', 'oshu')

        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=user,
            password=password,
            secure=False,
            verify=False
        )

        # Используем argMaxMerge для финализации данных AggregateFunction(argMax, ...)
        # Выбираем только те колонки, которые реально существуют в таблице
        query = f"""
        SELECT
            symbol,
            exchange,
            open_timestamp,
            argMaxMerge(last_price) as last_price,
            argMaxMerge(open_interest) as open_interest,
            argMaxMerge(funding_rate) as funding_rate,
            argMaxMerge(index_price) as index_price,
            argMaxMerge(mark_price) as mark_price
        FROM derivative_tickers_aggregated
        WHERE
            tf = '{timeframe}'
            AND symbol = '{symbol}'
            AND exchange = '{exchange}'
            AND open_timestamp BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY symbol, exchange, open_timestamp
        ORDER BY open_timestamp
        """

        result = client.query(query)
        df = pd.DataFrame(result.result_set, columns=result.column_names)
        logging.info(f"Успешно получены дериватив бары: {len(df)} записей.")
        return df

    except Exception as e:
        logging.error("Ошибка при получении дериватив баров из ClickHouse: %s", e)
        return pd.DataFrame()


def fetch_liquidation_bars(symbol='btc', exchange='binance-futures', timeframe='5min', start_date='2024-02-02',
                           end_date='2024-02-03'):
    try:
        host = os.getenv('CLICKHOUSE_HOST', '136.243.21.210')
        port = int(os.getenv('CLICKHOUSE_PORT', '8123'))
        user = os.getenv('CLICKHOUSE_USER', 'oshu')
        password = os.getenv('CLICKHOUSE_PASSWORD', 'oshu')

        client = clickhouse_connect.get_client(host=host, port=port, username=user, password=password, secure=False,
                                               verify=False)

        query = f"""
        SELECT
            symbol,
            exchange,
            open_timestamp,
            any(liqs) as liqs,
            any(volume) as volume,
            any(sell_volume) as sell_volume,
            any(buy_volume) as buy_volume,
            any(volume_usdt) as volume_usdt,
            any(buy_volume_usdt) as buy_volume_usdt,
            any(sell_volume_usdt) as sell_volume_usdt
        FROM liquidations_aggregated
        WHERE
            tf = '{timeframe}'
            AND symbol = '{symbol}'
            AND exchange = '{exchange}'
            AND open_timestamp BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY symbol, exchange, open_timestamp
        ORDER BY open_timestamp
        """

        result = client.query(query)
        df = pd.DataFrame(result.result_set, columns=result.column_names)
        logging.info(f"Успешно получены ликвидационные бары: {len(df)} записей.")
        return df

    except Exception as e:
        logging.error("Ошибка при получении ликвидационных баров из ClickHouse: %s", e)
        return pd.DataFrame()

def fetch_data(symbol='btc', exchange='binance-futures', timeframe='5min', start_date='2024-02-02',
                           end_date='2024-02-03'):
    trade_bar_df = fetch_trade_bars(symbol=symbol, exchange=exchange, timeframe=timeframe, start_date=start_date,
                                    end_date=end_date)
    derivative_bar_df = fetch_derivative_bars(symbol=symbol, exchange=exchange, timeframe=timeframe,
                                              start_date=start_date, end_date=end_date)
    liquidation_bar_df = fetch_liquidation_bars(symbol=symbol, exchange=exchange, timeframe=timeframe,
                                                start_date=start_date, end_date=end_date)

    if not trade_bar_df.empty:
        trade_bar_df.to_csv('trade_bar_data.csv', index=False)
    if not derivative_bar_df.empty:
        derivative_bar_df.to_csv('derivative_ticker_bar_data.csv', index=False)
    if not liquidation_bar_df.empty:
        liquidation_bar_df.to_csv('liquidation_bar_data.csv', index=False)

    return


if __name__ == "__main__":
    trade_bar_df = fetch_trade_bars(symbol='eth', exchange='binance-futures', timeframe='4h', start_date='2024-11-01',
                                    end_date='2024-12-01')
    derivative_bar_df = fetch_derivative_bars(symbol='btc', exchange='binance-futures', timeframe='5min',
                                              start_date='2024-11-24', end_date='2024-12-01')
    liquidation_bar_df = fetch_liquidation_bars(symbol='btc', exchange='binance-futures', timeframe='5min',
                                                start_date='2024-11-24', end_date='2024-12-01')

    if not trade_bar_df.empty:
        trade_bar_df.to_csv('trade_bar_data.csv', index=False)
    if not derivative_bar_df.empty:
        derivative_bar_df.to_csv('derivative_ticker_bar_data.csv', index=False)
    if not liquidation_bar_df.empty:
        liquidation_bar_df.to_csv('liquidation_bar_data.csv', index=False)

    logging.info("Данные успешно извлечены и сохранены.")