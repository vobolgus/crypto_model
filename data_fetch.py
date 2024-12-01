import os
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_signals.log',
    filemode='w'
)

# Загрузка переменных окружения из .env файла
load_dotenv()

def fetch_data(timeframe_days=1):
    """
    Подключается к базе данных PostgreSQL, выполняет SQL-запросы и возвращает данные в виде pandas DataFrame.
    Получает данные за последние timeframe_hours часов.
    """
    try:
        # Получение пароля из переменной окружения
        db_password = 'fCExl7mD6mBbKJCU332eRf5YX8cIzYja'
        if not db_password:
            raise ValueError("Пароль к базе данных не установлен в переменной окружения PGPASSWORD.")

        # Создание строки подключения для SQLAlchemy
        engine = create_engine(f'postgresql+psycopg2://postgres:{db_password}@data.oshu.io:5432/oshu')

        # Определение временного диапазона
        end_time = datetime.utcnow() - timedelta(days=90)
        start_time = end_time - timedelta(days=90)

        # Форматирование временных меток в формат ISO
        start_timestamp = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        end_timestamp = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')

        # SQL-запросы для разных типов данных
        queries = {
            'trade_bar': """
                SELECT *
                FROM tb.tb_30s_btc
                WHERE "exchange" = %s
                  AND "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" ASC NULLS LAST;
            """,
            'derivative_ticker_bar': """
                SELECT *
                FROM dt.dt_30s_btc
                WHERE "exchange" = %s
                  AND "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" ASC NULLS LAST;
            """,
            'liquidation_bar': """
                SELECT *
                FROM liq.liq_30s_btc
                WHERE "exchange" = %s
                  AND "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" ASC NULLS LAST;
            """
        }

        # Параметры запроса
        params = ('binance-futures', start_timestamp, end_timestamp)

        # Получение данных
        trade_bar_df = pd.read_sql_query(queries['trade_bar'], engine, params=params)
        logging.info('Данные TradeBar успешно загружены.')

        derivative_ticker_bar_df = pd.read_sql_query(queries['derivative_ticker_bar'], engine, params=params)
        logging.info('Данные DerivativeTickerBar успешно загружены.')

        liquidation_bar_df = pd.read_sql_query(queries['liquidation_bar'], engine, params=params)
        logging.info('Данные LiquidationBar успешно загружены.')

        return trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df

    except Exception as error:
        logging.error("Ошибка при подключении к PostgreSQL или выполнении запроса: %s", error)
        return None, None, None
    finally:
        if 'engine' in locals() and engine:
            engine.dispose()
            logging.info("Соединение с PostgreSQL закрыто.")

if __name__ == "__main__":
    # Получение данных за последние 24 часа
    trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df = fetch_data(timeframe_days=30)

    if trade_bar_df is not None and not trade_bar_df.empty:
        logging.info("Данные успешно получены и готовы к дальнейшей обработке.")
    else:
        logging.error("Не удалось получить данные из базы данных или данные пусты.")
        exit()

    # Сохранение данных в CSV-файлы (опционально)
    trade_bar_df.to_csv('trade_bar_data.csv', index=False)
    derivative_ticker_bar_df.to_csv('derivative_ticker_bar_data.csv', index=False)
    liquidation_bar_df.to_csv('liquidation_bar_data.csv', index=False)
    logging.info("Данные сохранены в CSV-файлы.")