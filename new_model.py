import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import logging
import os
from itertools import product
from tqdm import tqdm

# 1. Настройка Логирования
def setup_logging(log_file='trading_strategy.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info('Логирование настроено.')


# 2. Загрузка и Предобработка Данных
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'])
        data.set_index('timestamp', inplace=True)
        logging.info(f'Данные успешно загружены из {file_path}.')
        return data
    except FileNotFoundError:
        logging.error(f'Файл {file_path} не найден.')
        raise
    except Exception as e:
        logging.error(f'Ошибка при загрузке данных: {e}')
        raise


def preprocess_data(data):
    try:
        # Проверка на пропущенные значения
        if data.isnull().values.any():
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            logging.info('Пропущенные значения заполнены методом forward/backward fill.')
        else:
            logging.info('Пропущенные значения в данных не обнаружены.')

        # Дополнительная предобработка при необходимости
        # Например, проверка типов данных, преобразование валют и т.д.

        return data
    except Exception as e:
        logging.error(f'Ошибка при предобработке данных: {e}')
        raise


# 3. Вычисление Индикаторов
def calculate_indicators(data):
    try:
        # Проверка наличия необходимых столбцов
        required_columns = ['buy_volume', 'sell_volume', 'oi', 'volume_usdt_liq', 'fr', 'high', 'low',
                            'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logging.error(f'Отсутствует необходимый столбец: {col}')
                raise KeyError(f'Отсутствует необходимый столбец: {col}')

        # Cumulative Volume Delta (CVD)
        data['CVD'] = (data['buy_volume'] - data['sell_volume']).cumsum()
        logging.info('Cumulative Volume Delta (CVD) рассчитан.')

        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
        logging.info('Индекс относительной силы (RSI) рассчитан.')

        # Скользящие Средние (SMA и EMA)
        data['SMA_20'] = ta.trend.SMAIndicator(close=data['close'], window=20).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()
        logging.info('Скользящие средние (SMA и EMA) рассчитаны.')

        # ADX
        data['ADX'] = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=14).adx()
        logging.info('Индекс силы тренда (ADX) рассчитан.')

        # MACD
        macd = ta.trend.MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        logging.info('MACD рассчитан.')

        # Полосы Боллинджера
        bollinger = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
        data['BB_upper'] = bollinger.bollinger_hband()
        data['BB_middle'] = bollinger.bollinger_mavg()
        data['BB_lower'] = bollinger.bollinger_lband()
        logging.info('Полосы Боллинджера рассчитаны.')

        # Стохастический Осциллятор
        stoch = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14,
                                                 smooth_window=3)
        data['Stoch_%K'] = stoch.stoch()
        data['Stoch_%D'] = stoch.stoch_signal()
        logging.info('Стохастический осциллятор рассчитан.')

        return data
    except Exception as e:
        logging.error(f'Ошибка при расчете индикаторов: {e}')
        raise


def generate_signals(data, weights):
    try:
        # Инициализация сигналов для каждого индикатора
        data['Signal_RSI'] = 0
        data['Signal_MACD'] = 0
        data['Signal_Stochastic'] = 0
        data['Signal_OI'] = 0
        data['Signal_Liquidations'] = 0

        # Сигналы RSI
        data['Signal_RSI'] = np.where(
            data['RSI'] < 30, 1,
            np.where(data['RSI'] > 70, -1, 0)
        )

        # Сигналы MACD
        data['Signal_MACD'] = np.where(
            (data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1)), 1,
            np.where(
                (data['MACD'] < data['MACD_signal']) & (data['MACD'].shift(1) >= data['MACD_signal'].shift(1)), -1, 0
            )
        )

        # Сигналы Стохастика
        data['Signal_Stochastic'] = np.where(
            (data['Stoch_%K'] > data['Stoch_%D']) &
            (data['Stoch_%K'].shift(1) <= data['Stoch_%D'].shift(1)) &
            (data['Stoch_%K'] < 20), 1,
            np.where(
                (data['Stoch_%K'] < data['Stoch_%D']) &
                (data['Stoch_%K'].shift(1) >= data['Stoch_%D'].shift(1)) &
                (data['Stoch_%K'] > 80), -1, 0
            )
        )

        # Сигналы Open Interest (OI)
        oi_threshold = data['oi'].quantile(0.95)
        data['Signal_OI'] = np.where(data['oi'] > oi_threshold, 1, 0)

        # Сигналы Ликвидаций
        liquidation_threshold = data['volume_usdt_liq'].quantile(0.95)
        data['Signal_Liquidations'] = np.where(data['volume_usdt_liq'] > liquidation_threshold, 1, 0)

        # Фильтр по ADX (только при сильном тренде)
        data['Signal_RSI'] = np.where(data['ADX'] > 25, data['Signal_RSI'], 0)
        data['Signal_MACD'] = np.where(data['ADX'] > 25, data['Signal_MACD'], 0)
        data['Signal_Stochastic'] = np.where(data['ADX'] > 25, data['Signal_Stochastic'], 0)
        # OI и Liquidations можно оставить без фильтра по ADX, если это соответствует вашей стратегии

        # Создание списка индикаторов с их весами
        indicators = [
            ('Signal_RSI', 'RSI', weights['RSI']),
            ('Signal_MACD', 'MACD', weights['MACD']),
            ('Signal_Stochastic', 'Stochastic', weights['Stochastic']),
            ('Signal_OI', 'OI', weights['OI']),
            ('Signal_Liquidations', 'Liquidations', weights['Liquidations'])
        ]

        # Сортировка индикаторов по весам (приоритетам) в порядке убывания
        indicators.sort(key=lambda x: x[2], reverse=True)

        # Инициализация столбцов для финального сигнала и причины
        data['Signal'] = 0
        data['Signal_Reason'] = ''

        # Функция для определения финального сигнала и причины
        def determine_final_signal(row):
            for col_name, reason_name, weight in indicators:
                indicator_signal = row[col_name]
                if indicator_signal != 0:
                    return pd.Series({'Signal': indicator_signal, 'Signal_Reason': reason_name})
            return pd.Series({'Signal': 0, 'Signal_Reason': ''})

        # Применение функции к каждой строке данных
        data[['Signal', 'Signal_Reason']] = data.apply(determine_final_signal, axis=1)

        logging.info('Торговые сигналы успешно сгенерированы.')
        return data
    except Exception as e:
        logging.error(f'Ошибка при генерации сигналов: {e}')
        raise


# 5. Бэктестинг Стратегии
def backtest_strategy(data):
    try:
        # Создание позиции на основе сигнала
        data['Position'] = data['Signal'].replace(0, np.nan).ffill().fillna(0)

        # Вычисление ежедневной доходности
        data['Market_Return'] = data['close'].pct_change()
        data['Strategy_Return'] = data['Market_Return'] * data['Position'].shift(1)

        # Кумулятивная доходность
        data['Cumulative_Market_Return'] = (1 + data['Market_Return']).cumprod()
        data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

        logging.info('Бэктестинг выполнен.')
        return data
    except Exception as e:
        logging.error(f'Ошибка при бэктестинге: {e}')
        raise


# 6. Визуализация Результатов
def plot_results(data, weights, metric='Cumulative_Strategy_Return'):
    try:
        plt.figure(figsize=(14,7))
        plt.plot(data.index, data['Cumulative_Market_Return'], label='Рынок', color='blue')
        plt.plot(data.index, data['Cumulative_Strategy_Return'], label='Стратегия', color='orange')
        plt.title(f'Сравнение Кумулятивной Доходности Стратегии и Рынка\nВесовые Коэффициенты: {weights}')
        plt.xlabel('Дата')
        plt.ylabel('Кумулятивная Доходность')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_returns.png')
        plt.show()
        logging.info('Результаты успешно визуализированы и сохранены в "cumulative_returns.png".')
    except Exception as e:
        logging.error(f'Ошибка при визуализации результатов: {e}')
        raise

# 7. Вывод Результатов
def print_results(data):
    try:
        total_return_market = data['Cumulative_Market_Return'].iloc[-1] - 1
        total_return_strategy = data['Cumulative_Strategy_Return'].iloc[-1] - 1

        print(f"Доходность рынка: {total_return_market:.2%}")
        print(f"Доходность стратегии: {total_return_strategy:.2%}")

        logging.info(f"Доходность рынка: {total_return_market:.2%}")
        logging.info(f"Доходность стратегии: {total_return_strategy:.2%}")
    except Exception as e:
        logging.error(f'Ошибка при выводе результатов: {e}')
        raise


# 8. Функция Оптимизации (Grid Search)
def optimize_parameters(data, param_grid, metric='Cumulative_Strategy_Return'):
    """
    Проводит Grid Search для подбора оптимальных параметров стратегии.

    :param data: DataFrame с данными и рассчитанными индикаторами.
    :param param_grid: Словарь с параметрами и их значениями для перебора.
    :param metric: Метрика для оптимизации. По умолчанию 'Cumulative_Strategy_Return'.
    :return: Наилучшие параметры и соответствующая метрика.
    """
    best_params = None
    best_metric = -np.inf
    results = []

    # Генерация всех комбинаций параметров
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    logging.info(f'Начинается оптимизация по {len(param_combinations)} комбинациям параметров.')

    for params in tqdm(param_combinations, desc="Optimizing", unit="combo"):
        try:
            # Генерация сигналов с текущими параметрами
            temp_data = generate_signals(data.copy(), params)

            # Бэктестинг стратегии
            temp_data = backtest_strategy(temp_data)

            # Вычисление метрики
            current_metric = temp_data[metric].iloc[-1]

            # Сохранение результатов
            results.append((params, current_metric))

            # Обновление наилучших параметров
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = params
        except Exception as e:
            logging.error(f'Ошибка при оптимизации с параметрами {params}: {e}')
            continue

    # Сохранение всех результатов
    results_df = pd.DataFrame(results, columns=['Parameters', 'Metric'])
    results_df.to_csv('optimization_results.csv', index=False)
    logging.info('Результаты оптимизации сохранены в "optimization_results.csv".')

    logging.info(f'Оптимизация завершена. Лучшая метрика: {best_metric:.4f} с параметрами {best_params}')

    return best_params, best_metric

# 9. Основная Функция
def main():
    # Настройка логирования
    setup_logging()

    # Путь к данным
    data_file = 'preprocessed_data.csv'  # Замените на путь к вашему файлу данных

    # Проверка существования файла данных
    if not os.path.exists(data_file):
        logging.error(f'Файл данных {data_file} не существует.')
        return

    # Загрузка данных
    data = load_data(data_file)

    # Предобработка данных
    data = preprocess_data(data)

    # Вычисление индикаторов
    data = calculate_indicators(data)

    # Определение сетки параметров для оптимизации
    param_grid = {
        'RSI': [1],
        'MACD': [0.75],
        'Stochastic': [1],
        'OI': [1],
        'Liquidations': [1]
    }

    # Проведение оптимизации
    best_params, best_metric = optimize_parameters(data, param_grid, metric='Cumulative_Strategy_Return')

    # Генерация сигналов с наилучшими параметрами
    data = generate_signals(data, best_params)

    # Бэктестинг стратегии с наилучшими параметрами
    data = backtest_strategy(data)

    # Визуализация результатов
    plot_results(data, best_params)

    # Вывод результатов
    print_results(data)

    # Сохранение данных с сигналами и доходностями
    try:
        data.to_csv('strategy_backtest_results.csv')
        logging.info('Результаты бэктеста сохранены в "strategy_backtest_results.csv".')
    except Exception as e:
        logging.error(f'Ошибка при сохранении результатов бэктеста: {e}')
        raise



if __name__ == "__main__":
    main()