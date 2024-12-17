import traceback

import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import logging
import os
from itertools import product
from tqdm import tqdm
from data_fetch import fetch_data
from data_preprocessing import preprocess_data


# 1. Настройка Логирования
def setup_logging(log_file='trading_strategy.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    logging.info('Логирование настроено.')


# 2. Загрузка и Предобработка Данных
def load_data(symbol, timeframe):
    try:
        # Здесь мы используем fetch_data для загрузки данных.
        # Обратите внимание: параметры start_date и end_date можно при необходимости сделать параметризуемыми.
        fetch_data(
            symbol=symbol,
            exchange='binance-futures',
            timeframe=timeframe,
            start_date='2024-11-24',
            end_date='2024-12-01'
        )
        trade_bar_df = pd.read_csv('trade_bar_data.csv')
        derivative_ticker_bar_df = pd.read_csv('derivative_ticker_bar_data.csv')
        liquidation_bar_df = pd.read_csv('liquidation_bar_data.csv')
        data = preprocess_data(trade_bar_df, derivative_ticker_bar_df, liquidation_bar_df)
        logging.info(f'Данные успешно загружены для таймфрейма {timeframe}.')
        return data
    except FileNotFoundError:
        logging.error(f'Файлы с данными не найдены для таймфрейма {timeframe}.')
        raise
    except Exception as e:
        logging.error(f'Ошибка при загрузке данных для таймфрейма {timeframe}: {e}')
        raise


# 3. Вычисление Индикаторов
def calculate_indicators(data):
    try:
        required_columns = ['buy_volume', 'sell_volume', 'open_interest', 'high', 'low',
                            'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logging.error(f'Отсутствует необходимый столбец: {col}')
                raise KeyError(f'Отсутствует необходимый столбец: {col}')

        data['CVD'] = (data['buy_volume'] - data['sell_volume']).cumsum()
        data['RSI'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
        data['SMA_20'] = ta.trend.SMAIndicator(close=data['close'], window=20).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()
        data['ADX'] = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=14).adx()

        macd = ta.trend.MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()

        bollinger = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
        data['BB_upper'] = bollinger.bollinger_hband()
        data['BB_middle'] = bollinger.bollinger_mavg()
        data['BB_lower'] = bollinger.bollinger_lband()

        stoch = ta.momentum.StochasticOscillator(high=data['high'], low=data['low'], close=data['close'], window=14,
                                                 smooth_window=3)
        data['Stoch_%K'] = stoch.stoch()
        data['Stoch_%D'] = stoch.stoch_signal()

        logging.info('Индикаторы рассчитаны.')
        return data
    except Exception as e:
        logging.error(f'Ошибка при расчете индикаторов: {e}')
        raise

def compute_indicator_signals(data):
    try:
        data['Signal_RSI'] = 0
        data['Signal_MACD'] = 0
        data['Signal_Stochastic'] = 0
        data['Signal_OI'] = 0

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
            (data['Stoch_%K'] < 40), 1,
            np.where(
                (data['Stoch_%K'] < data['Stoch_%D']) &
                (data['Stoch_%K'].shift(1) >= data['Stoch_%D'].shift(1)) &
                (data['Stoch_%K'] > 60), -1, 0
            )
        )

        # Сигналы Open Interest (OI)
        oi_threshold = data['open_interest'].quantile(0.95)
        data['Signal_OI'] = np.where(data['open_interest'] > oi_threshold, 1, 0)

        # Фильтр по ADX
        data['Signal_RSI'] = np.where(data['ADX'] > 25, data['Signal_RSI'], 0)
        data['Signal_MACD'] = np.where(data['ADX'] > 25, data['Signal_MACD'], 0)
        data['Signal_Stochastic'] = np.where(data['ADX'] > 25, data['Signal_Stochastic'], 0)


        logging.info('Сигналы индикаторов успешно рассчитаны.')
        return data
    except Exception as e:
        logging.error(f'Ошибка при расчете сигналов индикаторов: {e}')
        raise


def compute_advanced_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    На основании индикаторов (CVD, Funding, OI, RSI, Stoch, ADX) формируем торговые сигналы.
    Логика:
    - Используем сигнал дивергенции CVD как первичный триггер.
    - Если Signal_CVD == 1 (бычий), проверяем funding_rate, OI и условия фильтра:
        * Funding должен быть отрицательным или близким к нулю, если цена не растет,
          что может говорить о загнанных в угол шортах.
        * ADX > 25 для уверенности в наличии тренда или возможности разворота
        * RSI < 50 или выходит из перепроданности
        * Stoch %K пересекает %D снизу в зоне <40
    - Если Signal_CVD == -1 (медвежий), проверяем обратные условия:
        * Funding позитивный или выше среднего, при этом цена не растет,
          что говорит о страданиях быков.
        * ADX > 25
        * RSI > 50
        * Stoch %K пересекает %D сверху в зоне >60

    Возвращает колонку 'signal':
    'buy'  = сигнал на покупку (лонг)
    'sell' = сигнал на продажу (шорт)
    'hold' = нет сигнала
    """
    # Инициализируем колонку сигналов
    data['Trade_Signal'] = 0

    try:
        # Начинаем с i=1, чтобы иметь доступ к i-1 строке
        for i in range(1, len(data)):
            close = data['close'].iloc[i]
            prev_close = data['close'].iloc[i-1]
            funding_rate = data['funding_rate'].iloc[i]
            oi = data['open_interest'].iloc[i]
            prev_oi = data['open_interest'].iloc[i-1]
            oi_change = data['oi_change'].iloc[i]
            cvd = data['cvd'].iloc[i]
            prev_cvd = data['cvd'].iloc[i-1]
            fr_change = data['funding_rate_change'].iloc[i]
            ma_10 = data['ma_10'].iloc[i]
            ma_50 = data['ma_50'].iloc[i]
            rsi = data['rsi'].iloc[i]

            logging.debug(
                f"Текущая строка {i}: close={close}, prev_close={prev_close}, "
                f"funding_rate={funding_rate}, oi={oi}, prev_oi={prev_oi}, "
                f"oi_change={oi_change}, cvd={cvd}, prev_cvd={prev_cvd}, "
                f"fr_change={fr_change}, ma_10={ma_10}, ma_50={ma_50}, rsi={rsi}"
            )

            # Проверка роста OI
            oi_growing = oi_change > 0
            logging.debug(f"OI растет: {oi_growing}")

            # Инициализируем сигнал по умолчанию
            signal = 0

            # Медвежий сигнал: Застрявшие лонги
            if (oi_growing and close <= prev_close and
                cvd >= prev_cvd and close <= ma_50 and rsi > 50):
                signal = 1
                logging.debug("Сгенерирован сигнал SELL: Застрявшие лонги.")
            if (oi_growing and close >= prev_close and
                  cvd <= prev_cvd and close >= ma_50 and rsi <= 70):
                signal = -1
                logging.debug("Сгенерирован сигнал BUY: Застрявшие шорты.")
            else:
                logging.debug("Сигнал HOLD: Нет явных дивергенций.")

            data.at[data.index[i], 'Trade_Signal'] = signal

        logging.info('Торговые сигналы успешно рассчитаны.')
        return data

    except Exception as e:
        logging.error(f'Ошибка при расчете торговых сигналов: {e}')
        raise


def backtest_strategy(data):
    try:
        data['Position'] = data['Trade_Signal'].replace(0, np.nan).ffill().fillna(0)
        data['Market_Return'] = data['close'].pct_change()
        data['Strategy_Return'] = data['Market_Return'] * data['Position'].shift(1)
        data['Cumulative_Market_Return'] = (1 + data['Market_Return']).cumprod()
        data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()
        logging.info('Бэктестинг выполнен.')
        return data
    except Exception as e:
        logging.error(f'Ошибка при бэктестинге: {e}')
        raise


def plot_results(data, weights, timeframe, symbol, results_dir):
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Cumulative_Market_Return'], label='Рынок', color='blue')
        plt.plot(data.index, data['Cumulative_Strategy_Return'], label='Стратегия', color='orange')
        plt.title(f'Сравнение Кумулятивной Доходности\nМонета: {symbol}, Таймфрейм: {timeframe}, Весовые Коэффициенты: {weights}')
        plt.xlabel('Дата')
        plt.ylabel('Кумулятивная Доходность')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_file = os.path.join(results_dir, f'cumulative_returns_{timeframe}.png')
        plt.savefig(plot_file)
        plt.close()
        logging.info(f'Результаты визуализированы и сохранены в "{plot_file}".')
    except Exception as e:
        logging.error(f'Ошибка при визуализации результатов: {e}')
        raise


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

def combine_signals_with_weights(data, params):
    """
    Combines the indicator signals using the provided weights.

    Parameters:
    - data (pd.DataFrame): The dataframe containing the signals.
    - params (dict): The parameter dictionary containing weights.

    Returns:
    - pd.DataFrame: Dataframe with the combined Trade_Signal.
    """
    # Compute the weighted sum of signals
    data['Trade_Signal'] = (
        params['w_RSI'] * data['Signal_RSI'] +
        params['w_MACD'] * data['Signal_MACD'] +
        params['w_Stochastic'] * data['Signal_Stochastic'] +
        params['w_OI'] * data['Signal_OI']
    )

    # Optionally, you can normalize or threshold the Trade_Signal
    # For example, to convert to discrete signals:
    data['Trade_Signal'] = data['Trade_Signal'].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    return data
def optimize_parameters(data, param_grid, metric='Cumulative_Strategy_Return'):
    """
    Optimizes the parameters including weights to maximize the specified metric.

    Parameters:
    - data (pd.DataFrame): The dataframe containing the indicators and signals.
    - param_grid (dict): The grid of parameters to search.
    - metric (str): The metric to optimize (default: 'Cumulative_Strategy_Return').

    Returns:
    - best_params (dict): The parameter combination with the best metric.
    - best_metric (float): The best metric value achieved.
    - results_df (pd.DataFrame): Dataframe containing all parameter combinations and their metrics.
    """
    best_params = None
    best_metric = -np.inf
    results = []

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    logging.info(f'Начинается оптимизация по {len(param_combinations)} комбинациям параметров.')

    # Initialize progress bar
    for params in tqdm(param_combinations, desc="Optimizing", unit="combo"):
        try:
            # Combine signals with current weights
            combined_data = combine_signals_with_weights(data.copy(), params)

            # Backtest the strategy
            backtested_data = backtest_strategy(combined_data)

            # Retrieve the metric value (ensure this column exists after backtesting)
            current_metric = backtested_data[metric].iloc[-1]

            results.append((params, current_metric))

            # Update the best parameters if current metric is better
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = params

        except Exception as e:
            logging.error(f'Ошибка при оптимизации с параметрами {params}: {e}')
            continue

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=['Parameters', 'Metric'])

    return best_params, best_metric, results_df


def main():
    import json

    # Simplified JSON data for demonstration (full data is too large to handle here)
    data_symb = [{"symbol":"btc","name":"Bitcoin"},{"symbol":"eth","name":"Ethereum"},{"symbol":"xrp","name":"XRP"},{"symbol":"sol","name":"Solana"},{"symbol":"bnb","name":"BNB"},{"symbol":"doge","name":"Dogecoin"},{"symbol":"usdc","name":"USDC"},{"symbol":"ada","name":"Cardano"},{"symbol":"avax","name":"Avalanche"},{"symbol":"trx","name":"TRON"},{"symbol":"shib","name":"Shiba Inu"},{"symbol":"ton","name":"Toncoin"},{"symbol":"xlm","name":"Stellar"},{"symbol":"dot","name":"Polkadot"},{"symbol":"link","name":"Chainlink"},{"symbol":"bch","name":"Bitcoin Cash"},{"symbol":"sui","name":"Sui"},{"symbol":"hbar","name":"Hedera"},{"symbol":"ltc","name":"Litecoin"},{"symbol":"uni","name":"Uniswap"},{"symbol":"pepe","name":"Pepe"},{"symbol":"leo","name":"UNUS SED LEO"},{"symbol":"near","name":"NEAR Protocol"},{"symbol":"apt","name":"Aptos"},{"symbol":"icp","name":"Internet Computer"},{"symbol":"dai","name":"Dai"},{"symbol":"etc","name":"Ethereum Classic"},{"symbol":"pol","name":"POL (ex-MATIC)"},{"symbol":"cro","name":"Cronos"},{"symbol":"tao","name":"Bittensor"},{"symbol":"render","name":"Render"},{"symbol":"fet","name":"Artificial Superintelligence Alliance"},{"symbol":"fil","name":"Filecoin"},{"symbol":"kas","name":"Kaspa"},{"symbol":"algo","name":"Algorand"},{"symbol":"vet","name":"VeChain"},{"symbol":"arb","name":"Arbitrum"},{"symbol":"aave","name":"Aave"},{"symbol":"atom","name":"Cosmos"},{"symbol":"om","name":"MANTRA"},{"symbol":"stx","name":"Stacks"},{"symbol":"okb","name":"OKB"},{"symbol":"tia","name":"Celestia"},{"symbol":"imx","name":"Immutable"},{"symbol":"bonk","name":"Bonk"},{"symbol":"xmr","name":"Monero"},{"symbol":"wif","name":"dogwifhat"},{"symbol":"mnt","name":"Mantle"},{"symbol":"op","name":"Optimism"},{"symbol":"theta","name":"Theta Network"},{"symbol":"ftm","name":"Fantom"},{"symbol":"inj","name":"Injective"},{"symbol":"grt","name":"The Graph"},{"symbol":"ena","name":"Ethena"},{"symbol":"sei","name":"Sei"},{"symbol":"wld","name":"Worldcoin"},{"symbol":"floki","name":"FLOKI"},{"symbol":"fdusd","name":"First Digital USD"},{"symbol":"brett","name":"Brett (Based)"},{"symbol":"rune","name":"THORChain"},{"symbol":"eos","name":"EOS"},{"symbol":"pyth","name":"Pyth Network"},{"symbol":"ondo","name":"Ondo"},{"symbol":"ar","name":"Arweave"},{"symbol":"xtz","name":"Tezos"},{"symbol":"mkr","name":"Maker"},{"symbol":"ldo","name":"Lido DAO"},{"symbol":"flr","name":"Flare"},{"symbol":"jup","name":"Jupiter"},{"symbol":"gala","name":"Gala"},{"symbol":"strk","name":"Starknet"},{"symbol":"sand","name":"The Sandbox"},{"symbol":"flow","name":"Flow"},{"symbol":"jasmy","name":"JasmyCoin"},{"symbol":"kcs","name":"KuCoin Token"},{"symbol":"ray","name":"Raydium"},{"symbol":"ens","name":"Ethereum Name Service"},{"symbol":"hnt","name":"Helium"},{"symbol":"core","name":"Core"},{"symbol":"bsv","name":"Bitcoin SV"},{"symbol":"qnt","name":"Quant"},{"symbol":"btt","name":"BitTorrent [New]"},{"symbol":"beam","name":"Beam"},{"symbol":"aioz","name":"AIOZ Network"},{"symbol":"axs","name":"Axie Infinity"},{"symbol":"iota","name":"IOTA"},{"symbol":"mana","name":"Decentraland"},{"symbol":"egld","name":"MultiversX"},{"symbol":"neo","name":"Neo"},{"symbol":"popcat","name":"Popcat (SOL)"},{"symbol":"ape","name":"ApeCoin"},{"symbol":"dydx","name":"dYdX (Native)"},{"symbol":"aero","name":"Aerodrome Finance"},{"symbol":"xdc","name":"XDC Network"},{"symbol":"akt","name":"Akash Network"},{"symbol":"cfx","name":"Conflux"},{"symbol":"zec","name":"Zcash"},{"symbol":"xec","name":"eCash"},{"symbol":"mina","name":"Mina"},{"symbol":"chz","name":"Chiliz"},{"symbol":"nexo","name":"Nexo"},{"symbol":"pendle","name":"Pendle"},{"symbol":"crv","name":"Curve DAO Token"},{"symbol":"w","name":"Wormhole"},{"symbol":"mog","name":"Mog Coin"},{"symbol":"not","name":"Notcoin"},{"symbol":"cake","name":"PancakeSwap"},{"symbol":"axl","name":"Axelar"},{"symbol":"ordi","name":"ORDI"},{"symbol":"mew","name":"cat in a dogs world"},{"symbol":"snx","name":"Synthetix"},{"symbol":"ftt","name":"FTX Token"},{"symbol":"eigen","name":"EigenLayer"},{"symbol":"ron","name":"Ronin"},{"symbol":"zk","name":"ZKsync"},{"symbol":"usdd","name":"USDD"},{"symbol":"neiro","name":"Neiro (First Neiro On Ethereum)"},{"symbol":"blur","name":"Blur"},{"symbol":"rose","name":"Oasis"},{"symbol":"ckb","name":"Nervos Network"},{"symbol":"lunc","name":"Terra Classic"},{"symbol":"zro","name":"LayerZero"},{"symbol":"gno","name":"Gnosis"},{"symbol":"comp","name":"Compound"},{"symbol":"super","name":"SuperVerse"},{"symbol":"xaut","name":"Tether Gold"},{"symbol":"dash","name":"Dash"},{"symbol":"kava","name":"Kava"},{"symbol":"ksm","name":"Kusama"},{"symbol":"safe","name":"Safe"},{"symbol":"ctc","name":"Creditcoin"},{"symbol":"btg","name":"Bitcoin Gold"},{"symbol":"bome","name":"BOOK OF MEME"},{"symbol":"1inch","name":"1inch Network"},{"symbol":"tfuel","name":"Theta Fuel"},{"symbol":"astr","name":"Astar"},{"symbol":"amp","name":"Amp"},{"symbol":"1000sats","name":"SATS"},{"symbol":"hot","name":"Holo"},{"symbol":"woo","name":"WOO"},{"symbol":"enj","name":"Enjin Coin"},{"symbol":"lpt","name":"Livepeer"},{"symbol":"nft","name":"APENFT"},{"symbol":"gmt","name":"GMT"},{"symbol":"pyusd","name":"PayPal USD"},{"symbol":"celo","name":"Celo"},{"symbol":"dexe","name":"DeXe"},{"symbol":"ethfi","name":"ether.fi"},{"symbol":"paxg","name":"PAX Gold"},{"symbol":"act","name":"Act I : The AI Prophecy"},{"symbol":"iotx","name":"IoTeX"},{"symbol":"wemix","name":"WEMIX"},{"symbol":"zil","name":"Zilliqa"},{"symbol":"tusd","name":"TrueUSD"},{"symbol":"twt","name":"Trust Wallet Token"},{"symbol":"rsr","name":"Reserve Rights"},{"symbol":"arkm","name":"Arkham"},{"symbol":"meme","name":"Memecoin"},{"symbol":"dym","name":"Dymension"},{"symbol":"zeta","name":"ZetaChain"},{"symbol":"zrx","name":"0x Protocol"},{"symbol":"turbo","name":"Turbo"},{"symbol":"prime","name":"Echelon Prime"},{"symbol":"ethw","name":"EthereumPoW"},{"symbol":"glm","name":"Golem"},{"symbol":"cvx","name":"Convex Finance"},{"symbol":"aevo","name":"Aevo"},{"symbol":"bat","name":"Basic Attention Token"},{"symbol":"jto","name":"Jito"},{"symbol":"xch","name":"Chia"},{"symbol":"manta","name":"Manta Network"},{"symbol":"id","name":"SPACE ID"},{"symbol":"skl","name":"SKALE"},{"symbol":"ankr","name":"Ankr"},{"symbol":"qtum","name":"Qtum"},{"symbol":"osmo","name":"Osmosis"},{"symbol":"elf","name":"aelf"},{"symbol":"sc","name":"Siacoin"},{"symbol":"one","name":"Harmony"},{"symbol":"luna","name":"Terra"},{"symbol":"io","name":"io.net"},{"symbol":"rvn","name":"Ravencoin"},{"symbol":"gas","name":"Gas"},{"symbol":"babydoge","name":"Baby Doge Coin"},{"symbol":"jst","name":"JUST"},{"symbol":"sfp","name":"SafePal"},{"symbol":"ath","name":"Aethir"},{"symbol":"ssv","name":"ssv.network"},{"symbol":"dogs","name":"DOGS"},{"symbol":"metis","name":"Metis"},{"symbol":"sushi","name":"SushiSwap"},{"symbol":"usde","name":"Ethena USDe"},{"symbol":"ftn","name":"Fasttoken"},{"symbol":"dog","name":"Dog (Bitcoin)"},{"symbol":"tel","name":"Telcoin"},{"symbol":"usdy","name":"Ondo US Dollar Yield"},{"symbol":"aleo","name":"Aleo"},{"symbol":"1mbabydoge","name":"Baby Doge Coin"},{"symbol":"mask","name":"Mask Network"},{"symbol":"alt","name":"Altlayer"},{"symbol":"drift","name":"Drift"},{"symbol":"bico","name":"Biconomy"},{"symbol":"mx","name":"MX Token"},{"symbol":"kda","name":"Kadena"},{"symbol":"polyx","name":"Polymesh"},{"symbol":"people","name":"ConstitutionDAO"},{"symbol":"t","name":"Threshold"},{"symbol":"lrc","name":"Loopring"},{"symbol":"fxs","name":"Frax Share"},{"symbol":"xrd","name":"Radix"},{"symbol":"dcr","name":"Decred"},{"symbol":"gmx","name":"GMX"},{"symbol":"g","name":"Gravity"},{"symbol":"moodeng","name":"Moo Deng (moodengsol.com)"},{"symbol":"ponke","name":"Ponke"},{"symbol":"ilv","name":"Illuvium"},{"symbol":"xai","name":"Xai"},{"symbol":"flux","name":"Flux"},{"symbol":"xem","name":"NEM"},{"symbol":"rpl","name":"Rocket Pool"},{"symbol":"band","name":"Band Protocol"},{"symbol":"glmr","name":"Moonbeam"},{"symbol":"tribe","name":"Tribe"},{"symbol":"uma","name":"UMA"},{"symbol":"pixel","name":"Pixels"},{"symbol":"coti","name":"COTI"},{"symbol":"cat","name":"Simon's Cat"},{"symbol":"blast","name":"Blast"},{"symbol":"sxp","name":"Solar"},{"symbol":"ygg","name":"Yield Guild Games"},{"symbol":"dgb","name":"DigiByte"},{"symbol":"yfi","name":"yearn.finance"},{"symbol":"zen","name":"Horizen"},{"symbol":"ont","name":"Ontology"},{"symbol":"mplx","name":"Metaplex"},{"symbol":"vtho","name":"VeThor Token"},{"symbol":"hmstr","name":"Hamster Kombat"},{"symbol":"ach","name":"Alchemy Pay"},{"symbol":"icx","name":"ICON"},{"symbol":"storj","name":"Storj"},{"symbol":"avail","name":"Avail"},{"symbol":"saga","name":"Saga"},{"symbol":"degen","name":"Degen"},{"symbol":"waves","name":"Waves"},{"symbol":"vanry","name":"Vanar Chain"},{"symbol":"agi","name":"Delysium"},{"symbol":"audio","name":"Audius"},{"symbol":"cspr","name":"Casper"},{"symbol":"bnx","name":"BinaryX"},{"symbol":"solo","name":"Sologenic"},{"symbol":"chr","name":"Chromia"},{"symbol":"sun","name":"Sun [New]"},{"symbol":"edu","name":"Open Campus"},{"symbol":"zig","name":"ZIGChain"},{"symbol":"cfg","name":"Centrifuge"},{"symbol":"scrt","name":"Secret"},{"symbol":"xno","name":"Nano"},{"symbol":"merl","name":"Merlin Chain"},{"symbol":"cetus","name":"Cetus Protocol"},{"symbol":"bigtime","name":"Big Time"},{"symbol":"banana","name":"Banana Gun"},{"symbol":"lsk","name":"Lisk"},{"symbol":"joe","name":"JOE"},{"symbol":"tai","name":"TARS AI"},{"symbol":"trb","name":"Tellor"},{"symbol":"iost","name":"IOST"},{"symbol":"waxp","name":"WAX"},{"symbol":"spec","name":"Spectral"},{"symbol":"snt","name":"Status"},{"symbol":"c98","name":"Coin98"},{"symbol":"api3","name":"API3"},{"symbol":"bb","name":"BounceBit"},{"symbol":"ai","name":"Sleepless AI"},{"symbol":"orca","name":"Orca"},{"symbol":"bal","name":"Balancer"},{"symbol":"cpool","name":"Clearpool"},{"symbol":"kub","name":"Bitkub Coin"},{"symbol":"taiko","name":"Taiko"},{"symbol":"xym","name":"Symbol"},{"symbol":"powr","name":"Powerledger"},{"symbol":"xvg","name":"Verge"},{"symbol":"bora","name":"BORA"},{"symbol":"ctsi","name":"Cartesi"},{"symbol":"xvs","name":"Venus"},{"symbol":"iq","name":"IQ"},{"symbol":"fida","name":"Solana Name Service"},{"symbol":"celr","name":"Celer Network"},{"symbol":"ong","name":"Ontology Gas"},{"symbol":"gomining","name":"Gomining"},{"symbol":"zent","name":"Zentry"},{"symbol":"slp","name":"Smooth Love Potion"},{"symbol":"rlc","name":"iExec RLC"},{"symbol":"cvc","name":"Civic"},{"symbol":"portal","name":"Portal"},{"symbol":"ctxc","name":"Cortex"},{"symbol":"ntrn","name":"Neutron"},{"symbol":"coq","name":"Coq Inu"},{"symbol":"mobile","name":"Helium Mobile"},{"symbol":"pyr","name":"Vulcan Forged (PYR)"},{"symbol":"nmr","name":"Numeraire"},{"symbol":"dent","name":"Dent"},{"symbol":"pond","name":"Marlin"},{"symbol":"magic","name":"Treasure"},{"symbol":"pundix","name":"Pundi X (New)"},{"symbol":"tru","name":"TrueFi"},{"symbol":"cati","name":"Catizen"},{"symbol":"movr","name":"Moonriver"},{"symbol":"mvl","name":"MVL"},{"symbol":"cyber","name":"Cyber"},{"symbol":"rio","name":"Realio Network"},{"symbol":"spell","name":"Spell Token"},{"symbol":"strax","name":"Stratis [New]"},{"symbol":"lqty","name":"Liquity"},{"symbol":"hive","name":"Hive"},{"symbol":"cgpt","name":"ChainGPT"},{"symbol":"sundog","name":"SUNDOG"},{"symbol":"bone","name":"Bone ShibaSwap"},{"symbol":"ustc","name":"TerraClassicUSD"},{"symbol":"velo","name":"Velo"},{"symbol":"oas","name":"Oasys"},{"symbol":"ctk","name":"Shentu"},{"symbol":"syn","name":"Synapse"},{"symbol":"slerf","name":"SLERF"},{"symbol":"rif","name":"Rootstock Infrastructure Framework"},{"symbol":"dusk","name":"Dusk"},{"symbol":"naka","name":"Nakamoto Games"},{"symbol":"steem","name":"Steem"},{"symbol":"ark","name":"Ark"},{"symbol":"mav","name":"Maverick Protocol"},{"symbol":"sfund","name":"Seedify.fund"},{"symbol":"knc","name":"Kyber Network Crystal v2"},{"symbol":"pha","name":"Phala Network"},{"symbol":"ace","name":"Fusionist"},{"symbol":"hook","name":"Hooked Protocol"},{"symbol":"high","name":"Highstreet"},{"symbol":"auction","name":"Bounce Token"},{"symbol":"mlk","name":"MiL.k"},{"symbol":"ardr","name":"Ardor"},{"symbol":"prom","name":"Prom"},{"symbol":"omni","name":"Omni Network"},{"symbol":"elon","name":"Dogelon Mars"},{"symbol":"dar","name":"Mines of Dalarnia"},{"symbol":"moca","name":"Moca Network"},{"symbol":"hft","name":"Hashflow"},{"symbol":"zcx","name":"Unizen"},{"symbol":"lmwr","name":"LimeWire"},{"symbol":"aurora","name":"Aurora"},{"symbol":"sys","name":"Syscoin"},{"symbol":"mtl","name":"Metal DAO"},{"symbol":"dodo","name":"DODO"},{"symbol":"phb","name":"Phoenix"},{"symbol":"oxt","name":"Orchid"},{"symbol":"zkj","name":"Polyhedra Network"},{"symbol":"aca","name":"Acala Token"},{"symbol":"agld","name":"Adventure Gold"},{"symbol":"win","name":"WINkLink"},{"symbol":"orbs","name":"Orbs"},{"symbol":"stpt","name":"STP"},{"symbol":"rss3","name":"RSS3"},{"symbol":"rare","name":"SuperRare"},{"symbol":"uxlink","name":"UXLINK"},{"symbol":"usdp","name":"Pax Dollar"},{"symbol":"mbox","name":"MOBOX"},{"symbol":"gtc","name":"Gitcoin"},{"symbol":"nfp","name":"NFPrompt"},{"symbol":"dia","name":"DIA"},{"symbol":"alice","name":"MyNeighborAlice"},{"symbol":"alpha","name":"Stella"},{"symbol":"lon","name":"Tokenlon Network Token"},{"symbol":"myro","name":"Myro"},{"symbol":"wen","name":"Wen"},{"symbol":"raca","name":"RACA"},{"symbol":"qi","name":"BENQI"},{"symbol":"cxt","name":"Covalent X Token"},{"symbol":"rdnt","name":"Radiant Capital"},{"symbol":"req","name":"Request"},{"symbol":"bake","name":"BakeryToken"},{"symbol":"hifi","name":"Hifi Finance"},{"symbol":"ogn","name":"Origin Protocol"},{"symbol":"bnt","name":"Bancor"},{"symbol":"lista","name":"Lista DAO"},{"symbol":"arpa","name":"ARPA"},{"symbol":"myria","name":"Myria"},{"symbol":"loom","name":"Loom Network"},{"symbol":"stmx","name":"StormX"},{"symbol":"rez","name":"Renzo"},{"symbol":"mbx","name":"MARBLEX"},{"symbol":"omi","name":"ECOMI"},{"symbol":"nkn","name":"NKN"},{"symbol":"qkc","name":"QuarkChain"},{"symbol":"mode","name":"Mode"},{"symbol":"lever","name":"LeverFi"},{"symbol":"apex","name":"ApeX Protocol"},{"symbol":"nym","name":"NYM"},{"symbol":"pokt","name":"Pocket Network"},{"symbol":"lto","name":"LTO Network"},{"symbol":"tnsr","name":"Tensor"},{"symbol":"ant","name":"Aragon"},{"symbol":"dao","name":"DAO Maker"},{"symbol":"gns","name":"Gains Network"},{"symbol":"clv","name":"CLV"},{"symbol":"gods","name":"Gods Unchained"},{"symbol":"mavia","name":"Heroes of Mavia"},{"symbol":"gear","name":"Gearbox Protocol"},{"symbol":"mob","name":"MobileCoin"},{"symbol":"cbk","name":"Cobak Token"},{"symbol":"stg","name":"Stargate Finance"},{"symbol":"maneki","name":"MANEKI"},{"symbol":"rad","name":"Radworks"},{"symbol":"badger","name":"Badger DAO"},{"symbol":"zbcn","name":"Zebec Network"},{"symbol":"venom","name":"Venom"},{"symbol":"mbl","name":"MovieBloc"},{"symbol":"acs","name":"Access Protocol"},{"symbol":"ladys","name":"Milady Meme Coin"},{"symbol":"ngl","name":"Entangle"},{"symbol":"vra","name":"Verasity"},{"symbol":"mct","name":"Metacraft"},{"symbol":"tlm","name":"Alien Worlds"},{"symbol":"ata","name":"Automata Network"},{"symbol":"token","name":"TokenFi"},{"symbol":"wrx","name":"WazirX"},{"symbol":"omg","name":"OMG Network"},{"symbol":"rei","name":"REI Network"},{"symbol":"dego","name":"Dego Finance"},{"symbol":"ern","name":"Ethernity Chain"},{"symbol":"blz","name":"Bluzelle"},{"symbol":"tko","name":"Toko Token"},{"symbol":"ghst","name":"Aavegotchi"},{"symbol":"aergo","name":"Aergo"},{"symbol":"fort","name":"Forta"},{"symbol":"lit","name":"Litentry"},{"symbol":"xcn","name":"Onyxcoin"},{"symbol":"cos","name":"Contentos"},{"symbol":"root","name":"The Root Network"},{"symbol":"fire","name":"Matr1x Fire"},{"symbol":"aidoge","name":"ArbDoge AI"},{"symbol":"perp","name":"Perpetual Protocol"},{"symbol":"lat","name":"PlatON"},{"symbol":"aeur","name":"Anchored Coins AEUR"},{"symbol":"forth","name":"Ampleforth Governance Token"},{"symbol":"sidus","name":"SIDUS"},{"symbol":"prcl","name":"Parcl"},{"symbol":"mln","name":"Enzyme"},{"symbol":"flm","name":"Flamingo"},{"symbol":"looks","name":"LooksRare"},{"symbol":"dora","name":"Dora Factory"},{"symbol":"sweat","name":"Sweat Economy"},{"symbol":"bel","name":"Bella Protocol"},{"symbol":"data","name":"Streamr"},{"symbol":"pols","name":"Polkastarter"},{"symbol":"wan","name":"Wanchain"},{"symbol":"mapo","name":"MAP Protocol"},{"symbol":"alcx","name":"Alchemix"},{"symbol":"voxel","name":"Voxies"},{"symbol":"ice","name":"Ice Open Network"},{"symbol":"slf","name":"Self Chain"},{"symbol":"mother","name":"Mother Iggy"},{"symbol":"kmd","name":"Komodo"},{"symbol":"ren","name":"Ren"},{"symbol":"nuls","name":"NULS"},{"symbol":"loka","name":"League of Kingdoms Arena"},{"symbol":"fun","name":"FUNToken"},{"symbol":"ulti","name":"Ultiverse"},{"symbol":"gst","name":"Green Satoshi Token (SOL)"},{"symbol":"vic","name":"Viction"},{"symbol":"vinu","name":"Vita Inu"},{"symbol":"gtai","name":"GT Protocol"},{"symbol":"euri","name":"Eurite"},{"symbol":"lina","name":"Linear Finance"},{"symbol":"boba","name":"Boba Network"},{"symbol":"masa","name":"Masa"},{"symbol":"fb","name":"Fractal Bitcoin"},{"symbol":"beta","name":"Beta Finance"},{"symbol":"dep","name":"DEAPcoin"},{"symbol":"sd","name":"Stader"},{"symbol":"idex","name":"IDEX"},{"symbol":"fis","name":"StaFi"},{"symbol":"bsw","name":"Biswap"},{"symbol":"kishu","name":"Kishu Inu"},{"symbol":"farm","name":"Harvest Finance"},{"symbol":"df","name":"dForce"},{"symbol":"chess","name":"Tranchess"},{"symbol":"quick","name":"QuickSwap [Old]"},{"symbol":"bcut","name":"bitsCrunch"},{"symbol":"mdt","name":"Measurable Data Token"},{"symbol":"reef","name":"Reef"},{"symbol":"ava","name":"AVA"},{"symbol":"pirate","name":"Pirate Nation"},{"symbol":"combo","name":"COMBO"},{"symbol":"ever","name":"Everscale"},{"symbol":"utk","name":"xMoney"},{"symbol":"alpaca","name":"Alpaca Finance"},{"symbol":"cusd","name":"Celo Dollar"},{"symbol":"troy","name":"TROY"},{"symbol":"cream","name":"Cream Finance"},{"symbol":"fio","name":"FIO Protocol"},{"symbol":"order","name":"Orderly Network"},{"symbol":"vidt","name":"VIDT DAO"},{"symbol":"leash","name":"Doge Killer"},{"symbol":"max","name":"Matr1x"},{"symbol":"gme","name":"GmeStop"},{"symbol":"flt","name":"Fluence"},{"symbol":"samo","name":"Samoyedcoin"},{"symbol":"wxt","name":"Wirex Token"},{"symbol":"pros","name":"Prosper"},{"symbol":"vrtx","name":"Vertex Protocol"},{"symbol":"shrap","name":"Shrapnel"},{"symbol":"ceek","name":"CEEK VR"},{"symbol":"pda","name":"PlayDapp"},{"symbol":"gog","name":"Guild of Guardians"},{"symbol":"wing","name":"Wing Finance"},{"symbol":"burger","name":"BurgerCities"},{"symbol":"adx","name":"AdEx"},{"symbol":"prq","name":"PARSIQ"},{"symbol":"a8","name":"Ancient8"},{"symbol":"zkl","name":"zkLink"},{"symbol":"santos","name":"Santos FC Fan Token"},{"symbol":"bzz","name":"Swarm"},{"symbol":"amb","name":"AirDAO"},{"symbol":"bifi","name":"Beefy"},{"symbol":"uft","name":"UniLend"},{"symbol":"pstake","name":"pSTAKE Finance"},{"symbol":"kasta","name":"Kasta"},{"symbol":"ztx","name":"ZTX"},{"symbol":"pivx","name":"PIVX"},{"symbol":"neon","name":"Neon EVM"},{"symbol":"mon","name":"MON"},{"symbol":"sca","name":"Scallop"},{"symbol":"og","name":"OG Fan Token"},{"symbol":"firo","name":"Firo"},{"symbol":"block","name":"Blockasset"},{"symbol":"gmrx","name":"Gaimin"},{"symbol":"dmail","name":"DMAIL Network"},{"symbol":"bob","name":"BOB (ETH)"},{"symbol":"trvl","name":"TRVL (Dtravel)"},{"symbol":"goal","name":"TOPGOAL"},{"symbol":"hard","name":"Kava Lend"},{"symbol":"xcad","name":"XCAD Network"},{"symbol":"krl","name":"Kryll"},{"symbol":"akro","name":"Kaon"},{"symbol":"psg","name":"Paris Saint-Germain Fan Token"},{"symbol":"bar","name":"FC Barcelona Fan Token"},{"symbol":"swftc","name":"SwftCoin"},{"symbol":"ast","name":"AirSwap"},{"symbol":"dlc","name":"Diamond Launch"},{"symbol":"alpine","name":"Alpine F1 Team Fan Token"},{"symbol":"mxc","name":"Moonchain"},{"symbol":"gal","name":"Galxe"},{"symbol":"wifi","name":"WiFi Map"},{"symbol":"iris","name":"IRISnet"},{"symbol":"city","name":"Manchester City Fan Token"},{"symbol":"vib","name":"Viberate"},{"symbol":"time","name":"Chrono.tech"},{"symbol":"yfii","name":"DFI.Money"},{"symbol":"psp","name":"ParaSwap"},{"symbol":"mbs","name":"UNKJD"},{"symbol":"peng","name":"Peng"},{"symbol":"aury","name":"Aurory"},{"symbol":"lazio","name":"S.S. Lazio Fan Token"},{"symbol":"porto","name":"FC Porto Fan Token"},{"symbol":"unfi","name":"Unifi Protocol DAO"},{"symbol":"vite","name":"VITE"},{"symbol":"fon","name":"FONSmartChain"},{"symbol":"mix","name":"MixMarvel"},{"symbol":"silly","name":"Silly Dragon"},{"symbol":"arty","name":"Artyfact"},{"symbol":"dfi","name":"DeFiChain"},{"symbol":"key","name":"SelfKey"},{"symbol":"asr","name":"AS Roma Fan Token"},{"symbol":"tava","name":"ALTAVA"},{"symbol":"juv","name":"Juventus Fan Token"},{"symbol":"duel","name":"GameGPT"},{"symbol":"bond","name":"BarnBridge"},{"symbol":"far","name":"Farcana"},{"symbol":"atm","name":"Atletico De Madrid Fan Token"},{"symbol":"gpt","name":"QnA3.AI"},{"symbol":"mdx","name":"Mdex"},{"symbol":"acm","name":"AC Milan Fan Token"},{"symbol":"bendog","name":"Ben the Dog"},{"symbol":"vext","name":"Veloce"},{"symbol":"aprs","name":"Apeiron"},{"symbol":"gft","name":"Gifto"},{"symbol":"cult","name":"Cult DAO"},{"symbol":"nyan","name":"Nyan Heroes"},{"symbol":"arg","name":"Argentine Football Association Fan Token"},{"symbol":"srm","name":"Serum"},{"symbol":"g3","name":"GAM3S.GG"},{"symbol":"qorpo","name":"QORPO WORLD"},{"symbol":"kp3r","name":"Keep3rV1"},{"symbol":"xzk","name":"Mystiko Network"},{"symbol":"afc","name":"Arsenal Fan Token"},{"symbol":"nibi","name":"Nibiru Chain"},{"symbol":"xr","name":"XRADERS"},{"symbol":"spurs","name":"Tottenham Hotspur Fan Token"},{"symbol":"fitfi","name":"Step App"},{"symbol":"zero","name":"ZeroLend"},{"symbol":"cel","name":"Celsius"},{"symbol":"hero","name":"Metahero"},{"symbol":"inter","name":"Inter Milan Fan Token"},{"symbol":"izi","name":"Izumi Finance"},{"symbol":"rep","name":"Augur"},{"symbol":"sis","name":"Symbiosis"},{"symbol":"gene","name":"Genopets"},{"symbol":"zkf","name":"ZKFair"},{"symbol":"5ire","name":"5ire"},{"symbol":"kan","name":"BitKan"},{"symbol":"elix","name":"Elixir Games"},{"symbol":"insp","name":"Inspect"},{"symbol":"mee","name":"Medieval Empires"},{"symbol":"oax","name":"OAX"},{"symbol":"tra","name":"Trabzonspor Fan Token"},{"symbol":"gswift","name":"GameSwift"},{"symbol":"mcrt","name":"MagicCraft"},{"symbol":"coval","name":"Circuits of Value"},{"symbol":"intx","name":"Intentx"},{"symbol":"sqr","name":"Magic Square"},{"symbol":"caps","name":"Ternoa"},{"symbol":"cvp","name":"PowerPool"},{"symbol":"radar","name":"DappRadar"},{"symbol":"vgx","name":"VGX Token"},{"symbol":"pip","name":"Pip"},{"symbol":"cookie","name":"Cookie"},{"symbol":"bubble","name":"Bubble"},{"symbol":"ese","name":"Eesee"},{"symbol":"gg","name":"Reboot"},{"symbol":"rain","name":"Rain Coin"},{"symbol":"tomi","name":"tomi"},{"symbol":"ptu","name":"Pintu Token"},{"symbol":"stat","name":"STAT"},{"symbol":"sswp","name":"Suiswap"},{"symbol":"rpk","name":"RepubliK"},{"symbol":"dome","name":"Everdome"},{"symbol":"chrp","name":"Chirpley"},{"symbol":"mengo","name":"Flamengo Fan Token"},{"symbol":"trc","name":"MetaTrace"},{"symbol":"saitama","name":"SAITAMA INU"},{"symbol":"mdao","name":"MarsDAO"},{"symbol":"front","name":"Frontier"},{"symbol":"mv","name":"GensoKishi Metaverse"},{"symbol":"ppt","name":"Populous"},{"symbol":"ll","name":"LightLink"},{"symbol":"ort","name":"Okratech Token"},{"symbol":"lamb","name":"Lambda"},{"symbol":"bbl","name":"beoble"},{"symbol":"every","name":"Everyworld"},{"symbol":"por","name":"Portugal National Team Fan Token"},{"symbol":"planet","name":"PLANET"},{"symbol":"cta","name":"Cross The Ages"},{"symbol":"dice","name":"Klaydice"},{"symbol":"ooki","name":"Ooki Protocol"},{"symbol":"epx","name":"Ellipsis"},{"symbol":"ertha","name":"Ertha"},{"symbol":"wsm","name":"Wall Street Memes"},{"symbol":"tama","name":"Tamadoge"},{"symbol":"wlkn","name":"Walken"},{"symbol":"aeg","name":"Aether Games"},{"symbol":"dock","name":"Dock"},{"symbol":"multi","name":"Multichain"},{"symbol":"agla","name":"Angola"},{"symbol":"mcg","name":"MetalCore"},{"symbol":"for","name":"ForTube"},{"symbol":"social","name":"Phavercoin"},{"symbol":"cot","name":"Cosplay Token"},{"symbol":"wwy","name":"WeWay"},{"symbol":"ego","name":"EGO"},{"symbol":"hvh","name":"HAVAH"},{"symbol":"xar","name":"Arcana Network"},{"symbol":"exvg","name":"Exverse"},{"symbol":"sos","name":"OpenDAO"},{"symbol":"kunci","name":"Kunci Coin"},{"symbol":"cwar","name":"Cryowar"},{"symbol":"strm","name":"StreamCoin"},{"symbol":"defi","name":"DeFi"},{"symbol":"dechat","name":"Dechat"},{"symbol":"vega","name":"Vega Protocol"},{"symbol":"rats","name":"GoldenRat"},{"symbol":"rond","name":"ROND"},{"symbol":"mojo","name":"Planet Mojo"},{"symbol":"xwg","name":"X World Games"},{"symbol":"cbx","name":"CropBytes"},{"symbol":"orb","name":"OrbCity"},{"symbol":"elda","name":"Eldarune"},{"symbol":"son","name":"SOUNI"},{"symbol":"pumlx","name":"PUMLx"},{"symbol":"sparta","name":"Spartan Protocol"},{"symbol":"igu","name":"IguVerse"},{"symbol":"thn","name":"Throne"},{"symbol":"aki","name":"Aki Network"},{"symbol":"plt","name":"Palette"},{"symbol":"pnt","name":"pNetwork"},{"symbol":"mtc","name":"Moonft"},{"symbol":"drep","name":"Drep [new]"},{"symbol":"galaxis","name":"Galaxis"},{"symbol":"ply","name":"Aurigami"},{"symbol":"shill","name":"SHILL Token"},{"symbol":"kine","name":"KINE"},{"symbol":"kmon","name":"Kryptomon"},{"symbol":"minu","name":"Minu"},{"symbol":"movez","name":"MOVEZ"},{"symbol":"ctt","name":"Castweet"},{"symbol":"obx","name":"OpenBlox"},{"symbol":"co","name":"Corite"},{"symbol":"toms","name":"TomTomCoin"},{"symbol":"gsts","name":"Gunstar Metaverse"},{"symbol":"sald","name":"Salad"},{"symbol":"dapp","name":"LiquidApps"},{"symbol":"lfw","name":"Linked Finance World"},{"symbol":"next","name":"ShopNEXT"},{"symbol":"kok","name":"KOK"},{"symbol":"azy","name":"Amazy"},{"symbol":"vv","name":"Virtual Versions"},{"symbol":"play","name":"Play Token"},{"symbol":"sats","name":"SATS (Ordinals)"},{"symbol":"l3","name":"Layer3"},{"symbol":"htx","name":"HTX"},{"symbol":"fmc","name":"Fimarkcoin"},{"symbol":"zeus","name":"Zeus Network"},{"symbol":"pixfi","name":"Pixelverse"},{"symbol":"lai","name":"LayerAI"},{"symbol":"kmno","name":"Kamino Finance"},{"symbol":"1cat","name":"Bitcoin Cats"},{"symbol":"foxy","name":"Foxy"},{"symbol":"well","name":"Moonwell"},{"symbol":"param","name":"Param"},{"symbol":"nrn","name":"Neuron"},{"symbol":"cloud","name":"Cloud"},{"symbol":"gummy","name":"GUMMY"},{"symbol":"runecoin","name":"RSIC•GENESIS•RUNE"},{"symbol":"port3","name":"Port3 Network"},{"symbol":"canto","name":"CANTO"},{"symbol":"pundu","name":"Pundu"},{"symbol":"flip","name":"Chainflip"},{"symbol":"xeta","name":"XANA"},{"symbol":"tenet","name":"TENET"},{"symbol":"mak","name":"MetaCene"},{"symbol":"zex","name":"Zeta"},{"symbol":"okt","name":"OKT Chain"},{"symbol":"polydoge","name":"PolyDoge"},{"symbol":"ocean","name":"Ocean Protocol"},{"symbol":"route","name":"Router Protocol (New)"},{"symbol":"fmb","name":"Flappymoonbird"},{"symbol":"turbos","name":"Turbos Finance"},{"symbol":"svl","name":"Slash Vision Labs"},{"symbol":"lbr","name":"Lybra Finance"},{"symbol":"klay","name":"Klaytn"},{"symbol":"pbux","name":"Playbux"},{"symbol":"sqt","name":"SubQuery Network"},{"symbol":"orn","name":"Orion"},{"symbol":"milo","name":"Milo Inu"},{"symbol":"mstar","name":"MerlinStarter"},{"symbol":"xava","name":"Avalaunch"},{"symbol":"nlk","name":"NuLink"},{"symbol":"agix","name":"SingularityNET"},{"symbol":"app","name":"RWAX"},{"symbol":"velar","name":"Velar"},{"symbol":"saros","name":"Saros"},{"symbol":"sqd","name":"Subsquid"},{"symbol":"tap","name":"Tap Protocol"},{"symbol":"zend","name":"zkLend"},{"symbol":"aark","name":"Aark"},{"symbol":"karate","name":"Karate Combat"},{"symbol":"gcake","name":"Pancake Games"},{"symbol":"melos","name":"Melos Studio"},{"symbol":"lft","name":"Lifeform Token"},{"symbol":"seilor","name":"Kryptonite"},{"symbol":"vela","name":"Vela Exchange"},{"symbol":"ness","name":"Ness LAB"},{"symbol":"real","name":"RealLink"},{"symbol":"suia","name":"SUIA"},{"symbol":"lgx","name":"Legion Network"},{"symbol":"lends","name":"Lends"},{"symbol":"gm","name":"GM Everyday"},{"symbol":"tst","name":"Teleport System Token"},{"symbol":"mxm","name":"MixMob"},{"symbol":"aptr","name":"Aperture Finance"},{"symbol":"kon","name":"KONPAY"},{"symbol":"irl","name":"Rebase GG"},{"symbol":"lis","name":"Realis Network"},{"symbol":"star","name":"StarHeroes"},{"symbol":"avive","name":"Avive World"},{"symbol":"vpad","name":"VLaunch"},{"symbol":"nuts","name":"Thetanuts Finance"},{"symbol":"1sol","name":"1Sol"},{"symbol":"seor","name":"SEOR Network"},{"symbol":"bonus","name":"BonusBlock"},{"symbol":"kcal","name":"KCAL"},{"symbol":"hlg","name":"Holograph"},{"symbol":"dzoo","name":"Degen Zoo"},{"symbol":"primal","name":"PRIMAL"},{"symbol":"omn","name":"Omega Network"},{"symbol":"capo","name":"IL CAPO OF CRYPTO"},{"symbol":"brawl","name":"BitBrawl"},{"symbol":"shark","name":"Sharky"},{"symbol":"com","name":"Communis"},{"symbol":"dsrun","name":"Derby Stars"},{"symbol":"vpr","name":"VaporFund"},{"symbol":"jeff","name":"Jeff World"},{"symbol":"grape","name":"GrapeCoin"},{"symbol":"3p","name":"Web3Camp"},{"symbol":"dpx","name":"Dopex"},{"symbol":"ecox","name":"ECOx"},{"symbol":"boring","name":"BoringDAO"},{"symbol":"zam","name":"Zamio"},{"symbol":"fame","name":"Fame MMA"},{"symbol":"sail","name":"Clipper"},{"symbol":"hon","name":"Heroes of NFT"},{"symbol":"candy","name":"Candy Pocket"},{"symbol":"taki","name":"Taki Games"},{"symbol":"mtk","name":"MetaToken"},{"symbol":"qmall","name":"QMALL TOKEN"},{"symbol":"afg","name":"Army of Fortune Gem"},{"symbol":"purse","name":"Pundi X PURSE"},{"symbol":"mmc","name":"MoveMoveCoin"},{"symbol":"elt","name":"EdenLoop"},{"symbol":"musd","name":"Mad USD"},{"symbol":"ruby","name":"Ruby Play Network"},{"symbol":"ggm","name":"Monster Galaxy"},{"symbol":"mnz","name":"Menzy"}]

    # Extracting symbols
    # symbols = [item["symbol"] for item in data_symb]
    symbols = ['ton']
    # timeframes = ['30s', '1min', '5min', '15min', '30min', '1h', '4h']
    timeframes = ['30s']

    # Сетка параметров для оптимизации
    param_grid = {
        # Weights for the linear combination
        'w_RSI': [0, 1, 2, 3],
        'w_MACD': [0, 1, 2, 3],
        'w_Stochastic': [0, 1, 2, 3],
        'w_OI': [0, 1, 2, 3],
    }

    # Общая директория для результатов
    base_results_dir = 'results'
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)

    num = 1

    for timeframe, symbol in product(timeframes, symbols):
        # Создаем отдельную папку для каждого таймфрейма
        results_dir = os.path.join(base_results_dir, timeframe, symbol)
        os.makedirs(results_dir, exist_ok=True)

        # Настройка логирования для текущего таймфрейма
        log_file = os.path.join(results_dir, 'trading_strategy.log')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        setup_logging(log_file=log_file)

        logging.info(f'Начинается обработка таймфрейма: {timeframe}')

        try:
            #обучаемая модель
            # data = load_data(symbol, timeframe)
            data = pd.read_csv('preprocessed_data.csv')

            data = calculate_indicators(data)
            data = compute_indicator_signals(data)

            # Оптимизация параметров
            best_params, best_metric, results_df = optimize_parameters(data, param_grid)

            # Сохранение результатов оптимизации
            opt_file = os.path.join(results_dir, f'optimization_results_{timeframe}.csv')
            results_df.to_csv(opt_file, index=False)
            logging.info(
                f'Результаты оптимизации сохранены в "{opt_file}". Лучшие параметры: {best_params} с метрикой {best_metric:.4f}')

            # Применение лучших параметров и бэктестинг
            data = combine_signals_with_weights(data, best_params)
            data = backtest_strategy(data)

            # Визуализация результатов
            plot_results(data, best_params, timeframe, symbol, results_dir)

            # Вывод результатов
            print_results(data)
            print(symbol, timeframe, num)
            num += 1

            # Сохранение итоговых результатов с сигналами
            final_csv = os.path.join(results_dir, f'strategy_backtest_results_{timeframe}.csv')
            data.to_csv(final_csv, index=False)
            logging.info(f'Итоговые результаты сохранены в "{final_csv}".')


            #hardcode модель

            # Загрузка и подготовка данных
            # data = load_data(symbol, timeframe)
            # data = pd.read_csv('preprocessed_data.csv')
            #
            # # Предварительный расчет сигналов
            # data = compute_advanced_signals(data)
            #
            # # Применение лучших параметров и бэктестинг
            # data = backtest_strategy(data)
            #
            # # Визуализация результатов
            # plot_results(data, 'hardcode', timeframe, symbol, results_dir)
            #
            # # Вывод результатов
            # print_results(data)
            # print(symbol, timeframe, num)
            # num += 1
            #
            # # Сохранение итоговых результатов с сигналами
            # final_csv = os.path.join(results_dir, f'strategy_backtest_results_{timeframe}.csv')
            # data.to_csv(final_csv, index=False)
            # logging.info(f'Итоговые результаты сохранены в "{final_csv}".')

        except Exception as e:
            logging.error(f'Ошибка при обработке таймфрейма {timeframe}: {e}')

        logging.info(f'Обработка таймфрейма {timeframe} завершена.\n')


if __name__ == "__main__":
    main()