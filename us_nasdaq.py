from datetime import datetime, timezone
import io
import logging
import pandas as pd
import requests
import yfinance as yf

# yfinanceの不要なエラーログ出力を抑制
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ==========================================
# 設定項目（NASDAQ 100用・高精度厳格化版）
# ==========================================
MIN_TRADING_VALUE_USD = 20_000_000  # 売買代金2000万ドル/日以上
MIN_VOL_RATIO = 1.20  # 出来高増加率（1.2倍以上）
RSI_MIN = 45  # RSIの下限
RSI_MAX = 65  # RSIの上限
HIGH_52W_RATIO = 0.90  # 52週高値からの距離（-10%以内）
MA50_DEV_MIN = 0.02  # 50日MA乖離率の下限 (+2%)
MA50_DEV_MAX = 0.10  # 50日MA乖離率の上限 (+10%)
EARNINGS_BUFFER_DAYS = 7  # 決算前後7日を除外

TAKE_PROFIT_RATIO = 1.05  # 目標利確（+5%）
STOP_LOSS_RATIO = 0.96  # 損切り（-4%）


def get_nasdaq100_tickers():
    """NASDAQ 100の構成銘柄リストを取得"""
    print('NASDAQ 100 銘柄リストを取得中...')
    try:
        url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                ' (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))

        df = None
        for table in tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                df = table
                break

        if df is None:
            raise ValueError('銘柄テーブルが見つかりませんでした。')

        col_name = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
        tickers = [str(t).replace('.', '-') for t in df[col_name].tolist()]
        print(f'取得完了: {len(tickers)} 銘柄')
        return tickers
    except Exception as e:
        print(f'NASDAQ 100の取得に失敗しました: {e}')
        return []


def is_near_earnings(ticker_obj) -> bool:
    """決算日の前後N日以内かどうかを判定"""
    try:
        earnings_df = ticker_obj.earnings_dates
        if earnings_df is None or earnings_df.empty:
            return False

        now = pd.Timestamp.now(tz=timezone.utc)
        for index in earnings_df.index:
            earning_date = pd.to_datetime(index)
            if earning_date.tzinfo is None:
                earning_date = earning_date.tz_localize(timezone.utc)

            diff_days = abs((earning_date - now).days)
            if diff_days <= EARNINGS_BUFFER_DAYS:
                return True
    except Exception:
        pass
    return False


def check_swing_signal(ticker_symbol: str):
    """高度なテクニカル＋モメンタム＋決算回避＋50日線乖離率フィルター（米国株版）"""
    try:
        tk = yf.Ticker(ticker_symbol)
        df = tk.history(period='1y', interval='1d')

        if df.empty or len(df) < 200:
            return None

        # 1. 移動平均線（25日, 50日, 200日）
        df['SMA25'] = df['Close'].rolling(window=25).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()

        # 2. 出来高平均（20日）と売買代金
        df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['Trading_Value'] = df['Close'] * df['Volume']
        df['Trading_Value_SMA20'] = df['Trading_Value'].rolling(window=20).mean()

        # 3. RSI（14日）
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 4. 52週高値
        high_52w = df['High'].tail(252).max()

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close_price = float(latest['Close'])
        prev_low = float(prev['Low'])
        sma25 = float(latest['SMA25'])
        sma50 = float(latest['SMA50'])
        sma200 = float(latest['SMA200'])

        vol_latest = float(latest['Volume'])
        vol_sma20 = float(latest['Vol_SMA20'])
        trading_value_sma20 = float(latest['Trading_Value_SMA20'])
        rsi_latest = float(latest['RSI'])

        # 50日線乖離率の計算
        ma50_dev = (close_price / sma50) - 1.0

        # --- 判定条件 ---
        is_uptrend = (close_price > sma50) and (sma50 > sma200)
        is_near_52w_high = close_price >= (high_52w * HIGH_52W_RATIO)
        is_proper_ma50_dev = MA50_DEV_MIN <= ma50_dev <= MA50_DEV_MAX
        is_dip = (prev_low <= sma25 * 1.02) and (close_price >= sma25 * 0.98)

        has_liquidity = trading_value_sma20 >= MIN_TRADING_VALUE_USD
        is_volume_up = vol_latest >= vol_sma20 * MIN_VOL_RATIO
        is_rsi_proper = RSI_MIN <= rsi_latest <= RSI_MAX

        if (
            is_uptrend
            and is_near_52w_high
            and is_proper_ma50_dev
            and is_dip
            and has_liquidity
            and is_volume_up
            and is_rsi_proper
        ):
            if is_near_earnings(tk):
                return None

            return {
                'Ticker': ticker_symbol,
                '現在株価($)': round(close_price, 2),
                'RSI(14)': round(rsi_latest, 1),
                '50日線乖離率': f'{round(ma50_dev * 100, 1)}%',
                '出来高倍率': round(vol_latest / vol_sma20, 2),
                '売買代金(M$)': round(trading_value_sma20 / 1_000_000, 1),
                '52週高値比': f'{round((close_price / high_52w) * 100, 1)}%',
                '目標利確(+5%)': round(close_price * TAKE_PROFIT_RATIO, 2),
                '損切り(-4%)': round(close_price * STOP_LOSS_RATIO, 2),
            }
    except Exception:
        return None
    return None


if __name__ == '__main__':
    watch_list = get_nasdaq100_tickers()

    results = []
    print('--- NASDAQ 100 高精度スクリーニングを開始します ---')

    for i, ticker in enumerate(watch_list, 1):
        print(f'[{i}/{len(watch_list)}] 分析中: {ticker}', end='\r')
        signal = check_swing_signal(ticker)
        if signal:
            print(
                f'\n【買シグナル】{signal["Ticker"]} | 株価: ${signal["現在株価($)"]}'
                f' | 50日線乖離: {signal["50日線乖離率"]} | RSI:'
                f' {signal["RSI(14)"]}'
            )
            results.append(signal)

    print('\n--- スクリーニング完了 ---')

    if results:
        result_df = pd.DataFrame(results)
        filename = "us_nasdaq_results.csv"
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f'抽出結果（{len(results)}件）をCSVファイルに保存しました: {filename}')
    else:
        print('本日条件を満たす銘柄は見つかりませんでした。')
