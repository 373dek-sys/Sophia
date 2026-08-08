from datetime import datetime, timezone
import logging
import pandas as pd
import requests
import yfinance as yf

# yfinanceの不要なエラーログ出力を抑制
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ==========================================
# 設定項目（日本株用・高精度厳格化版）
# ==========================================
MARKET_TARGET = 'プライム'  # 対象市場: 'プライム', 'スタンダード', 'グロース', または None

MIN_TRADING_VALUE = 1_000_000_000  # 最低売買代金（20日平均 10億円以上）
MIN_VOL_RATIO = 1.20  # 出来高増加率（1.2倍以上）
RSI_MIN = 45  # RSIの下限
RSI_MAX = 65  # RSIの上限
HIGH_52W_RATIO = 0.90  # 52週高値からの距離（高値の90%以上＝-10%以内）
EARNINGS_BUFFER_DAYS = 7  # 決算前後の危険期間（前後7日を除外）

TAKE_PROFIT_RATIO = 1.05  # 目標利確（+5%）
STOP_LOSS_RATIO = 0.96  # 損切り（-4%）


def get_jpx_tickers(market_filter=None):
    """JPX公式のExcelから上場銘柄リストを自動取得"""
    print('日本取引所グループ(JPX)から最新の銘柄リストを取得中...')
    url = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'

    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                ' (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers)
        df = pd.read_excel(response.content)

        if market_filter:
            df = df[df['市場・商品区分'].str.contains(market_filter, na=False)]

        tickers = [f'{code}.T' for code in df['コード']]
        print(
            f'取得完了: {len(tickers)} 銘柄（対象市場:'
            f' {market_filter or "全市場"}）'
        )
        return tickers
    except Exception as e:
        print(f'銘柄リストの取得に失敗しました: {e}')
        return []


def is_near_earnings(ticker_obj) -> bool:
    """決算日の前後N日以内かどうかを判定"""
    try:
        earnings_df = ticker_obj.earnings_dates
        if earnings_df is None or earnings_df.empty:
            return False

        now = pd.Timestamp.now(tz=timezone.utc)
        # UTCタイムゾーンに合わせて日付計算
        for index in earnings_df.index:
            earning_date = pd.to_datetime(index)
            if earning_date.tzinfo is None:
                earning_date = earning_date.tz_localize(timezone.utc)

            diff_days = abs((earning_date - now).days)
            if diff_days <= EARNINGS_BUFFER_DAYS:
                return True  # 決算前後の危険期間に該当
    except Exception:
        pass
    return False


def check_swing_signal(ticker_symbol: str):
    """高度なテクニカル＋モメンタム＋決算回避フィルター"""
    try:
        tk = yf.Ticker(ticker_symbol)
        df = tk.history(period='1y', interval='1d')

        if df.empty or len(df) < 200:  # 200日移動平均計算用に最長データ必要
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

        # 4. 52週高値（過去252営業日の最高値）
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

        # --- 判定条件 ---
        # ① トレンド順配列: 株価 > 50日線 かつ 50日線 > 200日線
        is_uptrend = (close_price > sma50) and (sma50 > sma200)

        # ② 52週高値からの距離: -10%以内（高値の90%以上）
        is_near_52w_high = close_price >= (high_52w * HIGH_52W_RATIO)

        # 押し目判定: 25日移動平均線の2%以内まで調整していること
        is_dip = (prev_low <= sma25 * 1.02) and (close_price >= sma25 * 0.98)

        # 流動性・出来高・RSI
        has_liquidity = trading_value_sma20 >= MIN_TRADING_VALUE
        is_volume_up = vol_latest >= vol_sma20 * MIN_VOL_RATIO
        is_rsi_proper = RSI_MIN <= rsi_latest <= RSI_MAX

        # すべてのテクニカル条件をクリアした場合のみ、決算日チェックを実行（処理速度最適化のため）
        if (
            is_uptrend
            and is_near_52w_high
            and is_dip
            and has_liquidity
            and is_volume_up
            and is_rsi_proper
        ):
            # ③ 決算前後7日以内の銘柄を除外
            if is_near_earnings(tk):
                return None

            return {
                'コード': ticker_symbol,
                '現在株価': round(close_price, 1),
                'RSI(14)': round(rsi_latest, 1),
                '出来高倍率': round(vol_latest / vol_sma20, 2),
                '売買代金(百万円)': round(trading_value_sma20 / 1_000_000, 0),
                '52週高値比': f'{round((close_price / high_52w) * 100, 1)}%',
                '目標利確(+5%)': round(close_price * TAKE_PROFIT_RATIO, 1),
                '損切り(-4%)': round(close_price * STOP_LOSS_RATIO, 1),
            }
    except Exception:
        return None
    return None


if __name__ == '__main__':
    watch_list = get_jpx_tickers(market_filter=MARKET_TARGET)

    results = []
    print('--- 高精度スクリーニング（トレンド・52週高値・決算回避）を開始します ---')

    for i, ticker in enumerate(watch_list, 1):
        print(f'[{i}/{len(watch_list)}] 分析中: {ticker}', end='\r')
        signal = check_swing_signal(ticker)
        if signal:
            print(
                f'\n【買シグナル検知】{signal["コード"]} | 株価:'
                f' {signal["現在株価"]}円 | RSI: {signal["RSI(14)"]} | 高値比:'
                f' {signal["52週高値比"]}'
            )
            results.append(signal)

    print('\n--- スクリーニング完了 ---')

    if results:
        result_df = pd.DataFrame(results)
        filename = "jp_stock_results.csv"
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(
            f'抽出結果（{len(results)}件）をCSVファイルに保存しました:'
            f' {filename}'
        )
    else:
        print('本日条件を満たす銘柄は見つかりませんでした。')
