from datetime import datetime
import io
import logging
import pandas as pd
import requests
import yfinance as yf

# yfinanceの不要なエラーログ出力を抑制
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ==========================================
# 設定項目（数値調整版）
# ==========================================
MIN_TRADING_VALUE_USD = 5_000_000  # 1日平均売買代金（500万ドル以上）
MIN_VOL_RATIO = 1.10  # 出来高増加率（1.2倍 -> 1.10倍に緩和）
RSI_MIN = 40  # RSI下限
RSI_MAX = 70  # RSI上限（65 -> 70に拡大）
TAKE_PROFIT_RATIO = 1.05  # 目標利確（+10% -> +5%に変更）
STOP_LOSS_RATIO = 0.96  # 損切り（-4%）


def get_nasdaq_100_tickers():
  """NASDAQ 100の銘柄リストを取得（二重ヘッダー対応＋バックアップ機能付）"""
  print('NASDAQ 100 銘柄リストを取得中...')
  try:
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ' (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
    }
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))

    for df in tables:
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns]

      for col in df.columns:
        col_str = str(col).lower()
        if 'ticker' in col_str or 'symbol' in col_str:
          raw_tickers = df[col].dropna().astype(str).tolist()
          tickers = []
          for t in raw_tickers:
            t_clean = t.strip().replace('.', '-')
            if 1 <= len(t_clean) <= 5:
              tickers.append(t_clean)

          if len(tickers) >= 50:
            print(f'取得完了: {len(tickers)} 銘柄')
            return tickers

    raise ValueError('該当するテーブル列が見つかりませんでした。')

  except Exception as e:
    print(
        f'WEBからの自動取得に失敗したため、バックアップ銘柄リストを使用します（理由: {e}）'
    )
    fallback_tickers = [
        'AAPL',
        'MSFT',
        'NVDA',
        'AMZN',
        'META',
        'GOOGL',
        'GOOG',
        'AVGO',
        'TSLA',
        'COST',
        'PEP',
        'TMUS',
        'CSCO',
        'ADBE',
        'NFLX',
        'AMD',
        'AMAT',
        'INTC',
        'TXN',
        'AMGN',
        'QCOM',
        'HON',
        'CMCSA',
        'INTU',
        'BKNG',
        'ISRG',
        'VRTX',
        'SBUX',
        'PANW',
        'MDLZ',
        'GILD',
        'LRCX',
        'REGN',
        'ADP',
        'ADI',
        'MU',
        'MELI',
        'KLAC',
        'PDD',
        'SNPS',
        'CDNS',
        'CSX',
        'PYPL',
        'CRWD',
        'MAR',
        'ORLY',
        'ASML',
        'CTAS',
        'ROP',
        'ROST',
        'CPRT',
        'ADSK',
        'PCAR',
        'DXCM',
        'PAYX',
        'FTNT',
        'AEP',
        'MRVL',
        'ODFL',
        'CHTR',
        'MCHP',
        'AZN',
        'KDP',
        'LULU',
        'EXC',
        'FAST',
        'IDXX',
        'KHC',
        'CSGP',
        'BKR',
        'CTSH',
        'GEHC',
        'ON',
        'VRSK',
        'CDW',
        'MNST',
        'DLTR',
        'BIIB',
        'TTD',
        'CEG',
        'FANG',
        'TEAM',
        'MDB',
        'ZS',
        'ILMN',
        'WBD',
        'GFS',
        'ARM',
    ]
    print(f'取得完了: {len(fallback_tickers)} 銘柄（バックアップ機能発動）')
    return fallback_tickers


def check_swing_signal(ticker_symbol: str):
  """テクニカル＋流動性＋過熱感の判定"""
  try:
    df = yf.download(
        ticker_symbol,
        period='1y',
        interval='1d',
        multi_level_index=False,
        progress=False,
    )
    if df.empty or len(df) < 100:  # 必要データ数を200->100に緩和
      return None

    # 列名のフォーマット調整（yfinanceの仕様差異対策）
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    # 1. 移動平均線（25日, 75日）
    df['SMA25'] = df['Close'].rolling(window=25).mean()
    df['SMA75'] = df['Close'].rolling(window=75).mean()

    # 2. 出来高平均（20日）と売買代金（USD）
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Trading_Value'] = df['Close'] * df['Volume']
    df['Trading_Value_SMA20'] = df['Trading_Value'].rolling(window=20).mean()

    # 3. RSI（14日）
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    sma25, sma75 = float(latest['SMA25']), float(latest['SMA75'])
    close_price = float(latest['Close'])
    prev_low = float(prev['Low'])

    vol_latest = float(latest['Volume'])
    vol_sma20 = float(latest['Vol_SMA20'])
    trading_value_sma20 = float(latest['Trading_Value_SMA20'])
    rsi_latest = float(latest['RSI'])

    # --- 判定条件の緩和 ---
    # トレンド判定: 200日線を外し、25日線 > 75日線（短期中期上昇）に緩和
    is_uptrend = sma25 > sma75

    # 押し目判定: 25日線の3%以内まで近づいたら押し目とみなす（条件緩和）
    is_dip = (prev_low <= sma25 * 1.03) and (close_price >= sma25 * 0.98)

    # その他指標
    has_liquidity = trading_value_sma20 >= MIN_TRADING_VALUE_USD
    is_volume_up = vol_latest >= vol_sma20 * MIN_VOL_RATIO
    is_rsi_proper = RSI_MIN <= rsi_latest <= RSI_MAX

    if (
        is_uptrend
        and is_dip
        and has_liquidity
        and is_volume_up
        and is_rsi_proper
    ):
      return {
          'Ticker': ticker_symbol,
          '現在株価($)': round(close_price, 2),
          'RSI(14)': round(rsi_latest, 1),
          '出来高倍率': round(vol_latest / vol_sma20, 2),
          '売買代金(M$)': round(trading_value_sma20 / 1_000_000, 1),
          '目標利確(+5%)': round(close_price * TAKE_PROFIT_RATIO, 2),
          '損切り(-4%)': round(close_price * STOP_LOSS_RATIO, 2),
      }
  except Exception:
    return None
  return None


if __name__ == '__main__':
  watch_list = get_nasdaq_100_tickers()

  results = []
  print('--- ナスダック スクリーニングを開始します ---')

  for i, ticker in enumerate(watch_list, 1):
    print(f'[{i}/{len(watch_list)}] 分析中: {ticker}', end='\r')
    signal = check_swing_signal(ticker)
    if signal:
      print(
          f'\n【買シグナル】{signal["Ticker"]} | 株価: ${signal["現在株価($)"]}'
          f' | RSI: {signal["RSI(14)"]} | 出来高: {signal["出来高倍率"]}倍'
      )
      results.append(signal)

  print('\n--- スクリーニング完了 ---')

if results:
    result_df = pd.DataFrame(results)
    filename = f"swing_candidates_NASDAQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(
        f'抽出結果（{len(results)}件）をCSVファイルに保存しました:'
        f' {filename}'
    )
  else:
    print('本日条件を満たす銘柄は見つかりませんでした。')
