from datetime import datetime
import logging
import pandas as pd
import requests
import yfinance as yf

# yfinanceの不要なエラーログ出力を抑制
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ==========================================
# 設定項目（日本株用・緩和版）
# ==========================================
# 対象市場: 'プライム', 'スタンダード', 'グロース', または None (全市場対象)
MARKET_TARGET = 'プライム'

# フィルター条件
MIN_TRADING_VALUE = 300_000_000  # 最低売買代金（20日平均3億円以上）
MIN_VOL_RATIO = 1.10  # 出来高増加率（1.2 -> 1.10倍に緩和）
RSI_MIN = 40  # RSIの下限
RSI_MAX = 70  # RSIの上限（65 -> 70に拡大）
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


def check_swing_signal(ticker_symbol: str):
  """テクニカル＋流動性＋過熱感の多角フィルターによるシグナル判定"""
  try:
    # データの取得（過去1年分）
    df = yf.download(
        ticker_symbol,
        period='1y',
        interval='1d',
        multi_level_index=False,
        progress=False,
    )
    if df.empty or len(df) < 100:  # 必要データ数を200->100日に緩和
      return None

    # 列名のフォーマット調整（yfinanceの仕様差異対策）
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    # 1. 移動平均線（25日, 75日）※200日線は除外
    df['SMA25'] = df['Close'].rolling(window=25).mean()
    df['SMA75'] = df['Close'].rolling(window=75).mean()

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

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    sma25, sma75 = float(latest['SMA25']), float(latest['SMA75'])
    close_price, prev_low = float(latest['Close']), float(prev['Low'])

    vol_latest = float(latest['Volume'])
    vol_sma20 = float(latest['Vol_SMA20'])
    trading_value_sma20 = float(latest['Trading_Value_SMA20'])
    rsi_latest = float(latest['RSI'])

    # --- 判定条件の緩和 ---
    # トレンド判定: 25日線 > 75日線（短期中期上昇）に緩和
    is_uptrend = sma25 > sma75

    # 押し目判定: 25日線の3%以内まで近づいたら押し目とみなす
    is_dip = (prev_low <= sma25 * 1.03) and (close_price >= sma25 * 0.98)

    has_liquidity = trading_value_sma20 >= MIN_TRADING_VALUE
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
          'コード': ticker_symbol,
          '現在株価': round(close_price, 1),
          'RSI(14)': round(rsi_latest, 1),
          '出来高倍率': round(vol_latest / vol_sma20, 2),
          '売買代金(百万円)': round(trading_value_sma20 / 1_000_000, 0),
          '目標利確(+5%)': round(close_price * TAKE_PROFIT_RATIO, 1),
          '損切り(-4%)': round(close_price * STOP_LOSS_RATIO, 1),
      }
  except Exception:
    return None
  return None


if __name__ == '__main__':
  # 1. 銘柄リスト取得
  watch_list = get_jpx_tickers(market_filter=MARKET_TARGET)

  results = []
  print('--- 多角的スクリーニングを開始します ---')

  for i, ticker in enumerate(watch_list, 1):
    print(f'[{i}/{len(watch_list)}] 分析中: {ticker}', end='\r')
    signal = check_swing_signal(ticker)
    if signal:
      print(
          f'\n【買シグナル検知】{signal["コード"]} | 株価:'
          f' {signal["現在株価"]}円 | RSI: {signal["RSI(14)"]} | 出来高:'
          f' {signal["出来高倍率"]}倍'
      )
      results.append(signal)

  print('\n--- スクリーニング完了 ---')

# 2. 結果のCSV保存
  if results:
    result_df = pd.DataFrame(results)
    target_name = MARKET_TARGET if MARKET_TARGET else 'ALL'
    filename = f"swing_candidates_{target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(
        f'抽出結果（{len(results)}件）をCSVファイルに保存しました:'
        f' {filename}'
    )
  else:
    print('本日条件を満たす銘柄は見つかりませんでした。')
