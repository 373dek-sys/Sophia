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

    # インデントを main ブロック内に揃える
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
