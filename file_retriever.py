import re
from eodhd import APIClient
import pandas as pd
from transformers import pipeline
import torch_directml
dml = torch_directml.device()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ast
import matplotlib.pyplot as plt
import mplcursors
import creds

def clean_and_lowercase(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()


def create_csv(ticker, start_date, end_date, limit):
    api = APIClient(creds.api_key)
    resp = api.financial_news(ticker, from_date = start_date, to_date = end_date, limit = limit)  # Get financial news for TIKR
    i = len(resp)
    has = 0
    doesnt_have = 0
    while i:
        i -= 1
        for t in resp[i]['symbols']:
            if t[:3] != ticker[:3]:  # remove any articles that do not exclusively have TIK tickers.
                doesnt_have += 1
            else:
                has += 1
        if doesnt_have == 0:
            pass
        elif float(has / doesnt_have) <= 0.5:
            del resp[i]
        has = 0
        doesnt_have = 0
    df = pd.DataFrame(resp)
    df['content'] = df['content'].apply(clean_and_lowercase)
    return df




def create_complete_csv(ticker):

    start_date = pd.Timestamp('2025-3-15', tz = 'UTC')
    end_date = pd.Timestamp('2025-04-09', tz = 'UTC')
    resp = pd.DataFrame()
    current_date = end_date
    while current_date >= start_date:

        start_str = (current_date - pd.Timedelta(weeks = 1)).strftime('%Y-%m-%d')
        end_str = current_date.strftime('%Y-%m-%d')

        temp_df = create_csv(ticker, start_str, end_str, 1000)
        resp = pd.concat([resp, temp_df])
        current_date = current_date - pd.Timedelta(weeks = 1)
    columns_to_drop = ['title', 'link', 'symbols', 'tags', 'sentiment']
    df = resp.drop(columns = columns_to_drop)
    return df



def create_final_csv(ticker):
    count = 0
    resp = create_complete_csv(ticker)
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = model.to(dml)
    pipe = pipeline("text-classification", model = model, tokenizer = tokenizer, device = -1)
    sentiment_column = []
    for entry in resp['content']:
        res = pipe(entry, padding = 'max_length', truncation = True)
        sentiment_column.append(res)
        if count % 100 == 0:
            print(count)
        count += 1
    resp['sentiment'] = sentiment_column
    resp.to_csv('C:/Users/ahlec/Downloads/' + ticker + '_data1.csv', index = False)
    return resp




def buy_or_sell(csv_file, week_increment):

    result = pd.DataFrame()
    sentiment_df = pd.read_csv(csv_file)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df = sentiment_df.set_index('date')

    start_date = pd.Timestamp('2025-03-15', tz = 'UTC')
    end_date = pd.Timestamp('2025-04-09', tz = 'UTC')

    current_date = start_date

    while current_date <= end_date:
        positives = 0
        negatives = 0
        total = 0

        week_df = sentiment_df[(sentiment_df.index >= current_date) & (sentiment_df.index <= (current_date + pd.Timedelta(weeks = week_increment)))]

        for entry in week_df['sentiment']:
            entry_list = ast.literal_eval(entry)
            label = entry_list[0]['label']

            if label == 'positive':
                positives += 1
                total += 1
            elif label == 'negative':
                negatives += 1
                total += 1

        pos_ratio = positives / total if total else 0
        neg_ratio = negatives / total if total else 0

        buy_or_sell = {}
        signal = "NEUTRAL"

        if pos_ratio > 0.70:
            signal = 'STRONG BUY'
        elif pos_ratio > 0.50:
            signal = 'BUY'
        elif neg_ratio > 0.70:
            signal = 'STRONG SELL'
        elif neg_ratio > 0.50:
            signal = 'SELL'

        buy_or_sell[str(current_date + pd.Timedelta(weeks = week_increment))] = signal


        temp_df = pd.DataFrame(buy_or_sell.items(), columns = ["Period", "Signal"])
        result = pd.concat([result, temp_df])
        current_date += pd.Timedelta(weeks = week_increment)
    result.rename(columns = {'Period': 'Date'}, inplace = True)
    result['Date'] = pd.to_datetime(result['Date'])
    return result




def get_eod_prices(SYMBOL_NAME, period_day_week_month):

    url = f'https://eodhd.com/api/eod/' + SYMBOL_NAME + '.US?from=2021-03-01&to=2025-04-02&period=' + period_day_week_month + '&api_token=67e12d8f64f5d6.71424304&fmt=csv'
    df = pd.read_csv(url)
    columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.drop(columns = columns_to_drop, inplace = True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df




def plot_all_data(df1, df2):

    df1['Date'] = pd.to_datetime(df1['Date']).dt.tz_localize(None)
    df2['Date'] = pd.to_datetime(df2['Date']).dt.tz_localize(None)
    merged_df = pd.merge(df1, df2, on = 'Date')


    color_map = {'BUY': 'green',
                 'STRONG BUY': 'turquoise',
                 'SELL': 'red',
                 'STRONG SELL': 'yellow',
                 'NEUTRAL': 'grey'}


    colors = [color_map.get(signal, 'black') for signal in merged_df['Signal']]
    plt.figure(figsize = (12, 6))
    plt.plot(merged_df['Date'], merged_df['Adjusted_close'], label = 'Adjusted Close', color = 'blue', marker = 'o', linestyle = '-', markersize = 0.5)
    scatter = plt.scatter(merged_df['Date'], merged_df['Adjusted_close'], color = colors, s = 10, zorder = 5)


    cursor = mplcursors.cursor(scatter, hover = True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"Date: {merged_df['Date'].iloc[sel.index].strftime('%Y-%m-%d')}\n"
        f"Price: {merged_df['Adjusted_close'].iloc[sel.index]:.2f}\n"
        f"Signal: {merged_df['Signal'].iloc[sel.index]}"
    ))


    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title('Historical Weekly Price and Signals')
    plt.legend()
    plt.xticks(rotation = 45)

    return plt


def backtest_dca_multiplier(signal_csv_file, ticker, investment_per_period = 100, week_increment = 1):
    ticker_price_data = get_eod_prices(ticker, "w")
    ticker_signal_data = buy_or_sell(signal_csv_file, 1)

    #localize the "Date" row so they match up
    ticker_price_data['Date'] = pd.to_datetime(ticker_price_data['Date']).dt.tz_localize(None)
    ticker_signal_data['Date'] = pd.to_datetime(ticker_signal_data['Date']).dt.tz_localize(None)

    #Merge the dataframes on the date column
    signal_and_price_data = pd.merge(ticker_price_data, ticker_signal_data, on = 'Date')
    signal_and_price_data.sort_values('Date', inplace = True)

    modified_DCA_investment_per_period = investment_per_period * .5
    DCA_shares_accumulated = 0
    DCA_total_invested = 0
    modified_DCA_shares_accumulated = 0
    modified_DCA_total_invested = 0
    modified_DCA_bank = 0
    for index, row in signal_and_price_data.iterrows():


        DCA_total_invested += investment_per_period      #add to the total invested
        DCA_shares_accumulated += (investment_per_period / row['Adjusted_close'])        #add the amount of shares accumulated based on price

        modified_DCA_total_invested += modified_DCA_investment_per_period
        modified_DCA_shares_accumulated += (modified_DCA_investment_per_period / row['Adjusted_close'])
        modified_DCA_bank += modified_DCA_investment_per_period

        if row['Signal'] == 'SELL' or row['Signal'] == 'STRONG SELL':
            curr_amount_to_invest = modified_DCA_bank * .5
            modified_DCA_shares_accumulated += (curr_amount_to_invest / row['Adjusted_close'])
            modified_DCA_total_invested += curr_amount_to_invest
            modified_DCA_bank = modified_DCA_bank * .5



    final_price = signal_and_price_data.iloc[-1]['Adjusted_close']
    modified_DCA_total_invested += modified_DCA_bank

    print("Total invested with DCA: " + str(DCA_total_invested))
    print("Total returned with DCA: " + str(DCA_shares_accumulated * final_price))
    print("Total return percentage with DCA: %" + str(((((DCA_shares_accumulated * final_price) / DCA_total_invested) - 1) * 100)))

    print("Total invested with modified DCA: " + str(modified_DCA_total_invested))
    print("Total returned with modified DCA: " + str((modified_DCA_shares_accumulated * final_price) + modified_DCA_bank))
    print("Total return percentage with modified DCA: %" + str((((((modified_DCA_shares_accumulated * final_price) + modified_DCA_bank) / modified_DCA_total_invested) - 1) * 100)))
























