import pandas as pd
import warnings
import statistics
import pickle
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# calculating last digit difference
def calculate_phone_number_diff(group):
    group['mobile_number_for_mobile_money'] = pd.to_numeric(group['mobile_number_for_mobile_money'], errors='coerce')
    group['phone_number_diff'] = group['mobile_number_for_mobile_money'].diff()
    group['phone_number_diff'] = group['phone_number_diff'].fillna(0)  # Or -999 for first transaction
    return group

# calculating metrics and merging directly into df1
def calculate_customer_metrics(group):
    unique_phone_numbers = group['mobile_number_for_mobile_money'].nunique()
    diff_number_txns = (group['phone_number_diff']!= 0).sum()
    total_transactions = len(group)
    proportion_different = diff_number_txns / total_transactions if total_transactions > 0 else 0
    differences = group['phone_number_diff'].abs()
    differences = differences[differences!= 0]
    average_difference = differences.mean() if not differences.empty else 0

    return pd.Series({
        'unique_phone_numbers': unique_phone_numbers,
        'diff_number_txns': diff_number_txns,
        'total_transactions': total_transactions,
        'proportion_different_numbers': proportion_different,
        'average_phone_number_difference': average_difference
    })

# checking out phone number pattern + high transaction amount
def check_phone_amount(group):
    group['phone_txn_amount_suspicious'] = 0
    for customer_email in group['customer_email'].unique():
        customer_group = group[group['customer_email'] == customer_email]
        phone_numbers = customer_group['mobile_number_for_mobile_money'].unique()
        if len(phone_numbers) > 1: #Check if the customer has more than one phone number
            high_amounts = customer_group[customer_group['amount'] > customer_group['amount'].quantile(0.90)] #Check for high transaction amounts
            if not high_amounts.empty: #If there are high transaction amounts, flag them
                group.loc[high_amounts.index, 'phone_txn_amount_suspicious'] = 1
    return group

# Function to flag duplicate transactions within the time window
def flag_duplicate_txns(df):
    time_window = 15 * 60  # 15 minutes
    df['payment_created_at'] = pd.to_datetime(df['payment_created_at'])
    df['duplicate_txn_flag'] = 0
    df['time_diff'] = df.groupby('customer_email')['payment_created_at'].diff().dt.total_seconds()

    for i in range(len(df)):
        if df['time_diff'][i] <= time_window:  # check if txns by a customer are within time window
            # Find potential duplicates in the window
            start_index = i - 1
            end_index = i
            while start_index >= 0 and df['customer_email'][start_index] == df['customer_email'][i] and \
            df['amount'][start_index] == df['amount'][i] and \
            df['payment_created_at'][i] - df['payment_created_at'][start_index] <= pd.Timedelta(seconds=time_window):
                start_index -= 1
            while end_index < len(df) and df['customer_email'][end_index] == df['customer_email'][i] and \
            df['amount'][end_index] == df['amount'][i] and \
            df['payment_created_at'][end_index] - df['payment_created_at'][i] <= pd.Timedelta(seconds=time_window):
                end_index += 1

            # Flag if at least 3 attempts
            if end_index - start_index - 1 >= 3:
                df['duplicate_txn_flag'][start_index + 1:end_index] = 1

    return df



def transform(data_extract):
    """Function preprocesses test data"""
    print('Transforming...')
    df = data_extract
    df = df.drop_duplicates()
    
    #rearranging order
    new_order = ['transaction_reference', 'payment_reference', 'merchant', 
             'customer_name', 'customer_email', 'customer_phone_number',
             'merchant_bears_cost', 'tran_date', 'payment_created_at', 
             'payment_completed_at', 'payment_updated_at', 'for_disbursement_wallet',
             'allowed_payment_methods', 'account_created_date', 'account_risk_level',
             'payment_source_type', 'processor', 'payment_status', 'payment_channel',
             'payment_reversals_type', 'currency', 'auth_model',
             'mobile_number_for_mobile_money', 'mobile_network_provider',
             'processor_response', 'kora_translated_processor_response', 'amount',
             'amount_collected', 'is_fraud']
    df = df[new_order]
    print("done rearranging order...")
    df['customer_phone_number'] = df['customer_phone_number'].astype(str)

    # dropping useless columns
    df = df.drop(columns=['allowed_payment_methods', 'payment_completed_at', 'for_disbursement_wallet', 
                          'payment_source_type', 'currency', 'processor', 'auth_model'], axis=1)
    print('done dropping useless columns...')
    # ensuring both variables are in datetime format
    df['payment_created_at'] = pd.to_datetime(df['payment_created_at'])
    df['payment_updated_at'] = pd.to_datetime(df['payment_updated_at'])
    # extracting features from 'payment_created_on'
    df['created_dow'] = df['payment_created_at'].dt.day_name()  # dow (day of week the txn was created)
    df['created_hod'] = df['payment_created_at'].dt.hour  # hod (hour of day the txn was created i.e 0-23)
    df['created_month'] = df['payment_created_at'].dt.month_name()  # month txn was created
    # extracting features from 'payment_completed_on'
    df['completed_dow'] = df['payment_updated_at'].dt.day_name()
    df['completed_hod'] = df['payment_updated_at'].dt.hour
    df['completed_month'] = df['payment_updated_at'].dt.month_name()

    df['payment_completion_time_secs'] = (df['payment_updated_at'] - df['payment_created_at']).dt.total_seconds()

    df = df.groupby('customer_email').apply(calculate_phone_number_diff)
    df = df.rename(columns={'customer_email': 'cust_email'}) # renaming customer_email to avoid errors
    df = df.reset_index()
    metrics_suffix = '_metrics' # Suffix for the metrics columns

    # --- START: DEBUGGING CODE ---
    # metrics_df = df.groupby('customer_email').apply(calculate_customer_metrics)
    # print(f"COLUMNS IN DF (Left side of join): {df.columns.to_list()}")
    # print(f"COLUMNS IN METRICS_DF (Right side of join): {metrics_df.columns.to_list()}")
    # --- END: DEBUGGING CODE ---

    # df = df.join(metrics_df, on='customer_email', how='left')
    df = df.join(df.groupby('customer_email').apply(calculate_customer_metrics), on='customer_email', how='left', rsuffix=metrics_suffix) #added the rsuffix to avoid column name conflicts
    print('done joining df with suffix...')
    print(df.head(5))  # Debugging: Check the first few rows of the DataFrame
    # Identify Shared Phone Numbers with Different Customer Info (Binary Flag)
    shared_phone_numbers = df.groupby('mobile_number_for_mobile_money', as_index=False).filter(lambda x: len(x) > 1)
    shared_phone_numbers = shared_phone_numbers.groupby('mobile_number_for_mobile_money', as_index=False).agg({'customer_email': 'nunique'}) # Count unique customer_ids for each phone
    shared_phone_numbers = shared_phone_numbers[shared_phone_numbers['customer_email'] > 1] #Keep phone numbers shared by more than one customer
    shared_phone_numbers['shared_phone_number_flag'] = 1  # Create the binary flag
    shared_phone_numbers = shared_phone_numbers[['mobile_number_for_mobile_money', 'shared_phone_number_flag']] #Keep only the phone number and the shared_phone_number_flag
    df = pd.merge(df, shared_phone_numbers, on='mobile_number_for_mobile_money', how='left')
    df['shared_phone_number_flag'] = df['shared_phone_number_flag'].fillna(0)
    print(df.head(5))  # Debugging: Check the first few rows of the DataFrame
    # Renaming columns to remove the suffix
    # metric_columns = [
    # 'unique_phone_numbers', 
    # 'diff_number_txns', 
    # 'total_transactions', 
    # 'proportion_different_numbers', 
    # 'average_phone_number_difference'
    # ]
    # #creating a dictionary to map the suffixed names back to the original names
    # rename_mapping = {
    #     f"{col}{metrics_suffix}": col 
    #     for col in metric_columns 
    #     if f"{col}{metrics_suffix}" in df.columns
    # }
    # #renaming the columns using the mapping
    # df = df.rename(columns=rename_mapping)
    print('about to merge metrics with suffix')
    df['shared_phone_number_flag']= df.shared_phone_number_flag.astype(int)
    df['unique_phone_numbers_metrics']= df.unique_phone_numbers_metrics.astype(int)
    df['diff_number_txns_metrics']= df.diff_number_txns_metrics.astype(int)
    df['total_transactions_metrics']= df.total_transactions_metrics.astype(int)
    print('done merging metrics with suffix')
    # frequency of txns per customer
    df['txn_count_per_customer'] = df.groupby('customer_email')['transaction_reference'].transform('count')
    # frequency of txns per customer per hour
    df['txn_count_per_customer_per_hour'] = df.groupby(['customer_email', 'created_hod'])['transaction_reference'].transform('count')

    # maximum transactions within an hour (per day)
    max_txns_per_hour_per_customer = df.groupby(['customer_email', 'created_dow', 'created_hod'])['transaction_reference'].count().groupby(['customer_email', 'created_dow']).max()
    df = df.merge(max_txns_per_hour_per_customer.rename('max_txns_in_an_hour_per_customer'), left_on=['customer_email', 'created_dow'], right_index=True, how='left')

    # frequency of txns per customer per day
    df['txn_count_per_customer_per_day'] = df.groupby(['customer_email', 'created_dow'])['transaction_reference'].transform('count')
    
    df = df.groupby('customer_email').apply(check_phone_amount).reset_index(drop=True)
    print('done checking phone amount...')
    # sudden spikes in txn count per customer
    # Here, I'm using a rolling window (7 days) to calculate a moving average of txn counts per customer.
    # The window size can be adjusted as needed. The spike flag is set if the current txn count is significantly 
    # higher (I set it to 2.5x here) than the rolling average.
    df['txn_count_per_customer'] = df.groupby('customer_email')['transaction_reference'].transform('count')
    df['rolling_txn_count_per_customer'] = df.groupby('customer_email')['transaction_reference'].transform(lambda x: x.rolling(window=7, min_periods=1).count()) # 7-day rolling window
    df['txn_spike_flag'] = (df['txn_count_per_customer'] > 2.5 * df['rolling_txn_count_per_customer']).astype(int) # Example: 2.5x the rolling average
    print('done calculating txn spike flag...')
    # unusual txn amounts per customer
    # txns 5x higher than a customer's average txn where customer's total txn count is >=10
    df['avg_txn_amount_per_customer'] = df.groupby('customer_email')['amount'].transform('mean')
    # txns where total_transactions > 9
    total_txn_threshold = df['total_transactions_metrics'] > 9
    df['unusual_txn_flag'] = 0
    df.loc[total_txn_threshold, 'unusual_txn_flag'] = (df.loc[total_txn_threshold, 'amount'] > 
                                                        5 * df.loc[total_txn_threshold, 'avg_txn_amount_per_customer']).astype(int)
    
    # Calculate rolling mean and standard deviation of transaction amount
    df['rolling_mean_amount'] = df.groupby('customer_email')['amount'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    # txns with a sudden spike in amount (i.e. above 2.5x the rolling mean)
    df['amount_spike_flag'] = (df['amount'] > 2.5 * df['rolling_mean_amount']).astype(int)

    df = flag_duplicate_txns(df)
    print('done flagging duplicate txns...')
    #odd hour txns
    df['odd_hour_flag'] = ((df['created_hod'] >= 0) & (df['created_hod'] <= 6)).astype(int)

    # FEATURE ENGINEERING
    # merchant
    df['is_Exness'] = (df['merchant']=='Exness').astype(int)
    df['is_Nexasoft'] = (df['merchant']=='Nexasoft Limited').astype(int)
    df['is_Headway'] = (df['merchant']=='Headway').astype(int)
    df['is_Blenet'] = (df['merchant']=='BLENET LTD').astype(int)
    df['is_Hongkong'] = (df['merchant']=='HONGKONG FORTUNETECH LIMITED').astype(int)
    df['is_Bitolo'] = (df['merchant']=='Bitolo').astype(int)
    df['is_Onus'] = (df['merchant']=='OnUs Financial Services').astype(int)
    df['is_Astropay'] = (df['merchant']=='AstroPay').astype(int)
    df['is_Spoynt'] = (df['merchant']=='Spoynt Limited').astype(int)
    df['is_Gateexpress'] = (df['merchant']=='Gate Express').astype(int)
    # account_risk_level
    account_risk_level_encoded = pd.get_dummies(df['account_risk_level'], prefix='account_risk_level').astype(int)
    df = pd.concat([df, account_risk_level_encoded], axis=1)
    # payment_status
    payment_status_encoded = pd.get_dummies(df['payment_status'], prefix='payment_status').astype(int)
    df = pd.concat([df, payment_status_encoded], axis=1)
    # payment_channel
    payment_channel_encoded = pd.get_dummies(df['payment_channel'], prefix='payment_channel').astype(int)
    df = pd.concat([df, payment_channel_encoded], axis=1)
    # mobile_network_provider
    mobile_network_provider_encoded = pd.get_dummies(df['mobile_network_provider'], prefix='mobile_network_provider').astype(int)
    df = pd.concat([df, mobile_network_provider_encoded], axis=1)
    # kora_translated_processor_response
    df['is_success'] = (df['kora_translated_processor_response']=='Charge successful').astype(int)
    df['is_insufficient_funds'] = ((df['kora_translated_processor_response']=='Insufficient funds or \
    transaction limit reached') | (df['kora_translated_processor_response']=='The balance is \
    insufficient for the transaction') | (df['kora_translated_processor_response']=='The amount\
    requested is above the limit permitted by your financial institution, please contact your financial institution')).astype(int)
    df['is_charge_attempt_failed'] = (df['kora_translated_processor_response']=='Charge attempt failed, please try again').astype(int)
    df['is_wallet_not_active'] = (df['kora_translated_processor_response']=='Your wallet is not active. Please contact your financial institution').astype(int)
    df['is_invalid_recipient_wallet'] = (df['kora_translated_processor_response']=='The recipient wallet is invalid').astype(int)
    df['is_invalid_PIN'] = (df['kora_translated_processor_response']=='Incorrect PIN').astype(int)
    # created_dow
    created_dow_encoded = pd.get_dummies(df['created_dow'], prefix='created_dow').astype(int)
    df = pd.concat([df, created_dow_encoded], axis=1)
    print('done encoding categorical features...')
    # confidence score
    df['confidence_score'] = 0
    df.loc[(df['phone_number_diff'].abs() > 0) & (df['phone_number_diff'].abs() < 101), 'confidence_score'] += 8
    df.loc[df['unique_phone_numbers_metrics'] > 5, 'confidence_score'] += 8
    df.loc[df['txn_count_per_customer_per_hour'] > 10, 'confidence_score'] += 8
    df.loc[df['txn_count_per_customer_per_day'] > 20, 'confidence_score'] += 8
    df.loc[df['txn_spike_flag'] == 1, 'confidence_score'] += 7
    df.loc[df['amount_spike_flag'] == 1, 'confidence_score'] += 8
    df.loc[df['unusual_txn_flag'] == 1, 'confidence_score'] += 8
    df.loc[df['phone_txn_amount_suspicious'] == 1, 'confidence_score'] += 7
    df.loc[df['duplicate_txn_flag'] == 1, 'confidence_score'] += 8
    df.loc[df['odd_hour_flag'] == 1, 'confidence_score'] += 7
    df.loc[df['is_invalid_PIN'] == 1, 'confidence_score'] += 7

    scaler = MinMaxScaler()
    df['confidence_score'] = scaler.fit_transform(df[['confidence_score']])  # normalising between 0 and 1
    print('done calculating confidence score...')
    df['temp_fraud'] = (
        ((df['phone_number_diff'].abs() > 0) & (df['phone_number_diff'].abs() < 101)) |
        (df['unique_phone_numbers_metrics'] > 3) |
        (df['txn_count_per_customer_per_hour'] > 10) |
        (df['txn_count_per_customer_per_day'] > 20) |
        (df['amount_spike_flag'] == 1) |
        (df['unusual_txn_flag'] == 1) |
        (df['duplicate_txn_flag'] == 1) |
        (df['confidence_score'] >= 0.3)
    ).astype(int)
    print('done calculating temp fraud...')
    # dropping columns
    irrelevant_cols = ['customer_email', 'level_1', 'payment_reference', 'merchant', 'customer_name', 
                    'cust_email', 'customer_phone_number', 'tran_date', 'created_dow',
                    'payment_created_at', 'payment_updated_at', 'account_created_date',
                    'account_risk_level', 'payment_status', 'payment_channel',
                    'payment_reversals_type', 'mobile_number_for_mobile_money',
                    'mobile_network_provider', 'processor_response', 'created_month',
                    'kora_translated_processor_response', 'amount_collected', 
                    'created_dow','completed_dow', 'completed_hod', 'completed_month',
                    'payment_completion_time_secs', 'time_diff']
    df.drop(columns=irrelevant_cols, inplace=True)
    print('done dropping irrelevant columns...')
    print('Transformation done!')

    return df


# ['id', 'reference', 'transaction_reference', 'channel', 'currency', 'amount', 
#  'amount_charged', 'status', 'auth_model', 'processor_reference', 'mobile_number', 
#  'network_provider', 'processor_response', 'processor_response_code', 
#  'otp_attempts', 'transaction_date', 'created_at', 'updated_at']