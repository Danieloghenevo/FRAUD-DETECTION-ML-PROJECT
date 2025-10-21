import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# HELPER FUNCTIONS ----------------------------------------------------------------

def calculate_customer_metrics(group):
    """
    Calculates aggregated metrics related to phone number usage for a customer.
    Returns only the new metrics as a pandas Series, which is ideal for groupby().apply().
    """
    unique_phone_numbers = group['mobile_number_for_mobile_money'].nunique()
    diff_number_txns = (group['phone_number_diff'] != 0).sum()
    total_transactions = len(group)
    proportion_different = diff_number_txns / total_transactions if total_transactions > 0 else 0
    differences = group['phone_number_diff'].abs()
    differences = differences[differences != 0]
    average_difference = differences.mean() if not differences.empty else 0

    return pd.Series({
        'unique_phone_numbers': unique_phone_numbers,
        'diff_number_txns': diff_number_txns,
        'total_transactions': total_transactions,
        'proportion_different_numbers': proportion_different,
        'average_phone_number_difference': average_difference
    })

def flag_suspicious_phone_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags transactions where a customer uses multiple phone numbers and makes a high-value transaction.
    This is a vectorized operation, which is much faster than looping.
    """
    # Find customers who use more than one phone number
    multi_phone_customers = df.groupby('customer_email')['mobile_number_for_mobile_money'].nunique()
    multi_phone_customers = multi_phone_customers[multi_phone_customers > 1].index

    # Find the 90th percentile amount for each of these customers
    high_amount_thresholds = df[df['customer_email'].isin(multi_phone_customers)].groupby('customer_email')['amount'].quantile(0.90)

    # Create a map from customer_email to their high-amount threshold
    threshold_map = high_amount_thresholds.to_dict()
    df['high_amount_threshold'] = df['customer_email'].map(threshold_map)

    # Flag is 1 if the customer is a multi-phone user AND their transaction amount is above their threshold
    df['phone_txn_amount_suspicious'] = (
        (df['customer_email'].isin(multi_phone_customers)) &
        (df['amount'] > df['high_amount_threshold'])
    ).astype(int)

    df = df.drop(columns=['high_amount_threshold'])  # Clean up the temporary column
    
    return df


def flag_duplicate_txns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags transactions that appear to be rapid, repeated attempts (3 or more)
    for the same amount by the same customer within a 15-minute window.
    This is an optimized, vectorized approach.
    """
    time_window = pd.Timedelta(minutes=15)
    df['payment_created_at'] = pd.to_datetime(df['payment_created_at'])
    df = df.sort_values(by=['customer_email', 'payment_created_at'])

    # Identify consecutive transactions by the same customer with the same amount
    is_potential_duplicate = (
        (df['customer_email'] == df['customer_email'].shift(1)) &
        (df['amount'] == df['amount'].shift(1))
    )

    # Check if they are within the time window
    time_diff = df['payment_created_at'].diff()
    is_within_window = time_diff <= time_window
    
    # A transaction is a duplicate candidate if it matches the previous one in customer, amount, and time
    duplicate_series = (is_potential_duplicate & is_within_window)

    # Use `cumsum` to create groups of consecutive duplicates
    consecutive_groups = (duplicate_series == False).cumsum()
    
    # Count the size of each consecutive group
    group_sizes = duplicate_series.groupby(consecutive_groups).transform('sum')

    # Flag if the group size is >= 2 (which means 3 total transactions: the first + 2 duplicates)
    df['duplicate_txn_flag'] = (group_sizes >= 2).astype(int)

    return df


# MAIN TRANSFORM FUNCTION ---------------------------------------------------------

def transform(data_extract: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to preprocess the transaction data and engineer features for fraud detection.
    """
    print('Transforming...')
    df = data_extract.copy()
    df = df.drop_duplicates()
    
    # --- 1. Initial Cleaning and Datetime Conversion ---
    df = df.drop(columns=['allowed_payment_methods', 'payment_completed_at', 'for_disbursement_wallet',
                        'payment_source_type', 'currency', 'processor', 'auth_model'], axis=1, errors='ignore')
    df['payment_created_at'] = pd.to_datetime(df['payment_created_at'])
    df['payment_updated_at'] = pd.to_datetime(df['payment_updated_at'])

    # --- 2. Date and Time Feature Engineering ---
    df['created_dow'] = df['payment_created_at'].dt.day_name()
    df['created_hod'] = df['payment_created_at'].dt.hour
    df['created_month'] = df['payment_created_at'].dt.month_name()
    df['payment_completion_time_secs'] = (df['payment_updated_at'] - df['payment_created_at']).dt.total_seconds()
    df['odd_hour_flag'] = ((df['created_hod'] >= 0) & (df['created_hod'] <= 6)).astype(int)

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

    # --- 3. Customer Behavior Metrics ---
    # Calculate phone number difference using transform for efficiency and to avoid index issues.
    df['mobile_number_for_mobile_money'] = pd.to_numeric(df['mobile_number_for_mobile_money'], errors='coerce')
    df['phone_number_diff'] = df.groupby('customer_email')['mobile_number_for_mobile_money'].transform('diff').fillna(0)
    
    # Calculate and join customer phone usage metrics. 
    # The 'include_groups=False' argument is the definitive fix for the "columns overlap" issue.
    metrics_df = df.groupby('customer_email').apply(calculate_customer_metrics, include_groups=False)
    print(metrics_df.head())
    df = pd.merge(
        df,
        metrics_df,
        left_on='customer_email',
        right_index=True,
        how='left'
    )

    # Flag suspicious phone/amount combinations (vectorized)
    df = flag_suspicious_phone_amount(df)
    print(df.head())
    # Flag duplicate transactions (vectorized)
    df = flag_duplicate_txns(df)
    print(df.head())
    # --- 4. Transaction Frequency and Spike Features ---
    df['txn_count_per_customer'] = df.groupby('customer_email')['transaction_reference'].transform('count')
    df['rolling_mean_amount'] = df.groupby('customer_email')['amount'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['amount_spike_flag'] = (df['amount'] > 2.5 * df['rolling_mean_amount']).astype(int)
    df['txn_count_per_customer_per_hour'] = df.groupby(['customer_email', 'created_hod'])['transaction_reference'].transform('count')
    max_txns_per_hour_per_customer = df.groupby(['customer_email', 'created_dow', 'created_hod'])['transaction_reference'].count().groupby(['customer_email', 'created_dow']).max()
    df = df.merge(max_txns_per_hour_per_customer.rename('max_txns_in_an_hour_per_customer'), left_on=['customer_email', 'created_dow'], right_index=True, how='left')
    df['txn_count_per_customer'] = df.groupby('customer_email')['transaction_reference'].transform('count')
    df['rolling_txn_count_per_customer'] = df.groupby('customer_email')['transaction_reference'].transform(lambda x: x.rolling(window=7, min_periods=1).count()) # 7-day rolling window
    df['txn_spike_flag'] = (df['txn_count_per_customer'] > 2.5 * df['rolling_txn_count_per_customer']).astype(int) # Example: 2.5x the rolling average

    # frequency of txns per customer per day
    df['txn_count_per_customer_per_day'] = df.groupby(['customer_email', 'created_dow'])['transaction_reference'].transform('count')
    # --- 5. One-Hot Encoding for Categorical Features ---
    categorical_cols = {
        'account_risk_level': 'account_risk_level',
        'payment_status': 'payment_status',
        'payment_channel': 'payment_channel',
        'mobile_network_provider': 'mobile_network_provider',
        'created_dow': 'created_dow'
    }
    for col, prefix in categorical_cols.items():
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefix, dummy_na=False).astype(int)
            df = pd.concat([df, dummies], axis=1)

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
    
    # --- 6. Rule-Based and Heuristic Features ---
    # Shared phone number flag
    print(df.head())
    shared_counts = df.groupby('mobile_number_for_mobile_money')['customer_email'].transform('nunique')
    df['shared_phone_number_flag'] = (shared_counts > 1).astype(int)
    
    # Unusual transaction amount flag
    df['avg_txn_amount_per_customer'] = df.groupby('customer_email')['amount'].transform('mean')
    is_high_volume_customer = df['total_transactions'] > 9
    is_unusual_amount = df['amount'] > (5 * df['avg_txn_amount_per_customer'])
    df['unusual_txn_flag'] = (is_high_volume_customer & is_unusual_amount).astype(int)

    # Confidence Score calculation
    df['confidence_score'] = 0
    df.loc[(df['phone_number_diff'].abs() > 0) & (df['phone_number_diff'].abs() < 101), 'confidence_score'] += 8
    df.loc[df['unique_phone_numbers'] > 5, 'confidence_score'] += 8
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
    df['confidence_score'] = scaler.fit_transform(df[['confidence_score']])

    # --- 7. Final Cleanup ---
    # Define columns to drop. Using a set for efficient lookup.
    irrelevant_cols = {
        'customer_email', 'payment_reference', 'merchant', 'customer_name', 
        'customer_phone_number', 'tran_date', 'payment_created_at', 
        'payment_updated_at', 'account_created_date', 'account_risk_level', 
        'payment_status', 'payment_channel', 'payment_reversals_type', 
        'mobile_number_for_mobile_money', 'mobile_network_provider', 
        'processor_response', 'created_month', 'kora_translated_processor_response', 
        'amount_collected', 'created_dow', 'payment_completion_time_secs'
    }
    
    # Drop only the columns that actually exist in the DataFrame
    cols_to_drop = list(irrelevant_cols.intersection(df.columns))
    df = df.drop(columns=cols_to_drop)

    print('Transformation done!')
    
    return df
