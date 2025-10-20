import pickle
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine

from dagster import op, job, schedule, resource
from mob_money_fraud_module_refactored import transform

# PIPELINE FOR MOBILE MONEY FRAUD (GHS)
# ─── Resources ─────────────────────────────────────────────────────────────────

@resource(config_schema={"conn_str": str})
def db_engine(context):
    """
    SQLAlchemy engine resource that yields a connection.
    This ensures the connection is properly closed after each use.
    """
    engine = create_engine(context.resource_config["conn_str"])
    conn = None
    try:
        conn = engine.connect()
        yield conn
    finally:
        if conn:
            conn.close()


@resource(config_schema={
    "model_path": str,
    "scaler_path": str,
    "training_columns_path": str,
})
def model_store(context):
    """
    Loads scaler and model from disk once per job run.
    """
    with open(context.resource_config["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    with open(context.resource_config["model_path"], "rb") as f:
        model = pickle.load(f)
    with open(context.resource_config["training_columns_path"], "rb") as f:
        training_cols = pickle.load(f)
    return {"scaler": scaler, "model": model, "training_cols": training_cols}


# ─── Ops ────────────────────────────────────────────────────────────────────────

@op(required_resource_keys={"db_engine"})
def extract_transactions(context) -> pd.DataFrame:
    """
    Queries the last 30 minutes of transactions.
    """
    engine = context.resources.db_engine
    end = datetime.now() # current time
    start = end - timedelta(minutes=300) # 300 minutes ago

    sql = f"""
        SELECT 
        a.name AS "merchant", c.name AS "customer_name", c.email AS "customer_email",
        c.phone AS "customer_phone_number", p.reference AS "payment_reference", ps.reference AS "transaction_reference",
        ps.merchant_bears_cost, p.created_at AS "payment_created_at", p.completed_at AS "payment_completed_at",
        p.updated_at AS "payment_updated_at", p.for_disbursement_wallet, p.allowed_payment_methods,
        a.created_at AS "account_created_date", a.risk_level AS "account_risk_level", ps.payment_source_type,
        ps.processor AS "processor", ps.status AS "payment_status", ps.channel AS "payment_channel",
        ps.amount_collected, ps.payment_reversals_type, ps.currency AS "currency",
        t.mobile_number AS "mobile_number_for_mobile_money", t.network_provider AS "mobile_network_provider",
        t.processor_response AS "processor_response", ps.message AS "kora_translated_processor_response",
        ps.amount AS "amount", t.auth_model, ps.created_at AS "tran_date", 0 AS "is_fraud"
    FROM korapay_mobile_money.transaction_nsano_mobile_money t
    INNER JOIN payment_sources ps ON ps.processor_reference = t.reference
    INNER JOIN payments p ON p.id = ps.payment_id
    INNER JOIN accounts a ON a.id = p.account_id
    INNER JOIN customers c ON c.id = p.customer_id
    WHERE p.created_at >= '{start}' AND p.created_at < '{end}'
    AND processor_response NOT IN ('TARGET_AUTHORIZATION_ERROR', 'AUTHORIZATION_SENDER_ACCOUNT_NOT_ACTIVE', 'ACCOUNTHOLDER_WITH_FRI_NOT_FOUND', 'RESOURCE_NOT_FOUND')
    UNION
    SELECT
        a.name AS "merchant", c.name AS "customer_name", c.email AS "customer_email",
        c.phone AS "customer_phone_number", p.reference AS "payment_reference", ps.reference AS "transaction_reference",
        ps.merchant_bears_cost, p.created_at AS "payment_create_date", p.completed_at AS "payment_complete_date",
        p.updated_at AS "payment_updated_at", p.for_disbursement_wallet, p.allowed_payment_methods,
        a.created_at AS "account_create_date", a.risk_level AS "account_risk_level", ps.payment_source_type,
        ps.processor AS "processor", ps.status AS "payment_status", ps.channel AS "payment_channel",
        ps.amount_collected, ps.payment_reversals_type, ps.currency AS "currency",
        t.mobile_number AS "mobile_number_for_mobile_money", t.network_provider AS "mobile_network_provider",
        t.processor_response AS "processor_response", ps.message AS "kora_translated_processor_response",
        ps.amount AS "amount", t.auth_model, ps.created_at AS "tran_date", 1 AS "is_fraud"
    FROM korapay_mobile_money.transaction_nsano_mobile_money t
    INNER JOIN payment_sources ps ON ps.processor_reference = t.reference
    INNER JOIN payments p ON p.id = ps.payment_id
    INNER JOIN accounts a ON a.id = p.account_id
    INNER JOIN customers c ON c.id = p.customer_id
    WHERE p.created_at >= '{start}' AND p.created_at < '{end}'
    AND processor_response IN ('TARGET_AUTHORIZATION_ERROR', 'AUTHORIZATION_SENDER_ACCOUNT_NOT_ACTIVE', 'ACCOUNTHOLDER_WITH_FRI_NOT_FOUND', 'RESOURCE_NOT_FOUND')
    ORDER BY "merchant", "mobile_number_for_mobile_money";
        -- LIMIT 200
    """
    df = pd.read_sql(sql, engine) #check if the read_sql prevents SQL injection
    context.log.info(f"Pulled {len(df)} transactions from {start} to {end}")
    return df


@op
def transform_features(context, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transform() logic.
    """
    context.log.info(f"Actual DataFrame columns: {df.columns.to_list()}")
    return transform(df)


@op(required_resource_keys={"model_store"})
def predict_fraud(context, df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Scale features, predict, and return transaction_reference + predicted_fraud.
    """
    resources = context.resources.model_store
    scaler = resources["scaler"]
    model = resources["model"]
    training_cols = resources["training_cols"]

    # Keep a copy of the original columns for the final output
    original_df = df_feat[["transaction_reference", "is_fraud"]].copy()

    current_cols = df_feat.columns.to_list()

    for col in training_cols:
        if col not in current_cols:
            df_feat[col] = 0

    # drops columns not seen during training
    df_aligned = df_feat[training_cols]

    # drop any non-feature cols
    # feature_cols = [c for c in df_feat.columns if c not in ("transaction_reference", "is_fraud")]
    X = scaler.transform(df_aligned) # changed from df_feat[feature_cols] to df_aligned

    # df_feat["predicted_fraud"] = model.predict(X)
    # return df_feat[["transaction_reference", "predicted_fraud"]]

    original_df["predicted_fraud"] = model.predict(X)
    return original_df


@op(required_resource_keys={"db_engine"})
def load_results(context, df_pred: pd.DataFrame):
    """
    Append the batch of predictions to fraud_predictions.
    """
    engine = context.resources.db_engine
    df_pred.to_sql("fraud_predictions", engine, if_exists="append", index=False)
    context.log.info(f"Wrote {len(df_pred)} predictions to fraud_predictions")


# ─── Job & Schedule ─────────────────────────────────────────────────────────────

@job(
    resource_defs={
        "db_engine": db_engine,
        "model_store": model_store,
    }
)
def fraud_detection_job():
    df_raw = extract_transactions()
    df_feat = transform_features(df_raw)
    df_pred = predict_fraud(df_feat)
    load_results(df_pred)


@schedule(
    cron_schedule="*/300 * * * *", # Every 300 minutes
    job=fraud_detection_job,
    execution_timezone="Africa/Lagos"
)
def fraud_detection_schedule(_context):
    """
    Every 30 minutes, provide the DB URI and model paths.
    """
    return {
        "resources": {
            "db_engine": {
                "config": {
                    "conn_str": "mysql+mysqlconnector://christopher568935m:f5k5Fe?CX$yakmCs&YK7JnDL@artemis-db-100-02-rep-01.c4hc4rp7skgd.eu-west-1.rds.amazonaws.com:3306/korapay_core_engine"
                } # this is a placeholder connection string
            },
            "model_store": {
                "config": {
                    "scaler_path": "models/scaler.pkl",
                    "model_path": "models/ghs_fraud_model_rfc.pkl",
                    "training_columns_path": "models/training_columns.pkl" #added this
                }
            }
        }
    }
