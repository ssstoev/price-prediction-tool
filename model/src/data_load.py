import logging
from pathlib import Path
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
# ____________________________________________
# CONFIGURATION
# ────────────────────────────────────────────
SUPABASE_HOST     = os.getenv("SUPABASE_HOST")      # aws-0-xx-xxx-1.pooler.supabase.com (pooler) or db.xxxx.supabase.co (direct)
SUPABASE_PORT     = int(os.getenv("SUPABASE_PORT", "6543"))
SUPABASE_DB       = "postgres"
SUPABASE_USER     = os.getenv("SUPABASE_USER")      # pooler: postgres.projectref  |  direct: postgres
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
CSV_PATH          = os.path.join(os.path.dirname(__file__), "../data/ads_appartments_cleaned.csv")
BATCH_SIZE        = 100                              # rows per insert batch
#______________________________________________
log = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    query = '''SELECT * FROM ads_appartments'''
    conn = None
    try:
        print("🔌 Connecting to Supabase...")
        conn = psycopg2.connect(
            host=SUPABASE_HOST,
            port=SUPABASE_PORT,
            dbname=SUPABASE_DB,
            user=SUPABASE_USER,
            password=SUPABASE_PASSWORD,
            sslmode="require"
        )

        df = pd.read_sql(query, conn)
        log.info("Loaded %d rows from DB", len(df))
        return df
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        if conn:
            log.error("Data file not found: %s")
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("🔒 Connection closed.")