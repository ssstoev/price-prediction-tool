import logging
from pathlib import Path
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)
# ____________________________________________
# CONFIGURATION
# ────────────────────────────────────────────
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
# CSV_PATH          = os.path.join(os.path.dirname(__file__), "../data/ads_appartments_cleaned.csv")
# BATCH_SIZE        = 100                              # rows per insert batch
#______________________________________________
log = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    query = '''SELECT * FROM public.ads_appartments'''
    conn = None
    try:
        print("🔌 Connecting to Neon...")
        conn = psycopg2.connect(NEON_DATABASE_URL)

        with conn.cursor() as cur:
            cur.execute(query)
            colnames = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=colnames)
        log.info("Loaded %d rows from DB", len(df))
        df = df[(df["price_m2_eur"] > 600) & (df["price_m2_eur"] < 15000) &
                (df["size_m2"] > 10) & (df["size_m2"] < 500)]
        
        print(f"length of df is: {len(df)}")
        return df
    except psycopg2.Error as e:
        log.error("❌ Database error: %s", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("🔒 Connection closed.")