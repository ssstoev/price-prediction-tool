import logging
from pathlib import Path
import pandas as pd
import psycopg
from dotenv import load_dotenv
import os

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)
# ____________________________________________
# CONFIGURATION
# ────────────────────────────────────────────
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
#______________________________________________
log = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    query = '''SELECT * FROM public.ads_appartments'''
    conn = None
    try:
        print("🔌 Connecting to Neon...")
        conn = psycopg.connect(NEON_DATABASE_URL)

        with conn.cursor() as cur:
            cur.execute(query)
            colnames = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=colnames)
        log.info("Loaded %d rows from DB", len(df))

        # WIP: wrap this data cleaning step in a function
        # At least REMOVE the hardcoded numbers
        df = df[(df["price_m2_eur"] > 600) & (df["price_m2_eur"] < 15000) &
                (df["size_m2"] > 10) & (df["size_m2"] < 500)]

        log.info(f"Length of df is: {len(df)}")
        return df
    except psycopg.Error as e:
        log.error("❌ Database error: %s", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            log.info("🔒 Connection closed.")