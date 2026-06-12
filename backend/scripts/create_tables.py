import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()
# ─────────────────────────────────────────────
# CONFIGURATION — loaded from .env
# ─────────────────────────────────────────────
SUPABASE_HOST     = os.getenv("SUPABASE_HOST")
SUPABASE_PORT     = int(os.getenv("SUPABASE_PORT", 5432))
SUPABASE_DB       = os.getenv("SUPABASE_DB")
SUPABASE_USER     = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
# ─────────────────────────────────────────────

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS ads_appartments (
        hash_id              VARCHAR(64)     PRIMARY KEY,
        title                VARCHAR(500),
        img_url              VARCHAR(1000),
        link                 VARCHAR(1000),
        neighbourhood        VARCHAR(255),
        type_of_estate       VARCHAR(100),
        total_price_eur      DECIMAL(10, 2),
        price_m2_eur         DECIMAL(10, 2),
        price_m2_bgn         DECIMAL(10, 2),
        size_m2              DECIMAL(10, 2),
        nr_of_rooms          SMALLINT,
        description          TEXT,
        appartment_floor     SMALLINT,
        total_floors         SMALLINT,
        is_first_floor       BOOL,
        is_last_floor        BOOL,
        near_public_transport BOOL,
        furnished            BOOL,
        includes_parking     BOOL,
        new_building         BOOL,
        akt16                BOOL,
        energy_class         VARCHAR(255),
        potreblenie          VARCHAR(255),
        broker_commision     BOOL,
        additional_notes     VARCHAR(1000),
        extras               VARCHAR(500),
        
        -- status tracking
        is_active            BOOL         DEFAULT TRUE,
        status               VARCHAR(50)  DEFAULT 'active',  -- 'active', 'sold', 'expired', 'removed'
        deactivated_at       TIMESTAMP,
        loaded_at            TIMESTAMP    DEFAULT NOW()
    );
"""

def create_table():
    conn = None
    try:
        print("Connecting to Supabase PostgreSQL...")
        conn = psycopg2.connect(
            host=SUPABASE_HOST,
            port=SUPABASE_PORT,
            dbname=SUPABASE_DB,
            user=SUPABASE_USER,
            password=SUPABASE_PASSWORD,
            sslmode="require"           # Supabase requires SSL
        )
        cur = conn.cursor()

        print("Creating table 'ads_appartments'...")
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()

        print("✅ Table 'ads_appartments' created successfully (or already existed).")
        cur.close()

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        if conn:
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    create_table()