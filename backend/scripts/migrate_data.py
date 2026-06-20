'''Migrate ads_appartments'''

import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
CSV_PATH          = os.path.join(os.path.dirname(__file__), "../data/ads_appartments_cleaned.csv")
BATCH_SIZE        = 100                              # rows per insert batch
# ─────────────────────────────────────────────

TABLE_COLUMNS = [
    "hash_id", "title", "img_url", "link", "neighbourhood",
    "type_of_estate", "total_price_eur", "price_m2_eur", "price_m2_bgn",
    "size_m2", "nr_of_rooms", "description", "appartment_floor",
    "total_floors", "is_first_floor", "is_last_floor", "near_public_transport",
    "furnished", "includes_parking", "new_building", "akt16", "energy_class",
    "potreblenie", "broker_commision", "additional_notes", "extras",
    "is_active", "status", "deactivated_at", "loaded_at"
]

def build_upsert_sql():
    cols        = ", ".join(TABLE_COLUMNS)
    placeholders = ", ".join([f"%({c})s" for c in TABLE_COLUMNS])
    updates     = ", ".join(
        [f"{c} = EXCLUDED.{c}" for c in TABLE_COLUMNS if c != "hash_id"]
    )
    return f"""
        INSERT INTO ads_appartments ({cols})
        VALUES ({placeholders})
        ON CONFLICT (hash_id) DO UPDATE SET {updates};
    """

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns that exist in both CSV and our table
    df = df[[c for c in TABLE_COLUMNS if c in df.columns]].copy()

    # Fill missing columns with None
    for col in TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # total_floors, appartment_floor, nr_of_rooms are SMALLINT in the DB.
    # CSV values are float64 — convert to Python int so psycopg2 sends int2, not float8.
    # Null out inf and out-of-range values.
    def _to_smallint(x):
        if x is None:
            return None
        try:
            if pd.isna(x) or x in (float("inf"), float("-inf")):
                return None
        except (TypeError, ValueError):
            pass
        val = int(x)
        return val if -32768 <= val <= 32767 else None

    # Cast boolean columns: CSV stores 0/1 as float64 due to NaN coercion.
    # Convert to nullable Int8 first, then to Python bool so psycopg2 sends True/False.
    bool_cols = [
        "is_first_floor", "is_last_floor", "near_public_transport",
        "furnished", "includes_parking", "new_building", "akt16", "is_active",
        "broker_commision",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int8").astype(object)  # keeps None as None
            df[col] = df[col].apply(lambda x: bool(x) if x is not None and not pd.isna(x) else None)

    # Replace NaN with None for numeric columns (psycopg2 inserts NULL)
    df = df.where(pd.notna(df), None)

    # Replace any surviving float NaN in object/string columns with None
    def _none_if_nan(val):
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(_none_if_nan)

    return df[TABLE_COLUMNS]  # enforce column order

def migrate():
    print(f"📂 Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   {len(df)} rows found.")

    df = clean_df(df)
    rows = df.to_dict(orient="records")

    # Post-process SMALLINT columns on plain Python dicts — bypasses pandas dtype coercion.
    # Pandas to_dict() emits float for float64 columns; psycopg2 can't cast float→smallint.
    def _as_smallint(v):
        if v is None:
            return None
        try:
            if isinstance(v, float) and (v != v or abs(v) == float("inf")):
                return None  # NaN or inf
            i = int(v)
            return i if -32768 <= i <= 32767 else None
        except (ValueError, TypeError):
            return None

    for row in rows:
        for col in ("total_floors", "appartment_floor", "nr_of_rooms"):
            row[col] = _as_smallint(row.get(col))

        # Replace inf/-inf with None in all float columns — DECIMAL columns reject infinity.
        for col, val in row.items():
            if isinstance(val, float) and (val != val or abs(val) == float("inf")):
                row[col] = None

    upsert_sql = build_upsert_sql()

    conn = None
    try:
        print("🔌 Connecting to Neon...")
        conn = psycopg2.connect(NEON_DATABASE_URL)
        cur = conn.cursor()

        print(f"⬆️  Uploading in batches of {BATCH_SIZE}...")
        # Insert one row at a time to identify the exact failing row/column
        for i, row in enumerate(rows):
            try:
                cur.execute(upsert_sql, row)
            except psycopg2.errors.NumericValueOutOfRange as e:
                conn.rollback()
                print(f"\n❌ Row {i} failed: {e}")
                print("Numeric columns in this row:")
                for k, v in row.items():
                    if isinstance(v, (int, float)) or (v is not None and type(v).__module__ == "numpy"):
                        print(f"  {k:30s} = {v!r:20}  ({type(v).__name__})")
                raise
        conn.commit()
        conn.commit()

        print(f"✅ Done! {len(rows)} rows upserted into 'ads_appartments'.")
        cur.close()

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("🔒 Connection closed.")

if __name__ == "__main__":
    migrate()