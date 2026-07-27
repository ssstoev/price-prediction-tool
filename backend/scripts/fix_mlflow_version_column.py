"""
Fix MLflow schema: ensure model_versions.version is INTEGER.

History:
  - MLflow < 2.9  created  model_versions.version as INTEGER.
  - MLflow 2.9-2.x expected VARCHAR — a previous run of this script converted it.
  - MLflow 3.x    reverted to INTEGER arithmetic in next_version(); VARCHAR causes:
      TypeError: can only concatenate str (not "int") to str

This script converts model_versions.version back to INTEGER (MLflow 3.x compatible).
model_version_tags.version must also remain INTEGER.

Run once:
  cd backend
  python scripts/fix_mlflow_version_column.py
"""

import os
import psycopg
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.getenv("MLFLOW_TRACKING_URI", "")
conn_str = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

print(f"Connecting to: {conn_str[:50]}...")

def col_type(cur, table, column):
    cur.execute("""
        SELECT data_type FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
    """, (table, column))
    row = cur.fetchone()
    return row[0].lower() if row else None

with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:

        # ── 1. model_versions.version must be INTEGER (MLflow 3.x) ──────────
        t = col_type(cur, "model_versions", "version")
        if t is None:
            print("Table 'model_versions' not found.")
        elif t in ("integer", "bigint", "smallint"):
            print(f"model_versions.version is already {t} — OK")
        else:
            print(f"model_versions.version is {t} — converting to INTEGER...")
            # Drop FK from model_version_tags first
            cur.execute("""
                SELECT conname FROM pg_constraint
                WHERE conrelid = 'model_version_tags'::regclass
                  AND contype = 'f'
                  AND confrelid = 'model_versions'::regclass
            """)
            for (fk,) in cur.fetchall():
                cur.execute(f'ALTER TABLE model_version_tags DROP CONSTRAINT IF EXISTS "{fk}"')
                print(f"  Dropped FK: {fk}")
            cur.execute("ALTER TABLE model_versions ALTER COLUMN version TYPE INTEGER USING version::INTEGER")
            print("  model_versions.version -> INTEGER ✓")

        # ── 2. model_version_tags.version must be INTEGER ────────────────────
        t = col_type(cur, "model_version_tags", "version")
        if t is None:
            print("Table 'model_version_tags' not found.")
        elif t in ("integer", "bigint", "smallint"):
            print(f"model_version_tags.version is already {t} — OK")
        else:
            print(f"model_version_tags.version is {t} — reverting to INTEGER...")
            cur.execute("ALTER TABLE model_version_tags ALTER COLUMN version TYPE INTEGER USING version::INTEGER")
            print("  model_version_tags.version -> INTEGER ✓")

        conn.commit()
        print("\nDone.")

