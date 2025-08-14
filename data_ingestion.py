# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:43:33 2025
@author: Akach

This script connects to Oracle via Easy Connect Plus with timeout and retry parameters
to avoid ORA-12170 TNS:Connect timeout errors.
"""

import os
import pandas as pd
import cx_Oracle

# --- Configuration (REPLACE with your credentials/DSN) ---
DB_USER    = os.getenv("DB_USER",    "drg_depot_staging")
DB_PASS    = os.getenv("DB_PASS",    "drgdb")
DB_HOST    = os.getenv("DB_HOST",    "infoinst-02.rc.uab.edu")
DB_PORT    = os.getenv("DB_PORT",    "1521")
DB_SERVICE = os.getenv("DB_SERVICE", "BIODB.RC.UAB.EDU")  # Oracle service name
# ------------------------------------------------------

def get_connection():
    """
    Try connecting with Easy Connect Plus, specifying:
      - transport_connect_timeout: time (s) to establish TCP connection
      - retry_count: number of retry attempts on timeout
    Falls back to plain Easy Connect without params if initial attempt fails.
    """
    # Build Easy Connect Plus string
    ezplus_dsn = (
        f"{DB_HOST}:{DB_PORT}/{DB_SERVICE}"
        "?transport_connect_timeout=60"  # wait up to 60s for TCP connect :contentReference[oaicite:4]{index=4}
        "&retry_count=3"                # retry up to 3 times on timeout :contentReference[oaicite:5]{index=5}
    )
    try:
        return cx_Oracle.connect(DB_USER, DB_PASS, ezplus_dsn, encoding="UTF-8")
    except cx_Oracle.DatabaseError as e:
        print("Easy Connect Plus failed, trying basic Easy Connect…", e)
        # Fallback: host:port/service without extra params :contentReference[oaicite:6]{index=6}
        basic_dsn = f"{DB_HOST}:{DB_PORT}/{DB_SERVICE}"
        return cx_Oracle.connect(DB_USER, DB_PASS, basic_dsn, encoding="UTF-8")

def load_table(table_name: str) -> pd.DataFrame:
    """
    Load a full table from Oracle into a Pandas DataFrame.
    """
    conn = get_connection()
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    # Test loading key tables
    try:
        drugs_df = load_table("DRUGS")
        lincs_df = load_table("LINCS_L1000_SIGNATURES")
        cp_df    = load_table("CELL_PAINTING")
    except cx_Oracle.DatabaseError as err:
        print("Failed to load tables:", err)
        raise

    # Save locally for faster access
    os.makedirs("data", exist_ok=True)
    drugs_df.to_pickle("data/drugs.pkl")
    lincs_df.to_pickle("data/lincs.pkl")
    cp_df.to_pickle("data/cell_painting.pkl")

    # Final connection test
    conn = get_connection()
    print("Connected successfully. DSN used:", conn.dsn)
    conn.close()