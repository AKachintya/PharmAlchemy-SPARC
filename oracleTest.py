# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:42:35 2025

@author: Akach
"""

import os
import cx_Oracle
from sqlalchemy import create_engine, text

# Set Oracle Instant Client path
cx_Oracle.init_oracle_client(lib_dir=r"C:\Users\Akach\OneDrive\Desktop\instantclient\instantclient_23_6")

# Add to environment PATH (if not already set)
os.environ["PATH"] = r"C:\Users\Akach\OneDrive\Desktop\instantclient\instantclient_23_6;" + os.environ["PATH"]

# Replace with your credentials
user = "drg_depot_staging"
password = "drgdb"
host = "infoinst-02.rc.uab.edu"
port = "1521"
service_name = "BIODB.RC.UAB.EDU"

DATABASE_URL = f"oracle+cx_oracle://{user}:{password}@{host}:{port}/?service_name={service_name}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as connection:
        print("SQLAlchemy connected successfully!")
        result = connection.execute(text("SELECT 'Hello from SQLAlchemy!' AS message FROM DUAL"))
        for row in result:
            print(row)
except Exception as e:
    print(f"Failed to connect using SQLAlchemy: {e}")
