#!/usr/bin/python
import yaml
import mysql.connector
import os
from pathlib import Path
import pandas as pd
ROOT_DIR = Path(__file__).parent.parent.parent.parent # This is your Project Root

def reading_config():
    file = os.path.join(ROOT_DIR, "training_set", "configurations", 'mysql_config.yml')
    with open(file, 'r') as stream:
        try:
            s = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return s
config = reading_config()
mydb = mysql.connector.connect(
    host=config.get("host"),
    user=config.get("user"),
    password=config.get("passwd"),
    db=config.get("db")
)
mycursor = mydb.cursor()
def run_query(query, table, group_by_col = None, attach_column_names = True):


    mycursor.execute(query)
    sql_cols = '''
    SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = {db} AND TABLE_NAME = {table};
    '''.format(db= "'" + config.get("db") + "'", table="'" + table + "'")
    myresult = mycursor.fetchall()
    df = pd.DataFrame.from_records(myresult)
    if df.shape[0] == 1 and df.shape[1] == 1:
        return df
    elif group_by_col is not None:
        cols = [group_by_col, 'count']
        df.columns = cols[0:df.shape[1]]
        return df
    elif attach_column_names:
        mycursor.execute(sql_cols)
        cols = mycursor.fetchall()
        cols = [col[0] for col in cols]
        df.columns = cols
        return df
    else:
        return df

def run_query_col_names(query, agg_func_terms, group_by_terms):
    try:

        mycursor.execute(query)
        select_cols = group_by_terms[0].split(',')
        select_cols.append(agg_func_terms[0][0] + '_' +  agg_func_terms[0][1])
        myresult = mycursor.fetchall()
        df = pd.DataFrame.from_records(myresult)
        if df.shape[0] == 0:
            return df
        df.columns = select_cols
    except Exception as e:
        str_error = ("#HUNCH - ERROR - Unable to run query %s because %s" ) % (str(query), str(e))
        print (str_error)
    return df


# for x in myresult:
#   print(x)