from time import time
from time import sleep
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from main import anal
import pandas as pd
from email_func import send_email
import smtplib

###CONFIG###
SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-wi4ll-never-guess'
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:test1234@127.0.0.1:5432/gumdrop'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# test email:

"test,this,is,a,csv,\n 1,2,3,4,5\n"
send_email('test1','georLeonard@gmail.com', ['georLeonard@gmail.com',],'testing 1234 testing','nonsense')

db = create_engine(SQLALCHEMY_DATABASE_URI)

####build query to check for untested data

while False:
    data_conn = db.execute(
        """
        SELECT data, user_id, num_requested, id FROM project_index 
        WHERE (analysis_in_progress = FALSE AND 
        analysis_complete = FALSE) 
        ORDER BY timestamp_created ASC 
        LIMIT 1""")
    #
    # data_conn = db.execute(
    #     """
    #     SELECT data, user_id, num_requested, id FROM project_index
    #     """)

    try:
        for row in data_conn:
            data = row[0]  # call master function on data here
            user_id = row[1]
            num_requested = row[2]
            id_ = row[3]

        db.execute(
            """
            UPDATE project_index
            SET analysis_in_progress = TRUE
            WHERE id = {}
            """.format(id_))
            ####query to change in_progress FLAG To TRUE
        data_conn.close()


    except: # Exception as e:
        data_conn.close()
        print('no data available for analysis at {}'.format(datetime.utcnow()))
        # print(e)
        sleep(180)
        continue

    ###restart timer and recall master function here!

    user_email = db.execute("""
        SELECT email from "user" 
        WHERE ID = {} 
        LIMIT 1 
        """.format(user_id))  # user is special case in Postgres, needs quotes!.

    for row in user_email:
        user_email_address = row[0]
    user_email.close()


    # print(data)
    in_df = pd.read_json(data, orient='split')


    try:
        out_df = anal(in_df, num_requested)
    except:
        print('FORMATTING ERROR Analysis of id# {} failed at {}'.format(id_, datetime.utcnow()))
        db.execute("""
        UPDATE project_index
        SET analysis_in_progress = FALSE,
        analysis_complete = TRUE,
        timestamp_updated = '{}',
        WHERE id = {}
        """.format(datetime.utcnow(), id_))

    out_JSON = out_df.to_json(double_precision=15, orient='split')
    # out_df.to_json(r'C:\Users\georg\PycharmProjects\Bambi\test.json', orient='split')  #save local copy for eval
    out_csv = out_df.to_csv()
    # print(datetime.utcnow())
    # print(str(datetime.utcnow()))

    db.execute("""
    UPDATE project_index
    SET analysis_in_progress = FALSE,
    analysis_complete = TRUE,
    timestamp_updated = '{}',
    results = '{}'
    WHERE id = {}
    """.format(datetime.utcnow(), out_JSON, id_))

    print('Analysis of id# {} complete at {}'.format(id_,datetime.utcnow()))

    data, user_id, num_requested, id_ = None, None, None, None #reset variables so they don't carry over  # call master function on data here

    # print(type(out_csv))
    # print(type(out_JSON))
    # db.close()
    ###write query to add out_JSON to db + TIMESTAMP + COMPLETE FLAG + INPROGRESS FLAG
    # print(out_df)

    # break

# call email results here
