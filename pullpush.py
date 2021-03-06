from time import sleep
from datetime import datetime
from sqlalchemy import create_engine
# import psycopg2
import os
from main import anal
import pandas as pd
import subprocess
from email_func import send_email

###CONFIG###
# nope...

while True:
    data, user_id, num_requested, id_, project_label, user_email_address = None, None, None, None, None, None  # reset variables so they don't carry over  # call master function on data here

    # try:
    #     SQLALCHEMY_DATABASE_URI = subprocess.Popen(r'heroku config:get DATABASE_URL -a gumdrop', shell=True,
    #                                                stdout=subprocess.PIPE)
    #     SQLALCHEMY_DATABASE_URI.wait()
    #     SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.stdout.read().decode('utf-8')
    #     SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI + r'?ssl=true'
    #
    #
    # except:
    #     sleep(180)
    #     continue

    try:
        data_conn = db.execute(
            """
            SELECT data, user_id, num_requested, id, project_label FROM project_index 
            WHERE (analysis_in_progress = FALSE AND 
            analysis_complete = FALSE) 
            ORDER BY timestamp_created ASC 
            LIMIT 1""")
    #
    # data_conn = db.execute(
    #     """
    #     SELECT data, user_id, num_requested, id FROM project_index
    #     """)
    except Exception as e:
        print('db conn error')
        print(e)
        sleep(5)
        continue

    try:
        for row in data_conn:
            data = row[0]  # call master function on data here
            user_id = row[1]
            num_requested = row[2]
            id_ = row[3]
            project_label = row[4]

        db.execute(
            """
            UPDATE project_index
            SET analysis_in_progress = TRUE
            WHERE id = {}
            """.format(id_))
        ####query to change in_progress FLAG To TRUE
        data_conn.close()


    except:  # Exception as e:
        data_conn.close()
        print('no data available for analysis at {}'.format(datetime.utcnow()))
        # print(e)
        sleep(wait_seconds)
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
    except Exception as e:
        print('ERROR in Analysis of id# {} failed at {} -- {}'.format(id_, datetime.utcnow(), e))
        db.execute("""
        UPDATE project_index
        SET error = True,
        timestamp_updated = '{}'
        WHERE id = {}
        """.format(datetime.utcnow(), id_))
        # will consider in_progress/not complete error signal in evaluation

        message_text = """ Sorry but there appears to be a formatting error with project:{}.  
        This error occured at {} UTC. 
        Please double check your inputs against the instructions provided in the template and resubmit.
        
        If that does not solve the issue please contact support@highratiotech.com
        """.format(project_label, datetime.utcnow())
        send_from = 'noreply@highratiotech.com'
        subject = 'Analysis of Project:{} Failed Formatting Error.'.format(project_label)
        # user_email_address = user_email_address

        try:
            send_email(user_email_address, send_from, subject,
                       message_text)
        except Exception as e:
            print(e)
            sleep(10)
            try:  # second attempt to send
                send_email(user_email_address, send_from, subject,
                           message_text)
            except:
                q = None  # does nothing...
        continue

    try:
        out_JSON = out_df.to_json(double_precision=15, orient='split')
        # out_df.to_json(r'C:\Users\georg\PycharmProjects\Bambi\test.json', orient='split')  #save local copy for eval
        out_csv = out_df.to_csv()
        # print(datetime.utcnow())
        # print(str(datetime.utcnow()))

        # print(out_JSON)
        # print(out_df)

        db.execute("""
        UPDATE project_index
        SET analysis_in_progress = FALSE,
        analysis_complete = TRUE,
        timestamp_updated = '{}',
        results = '{}'
        WHERE id = {}
        """.format(datetime.utcnow(), out_JSON, id_))

        message_text = """ 
        Please find a csv including your data with {} suggested experiments 
        Your data and results can be viewed at <PLACEHOLDER URL GOES HERE>""".format(num_requested)

        send_from = 'noreply@highratiotech.com'
        subject = 'Analysis of Project:{} completed.'.format(project_label)
        filename = '{}_{}.csv'.format(project_label, num_requested)
        # user_email_address = 'georLeonard@gmail.com'  # overwrite for testing purposes!!!

        try:
            send_email(user_email_address, send_from, subject,
                       message_text, file=out_csv, filename=filename)
        except Exception as e:
            print(e)
            sleep(10)
            try:  # second attempt to send
                send_email(user_email_address, send_from, subject,
                           message_text, file=out_csv, filename=filename)
            except Exception:
                sleep(30)
            try:
                send_email(user_email_address, send_from, subject,
                           message_text, file=out_csv, filename=filename)
            except Exception:
                pass  # does nothing...

        # send_email('georLeonard@gmail.com','georLeonard@gmail.com','asdf','test')
        print('Analysis of id# {} complete at {}'.format(id_, datetime.utcnow()))


    except Exception as e:
        print('ERROR in Analysis of id# {} failed at {} -- {}'.format(id_, datetime.utcnow(), e))
        db.execute("""
        UPDATE project_index
        SET error = True,
        timestamp_updated = '{}'
        WHERE id = {}
        """.format(datetime.utcnow(), id_))
    # db.execute("""
    # UPDATE project_index
    # SET analysis_in_progress = FALSE,
    # analysis_complete = TRUE,
    # timestamp_updated = '{}',
    # WHERE id = {}
    # """.format(datetime.utcnow(), id_))
    # will consider in_progress/not complete error signal in evaluation

    # print(type(out_csv))
    # print(type(out_JSON))
    # db.close()
    ###write query to add out_JSON to db + TIMESTAMP + COMPLETE FLAG + INPROGRESS FLAG
    # print(out_df)

    # break

# call email results here
