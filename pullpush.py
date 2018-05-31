from time import time
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from main import anal
import pandas as pd


###CONFIG###
SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-wi4ll-never-guess'
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:test1234@127.0.0.1:5432/gumdrop'
SQLALCHEMY_TRACK_MODIFICATIONS = False
MAIL_SERVER = os.environ.get('MAIL_SERVER')
MAIL_PORT = int(os.environ.get('MAIL_PORT') or 784)
MAIL_USE_TLS = 1  # os.environ.get('MAIL_USE_TLS') is not None  #always use TLS
MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
ADMINS = ['highratiotech@gmail.com']

db = create_engine(SQLALCHEMY_DATABASE_URI)
sess = sessionmaker(bind=db)
sess = sess()
# def read_process_repeat(db, loop_time, )

####build query to check for untested data
data = db.execute(
    """
    SELECT data, user_id, num_requested FROM project_index 
    WHERE (analysis_in_progress = FALSE AND 
    analysis_complete = FALSE) 
    ORDER BY timestamp_created ASC 
    LIMIT 1""")

# print("""SELECT email from user WHERE id = {} LIMIT 1""".format('asdff'))

try:
    for row in data:
        data = row[0]  # call master function on data here
        user_id = row[1]
        num_requested = row[2]

        ####query to change in_progress FLAG To TRUE

except:
    # data.close()
    print('no data available for analysis at {}'.format(datetime.datetime.now()))
    ###restart timer and recall master function here!

user_email = db.execute("""
    SELECT email from "user" 
    WHERE ID = {} 
    LIMIT 1 
    """.format(user_id))  # user is special case in Postgres, needs quotes!.





for row in user_email:
    user_email = row[0]

print(data)
in_df = pd.read_json(data, orient='split')
out_df = anal(in_df, num_requested)
out_JSON = out_df.to_json(double_precision=15, orient='split')
out_csv = out_df.to_csv()

###write query to add out_JSON to db + TIMESTAMP + COMPLETE FLAG + INPROGRESS FLAG
print(out_df)


# call email results here


# print(type(row[0]))


# class ProjectIndex(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     project_label = db.Column(db.String(256), unique=False)
#     num_analysis = db.Column(db.Integer, default=1)
#     num_requested = db.Column(db.Integer)
#     analysis_complete = db.Column(db.Boolean, default=False)
#     analysis_in_progress = db.Column(db.Boolean, default=False)
#     timestamp_created = db.Column(db.DateTime, index=True, default=datetime.utcnow)
#     timestamp_updated = db.Column(db.DateTime, index=True)
#     data = db.Column(db.JSON)
#     results = db.Column(db.JSON)
#
#     def __repr__(self):
#         return '<Project {}>'.format(self.project_label)
