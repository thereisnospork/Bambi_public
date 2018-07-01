import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import charset
from email.mime.nonmultipart import MIMENonMultipart
from email.charset import Charset, BASE64
from threading import Thread
import smtplib

MAIL_SERVER = 'email-smtp.us-west-2.amazonaws.com' #os.environ.get('MAIL_SERVER')
MAIL_PORT = int(587) # os.environ.get('MAIL_PORT')
MAIL_USE_TLS = 1  # os.environ.get('MAIL_USE_TLS') is not None  #always use TLS
MAIL_USERNAME = 'AKIAI7B23SS5X4L2R2RA' # os.environ.get('MAIL_USERNAME')
MAIL_PASSWORD = 'AkxJVyoDGE5rdVHek09AaWpVyauTldBI2FqOiopiWB/r' #os.environ.get('MAIL_PASSWORD')
ADMINS = ['highratiotech@gmail.com', 'georLeonard@gmail.com']

#
# file = {
#     'content': u'This,is,a,test,file',
#     'filename': 'testfile.csv'
# }


def send_email(send_to, send_from, subject, message_text, file= None, filename = None):
    # Create message container - multi/mixed MIME type for attachment
    full_email = MIMEMultipart('mixed')
    full_email['Subject'] = subject
    full_email['From'] = send_from

    if type(send_to) is list or type(send_to) is tuple:
        full_email['To'] = COMMASPACE.join(send_to)
    else:
        full_email['To'] = send_to

    body = MIMEMultipart('alternative')
    body.attach(MIMEText(message_text.encode('utf-8'),'plain', _charset='utf-8'))
    body.attach(MIMEText(("""<html>
                              <head></head>
                              <body>
                                <p>""" + message_text + """</p>
                              </body>
                            </html>
                            """).encode('utf-8'), 'html', _charset='utf-8'))
    full_email.attach(body)

    # create the attachment of the message in
    if file:
        attachment = MIMENonMultipart('text','csv', charset='utf-8')
        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        cs = Charset('utf-8')
        cs.body_encoding = BASE64
        attachment.set_payload(file.encode('utf-8'), charset=cs)
        full_email.attach(attachment)

    # Send the message via SMTP server.
    s = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)
    s.starttls()
    s.login(MAIL_USERNAME, MAIL_PASSWORD)
    # sendmail function takes 3 arguments: sender's address, recipient's address
    # and message to send - here it is sent as one string.
    s.sendmail(send_from, send_to, full_email.as_string())
    s.quit()


# test
# send_email(ADMINS[0],ADMINS[0],'test message-0 new eq', 'your results are enclosed this is message body', 'pretend, this, is a, csv,', 'test.csv')





# def send_email(subject, sender, recipients, text_body, html_body, files = None):
#
#     assert isinstance(recipients, list)
#     msg = MIMEMultipart()
#     msg['From'] = sender
#     msg['To'] = COMMASPACE.join(recipients)
#     msg['Date'] = formatdate(localtime=True)
#     msg['Subject'] = subject
#
#     msg.attach(MIMEText(text_body))
#
#     for f in files or []:
#         with open(f, "rb") as fil:
#             part = MIMEApplication(
#                 fil.read(),
#                 Name=str(f)
#             )
#         # After the file is closed
#         part['Content-Disposition'] = 'attachment; filename="{}"'.format(f)
#         msg.attach(part)
#
#     server = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)
#     server.starttls()
#     server.login(MAIL_USERNAME, MAIL_PASSWORD)
#     server.sendmail(sender, recipients, msg.as_string())
#     server.close()



#
# def send_async_email(app, msg):
#     with app.app_context():
#         mail.send(msg)