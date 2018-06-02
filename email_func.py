import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

from threading import Thread
import smtplib

MAIL_SERVER = 'email-smtp.us-west-2.amazonaws.com' #os.environ.get('MAIL_SERVER')
MAIL_PORT = int(587)
MAIL_USE_TLS = 1  # os.environ.get('MAIL_USE_TLS') is not None  #always use TLS
MAIL_USERNAME = 'AKIAI7B23SS5X4L2R2RA' # os.environ.get('MAIL_USERNAME')
MAIL_PASSWORD = 'AkxJVyoDGE5rdVHek09AaWpVyauTldBI2FqOiopiWB/r' #os.environ.get('MAIL_PASSWORD')
ADMINS = ['highratiotech@gmail.com']



def send_email(subject, sender, recipients, text_body, html_body, files = None):

    assert isinstance(recipients, list)
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = COMMASPACE.join(recipients)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text_body))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=str(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="{}"'.format(f)
        msg.attach(part)

    server = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)
    server.starttls()
    server.login(MAIL_USERNAME, MAIL_PASSWORD)
    server.sendmail(sender, recipients, msg.as_string())
    server.close()



#
# def send_async_email(app, msg):
#     with app.app_context():
#         mail.send(msg)