import logging
import os
from datetime import datetime

LOG_FILE=f'{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log'
## creates a log file with currect date and time
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
#It creates a full file path to a log file inside a logs/ directory located in the current working directory.
os.makedirs(logs_path,exist_ok=True)
##This line creates a directory (or directories) specified by logs_path.
# exist_ok=True:
# Tells Python not to raise an error if the directory already exists.

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)
# It creates the full path to the log file by combining:
# logs_path: the path to the logs directory (e.g., /home/user/project/logs)
# LOG_FILE: the name of the log file (e.g., "app.log")




logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]-%(lineno)d-%(name)s-%(levelname)s-%(message)s ",
    level=logging.INFO,
)

