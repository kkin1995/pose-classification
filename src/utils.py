import os
import time

def logger(message):
    """
    Logs a message with timestamp to log file

    Parameters:
    message: String Representation of message to be logged
    """
    logfile = "../logs/" + time.asctime().split()[2] + "-" + time.asctime().split()[1] + "-" + time.asctime().split()[-1] + ".log"
    if os.path.exists(logfile) == True:
        f = open(logfile, "a")
        f.write("[" + time.asctime().split()[3] + "] " + message + "\n")
        f.close()
    else:
        f = open(logfile, "w")
        f.write("[" + time.asctime().split()[3] + "] " + message + "\n")
        f.close()