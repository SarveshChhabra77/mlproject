import sys
from src.logger import logging

## this all available in custom exception handling
def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()## this gives which line and which file an exception has occured
    file_name=exc_tb.tb_frame.f_code.co_filename
    line_number=exc_tb.tb_lineno
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,line_number,str(error))
    return error_message



class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

    