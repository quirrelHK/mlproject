import sys, logging
import logger


def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    # the file in which error in raised
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message='Error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    
    def __init__(self,error_message,error_detail:sys):
        # inheriting error_message
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
if __name__=='__main__':

    try:
        a=10/0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e,sys)
