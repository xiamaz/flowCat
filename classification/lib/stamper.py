'''Create lexically sortable timestamps that are usable as filepaths.'''
import datetime

def create_stamp():
    stamp = datetime.datetime.now()
    return stamp.strftime("%Y%m%d_%H%M")
