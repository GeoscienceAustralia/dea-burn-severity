import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Iterable


def is_date_iso(value: str) -> bool:
    """ checks to see if date string has desiered ios UTC format by trying to convert it to datetime object
    returns TRUE if string is in correct format
    returns FLASE if it is no :(
    
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return True
    except (ValueError, TypeError):
        return False


def is_date_DEA_format(value: str) -> bool:
    # test for YYYY-MM-DD format date:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def process_date(value: Any) -> str | None:
    """check the format for a date is as desiered 'yyyy-MM-ddTHH:mm:ss.sZ ', If it is not then format assuming it's 
    a valid date either as YYYY-MM-DD, YYYYMMDD or YYYYMMDDhhmmss and format as above (assume the time is noon if no time is given)

    otherwise return None. """
    if pd.isna(date):  # Handles NaT and None
        return None
    
    elif type(date) == pd._libs.tslibs.timestamps.Timestamp:
        date = date.to_pydatetime()
        return date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    elif isinstance(date, datetime):
        return date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    elif is_date_iso(date) == True:
        return date
        
    elif is_date_DEA_format(date) == True:
        return f'{date}T12:00:00.0Z'
        
    elif bool(re.match(r'^\d+$', date)) == True:
        if len(date) == 14:
            return f'{date[0:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}:{date[12:14]}.0Z'
        else:
            return f'{date[0:4]}-{date[4:6]}-{date[6:8]}T12:00:00.0Z'
    else:
        try:
            as_date = datetime.strptime(date, "%Y%m%d%H%M%S.%f")
            return as_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            
        except ValueError:
            return