from datetime import datetime

def date_to_year_week(date_str):
    """
    Convert a date string to year and week number.
    
    Parameters
    ----------
    date_str : str
        Date string in the format 'YYYY-MM-DD'
    
    Returns
    -------
    year : int
        The year of the given date
    week : int
        The ISO week number of the given date
    """
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Get the ISO calendar tuple (year, week, weekday)
    year, week, _ = date_obj.isocalendar()
    
    return year, week