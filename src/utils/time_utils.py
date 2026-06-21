def time_now_ist():
    from datetime import datetime
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')