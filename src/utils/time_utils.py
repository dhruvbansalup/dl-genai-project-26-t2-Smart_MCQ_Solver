def time_now_ist(cleaned=False):
    from datetime import datetime
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    if cleaned:
        return datetime.now(ist).strftime('%Y-%m-%d_%H-%M-%S')
    return datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')