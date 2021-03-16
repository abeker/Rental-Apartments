def get_points(verification):
    switcher = {
        'email': 1,
        'phone': 1,
        'facebook': 1,
        'jumio': 1,
        'offline_government_id': 1,
        'selfie': 1,
        'reviews': 1,
        'kba': 1,
        'work_email': 1,
        'google': 1,
        'manual_offline': 1,
        'manual_online': 1,
        'sesame_offline': 1,
        'sesame': 1,
        'weibo': 1,
        'sent_id': 1,
        'zhima_selfie': 1,
        'government_id': 1,
        'identity_manual': 1,
    }
    return switcher.get(verification, 0)