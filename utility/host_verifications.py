import utility.verifications_mapper as mapper

def get_points_for_verification(verification_array):
    verification_array = verification_array.replace("[", "")
    verification_array = verification_array.replace("]", "")
    ver_splitted = verification_array.split(",")
    summary = 0
    for verification in ver_splitted:
        verification = verification.replace("\'", "")
        verification = verification.strip()
        summary = summary + mapper.get_points(verification)
        # print(verification, ':', mapper.get_points(verification))
    # print('sum: ', summary)
    # print('------------------------------------------')
    return summary