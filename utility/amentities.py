import utility.aminitie_mapper as mapper

def get_points_for_amentities(amenities_json):
    amenities_json = amenities_json.replace("{", "")
    amenities_json = amenities_json.replace("}", "")
    am_splitted = amenities_json.split(",")
    summary = 0
    for amenitie in am_splitted:
        amenitie = amenitie.replace("\"", "")
        amenitie = amenitie.strip()
        summary = summary + mapper.get_points(amenitie)
        # print(amenitie, ':', mapper.get_points(amenitie))
    # print('sum: ', summary)
    # print('------------------------------------------')
    return summary