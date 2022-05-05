# Load libraries ---------------------------------------------

from collections import defaultdict
import numpy as np

# ------------------------------------------------------------


class DatasetSpecification(object):

    def __init__(self):
        pass

    # ################
    # Original dataset functions
    # ################

    def get_sum_columns(self):
        return ["n_people", "n_children_1", "n_children_2", "n_children_3", "accommodation_price", "meal_price",
                "service_price", "paid", "n_rooms"]

    def get_mean_columns(self):
        return ['discount']

    def get_mode_columns(self):
        return ["room_id", "room_group_id", "date_from", "date_to", "booking_date", "rate_plan",
                "length_of_stay", "book_to_arrival", "weekend_stay"]

    def get_first_columns(self):
        return ["user_id", "client_id", "client_name", "email", "phone", "is_company"]

    def get_id_columns(self):
        return ["client_id", "client_name", "email", "phone"]

    # ################
    # Output dataset functions
    # ################

    def get_people_df_id_columns(self):
        return ['user_id']

    def get_people_df_feature_columns(self):
        return []

    def get_items_df_id_columns(self):
        return ['item_id']

    def get_items_df_feature_columns(self):
        return ['term', 'length_of_stay_bucket', 'rate_plan', 'room_segment', 'n_people_bucket', 'weekend_stay']

    def get_purchases_df_id_columns(self):
        return ['user_id', 'item_id']

    def get_purchases_df_feature_columns(self):
        return []

    # ################
    # Mapping functions
    # ################

    def get_nights_buckets(self):
        return [[0, 1], [2, 3], [4, 7], [8, np.inf]]

    def get_npeople_buckets(self):
        return [[1, 1], [2, 2], [3, 4], [5, np.inf]]

    def get_room_segment_buckets(self):
        return [[0, 160], [160, 260], [260, 360], [360, 500], [500, 900], [900, np.inf]]

    def get_book_to_arrival_buckets(self):
        return [[0, 0], [1, 2], [3, 4], [5, 7], [8, 14], [15, 30], [31, 60], [61, 90], [91, 180], [181, np.inf]]

    def get_arrival_terms(self):
        arrival_terms = {"Easter": [{"start": {"m": np.nan, "d": np.nan}, "end": {"m": np.nan, "d": np.nan}}],
                         # Treated with priority
                         "Christmas": [{"start": {"m": 12, "d": 22}, "end": {"m": 12, "d": 27}}],
                         "NewYear": [{"start": {"m": 12, "d": 28}, "end": {"m": 1, "d": 4}}],
                         "WinterVacation": [{"start": {"m": 1, "d": 5}, "end": {"m": 2, "d": 29}}],
                         "OffSeason": [
                             {"start": {"m": 3, "d": 1}, "end": {"m": 4, "d": 27}},
                             {"start": {"m": 5, "d": 6}, "end": {"m": 6, "d": 20}},
                             {"start": {"m": 9, "d": 26}, "end": {"m": 12, "d": 21}}],
                         "MayLongWeekend": [{"start": {"m": 4, "d": 28}, "end": {"m": 5, "d": 5}}],
                         "LowSeason": [
                             {"start": {"m": 6, "d": 21}, "end": {"m": 7, "d": 10}},
                             {"start": {"m": 8, "d": 23}, "end": {"m": 9, "d": 25}}],
                         "HighSeason": [{"start": {"m": 7, "d": 11}, "end": {"m": 8, "d": 22}}]}
        return arrival_terms
