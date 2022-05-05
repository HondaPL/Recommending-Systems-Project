# Load libraries ---------------------------------------------

from datetime import datetime, timedelta
from dateutil.easter import easter
from data_preprocessing.dataset_specification import DatasetSpecification

import pandas as pd
import numpy as np
# ------------------------------------------------------------


class DataPreprocessingToolkit(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self):
        dataset_specification = DatasetSpecification()

        self.sum_columns = dataset_specification.get_sum_columns()
        self.mean_columns = dataset_specification.get_mean_columns()
        self.mode_columns = dataset_specification.get_mode_columns()
        self.first_columns = dataset_specification.get_first_columns()

        self.nights_buckets = dataset_specification.get_nights_buckets()
        self.npeople_buckets = dataset_specification.get_npeople_buckets()
        self.room_segment_buckets = dataset_specification.get_room_segment_buckets()

        self.arrival_terms = dataset_specification.get_arrival_terms()

        self.item_features_columns = dataset_specification.get_items_df_feature_columns()

    # #########################
    # Entire datasets functions
    # #########################

    @staticmethod
    def filter_out_company_clients(df):
        """
        Filters out company clients is_company=0.

        :param pd.DataFrame df: DataFrame with at least the is_company column.
        :return: A DataFrame with filtered out corporate reservations.
        :rtype: pd.DataFrame
        """
        df = df.loc[df["is_company"] == 0]
        return df

    @staticmethod
    def filter_out_long_stays(df):
        """
        Leaves only stays with length_of_stay less or equal to 21.

        :param pd.DataFrame df: DataFrame with at least the length_of_stay column.
        :return: A DataFrame with reservations shorter than 22 days.
        :rtype: pd.DataFrame
        """
        df = df.loc[df["length_of_stay"] <= 21]
        return df

    @staticmethod
    def filter_out_low_prices(df):
        """
        Leaves only stays with accommodation price bigger than 50. Smaller prices are considered not reliable
        and likely a mistake of the hotel staff.

        :param pd.DataFrame df: DataFrame with at least the accommodation_price column.
        :return: A DataFrame with reservations with accommodation price bigger than 50.
        :rtype: pd.DataFrame
        """
        df = df.loc[df["accommodation_price"] > 50]
        return df

    @staticmethod
    def fix_date_to(df):
        """
        Fixes date_to to be the departure date in the df.

        :param pd.DataFrame df: DataFrame with at least the date_to column.
        :return: A DataFrame with fixed date_to.
        :rtype: pd.DataFrame
        """
        df.loc[:, "date_to"] = df["date_to"] + timedelta(days=1)
        return df

    @staticmethod
    def add_length_of_stay(df):
        """
        Adds length_of_stay column which is the difference between date_from and date_to (in days).

        :param pd.DataFrame df: DataFrame with at least the date_to and date_from columns.
        :return: A DataFrame with added length_of_stay column.
        :rtype: pd.DataFrame
        """
        df.loc[:, "length_of_stay"] = (df["date_to"] - df["date_from"]).dt.days
        return df

    @staticmethod
    def add_book_to_arrival(df):
        """
        Adds book_to_arrival column which is the difference between date_from and booking_date (in days).

        :param pd.DataFrame df: DataFrame with at least the date_from and booking_date columns.
        :return: A DataFrame with added book_to_arrival column.
        :rtype: pd.DataFrame
        """
        df.loc[:, "book_to_arrival"] = (
            df["date_from"] - df["booking_date"]).dt.days
        return df

    @staticmethod
    def add_nrooms(df):
        """
        Adds n_rooms columns with value 1 for further grouping.

        :param pd.DataFrame df: Any DataFrame.
        :return: A DataFrame with added n_rooms column.
        :rtype: pd.DataFrame
        """
        df['n_rooms'] = 1
        return df

    @staticmethod
    def add_weekend_stay(df):
        """
        Adds weekend_stay column with 'True'/'False' strings indicating if the interval date_from to date_to contains
        any weekend days (defined as Friday and Saturday).

        :param pd.DataFrame df: DataFrame with at least the date_from and date_to columns.
        :return: A DataFrame with added weekend_stay column.
        :rtype: pd.DataFrame
        """
        start_day = df["date_from"].dt.dayofweek
        end_day = df["date_to"].dt.dayofweek
        df.loc[:, "weekend_stay"] = (((start_day >= 4) & (start_day != 6)) | (end_day >= 5) | ((end_day < start_day) & (start_day != 6)) | (df["length_of_stay"] >= 6))
        df.loc[:, "weekend_stay"] = df["weekend_stay"].replace({True: 'True', False: 'False'})
        
        return df

    @staticmethod
    def add_night_price(df):
        """
        Adds night_price column with the price per one night per room - calculated as accommodation_price divided by
        length_of_stay and by n_rooms.

        :param pd.DataFrame df: DataFrame with at least the accommodation_price, length_of_stay, n_rooms columns.
        :return: A DataFrame with added night_price column.
        :rtype: pd.DataFrame
        """
        df.loc[:, "night_price"] = np.round(
            (df["accommodation_price"] / df["length_of_stay"] / df["n_rooms"]), 2)
        return df

    @staticmethod
    def clip_book_to_arrival(df):
        """
        Clips book_to_arrival to be greater or equal to zero.

        :param pd.DataFrame df: DataFrame with at least the book_to_arrival column.
        :return: A DataFrame with clipped book_to_arrival.
        :rtype: pd.DataFrame
        """
        df['book_to_arrival'] = np.maximum(df['book_to_arrival'], 0)
        return df

    @staticmethod
    def sum_npeople(df):
        """
        Sums n_people, n_children_1, n_children_2, n_children_3 and sets the result to the n_people column.

        :param pd.DataFrame df: DataFrame with at least n_people, n_children_1, n_children_2, n_children_3 columns.
        :return: A DataFrame with n_people column containing the number of all people in the reservation.
        :rtype: pd.DataFrame
        """
        df.loc[:, "n_people"] = df["n_people"] + df["n_children_1"] + df["n_children_2"] + df["n_children_3"]
        return df

    @staticmethod
    def leave_one_from_group_reservations(df):
        """
        Leaves only the first reservation from every group reservation.

        :param pd.DataFrame df: DataFrame with at least the group_id column.
        :return: A DataFrame with one reservation per group reservation.
        :rtype: pd.DataFrame
        """
        unique_group_rows = []

        df['group_id'] = df['group_id'].fillna(-1)

        group_ids = []
        for idx, row in df.iterrows():
            if row['group_id'] != -1:
                if row['group_id'] not in group_ids:
                    unique_group_rows.append(row)
                    group_ids.append(row['group_id'])
            else:
                unique_group_rows.append(row)

        df = pd.DataFrame(unique_group_rows, columns=df.columns)

        return df

    def aggregate_group_reservations(self, df):
        """
        Aggregates every group reservation into one reservation with aggregated data (for self.sum_columns a sum is
        taken, for self.mean_columns a mean, for self.mode_columns a mode, for self.first_columns the first value).

        :param pd.DataFrame df: DataFrame with at least the group_id column.
        :return: A DataFrame with aggregated group reservations.
        :rtype: pd.DataFrame
        """
        non_group_reservations = df.loc[df['group_id'] == "",
                                        self.sum_columns + self.mean_columns + self.mode_columns + self.first_columns]
        group_reservations = df.loc[df['group_id'] != ""]
        
        sum_group = group_reservations.loc[:, ["group_id"] + self.sum_columns].groupby("group_id").sum()
        mean_group = group_reservations.loc[:, ["group_id"] + self.mean_columns].groupby("group_id").mean()
        mode_group = group_reservations.loc[:, ["group_id"] + self.mode_columns].groupby("group_id").agg(lambda x: x.value_counts().index[0])
        first_group = group_reservations.loc[:, ["group_id"] + self.first_columns].groupby("group_id").first()
        
        group_reservations = sum_group
        group_reservations = group_reservations.merge(mean_group, on="group_id")
        group_reservations = group_reservations.merge(mode_group, on="group_id")
        group_reservations = group_reservations.merge(first_group, on="group_id")
        
        group_reservations = group_reservations.reset_index(drop=True)
        
        df = pd.concat([non_group_reservations, group_reservations])

        return df

    @staticmethod
    def leave_only_ota(df):
        df = df.loc[df.loc[:, 'Source'].apply(
            lambda x: "booking" in x.lower() or "expedia" in x.lower())]
        return df

    def map_dates_to_terms(self, df):
        """
        Maps arrival date (date_from) to term.

        :param pd.DataFrame df: DataFrame with at least the date_from column.
        :return: A DataFrame with the term column.
        :rtype: pd.DataFrame
        """
        df['date_from'] = df['date_from'].astype(str).apply(lambda x: x[:10])
        print()
        df['term'] = df['date_from'].apply(lambda x: self.map_date_to_term(x))
        return df

    def map_lengths_of_stay_to_nights_buckets(self, df):
        """
        Maps length_of_stay to nights buckets.

        :param pd.DataFrame df: DataFrame with at least the length_of_stay column.
        :return: A DataFrame with the length_of_stay_bucket column.
        :rtype: pd.DataFrame
        """
        df['length_of_stay_bucket'] = df['length_of_stay'].apply(
            lambda x: self.map_value_to_bucket(x, self.nights_buckets))
        return df

    def map_night_prices_to_room_segment_buckets(self, df):
        """
        Maps room_group_id to room_segment based on the average night price of the room group id.

        :param pd.DataFrame df: DataFrame with at least the room_group_id, night_price columns.
        :return: A DataFrame with the room_segment column.
        :rtype: pd.DataFrame
        """
        rooms_grouped = df.groupby('room_group_id')
        rooms_mean_prices = rooms_grouped['night_price'].mean().reset_index()
        rooms_mean_prices.columns = ['room_group_id', 'room_mean_price']
        df = df.merge(rooms_mean_prices, on='room_group_id', how='left')
        
        df['room_segment'] = df['room_mean_price'].fillna(value=0).apply(
            lambda value: self.map_value_to_bucket(value, self.room_segment_buckets))
        df = df.drop(columns=['room_mean_price'])
        
        return df

    def map_npeople_to_npeople_buckets(self, df):
        """
        Maps n_people to n_people buckets.

        :param pd.DataFrame df: DataFrame with at least the n_people column.
        :return: A DataFrame with the n_people_bucket column.
        :rtype: pd.DataFrame
        """
        df['n_people_bucket'] = df['n_people'].apply(
            lambda x: self.map_value_to_bucket(x, self.npeople_buckets))
        return df

    def map_item_to_item_id(self, df):
        """_summary_

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        df['item'] = df[self.item_features_columns].astype(
            str).agg(' '.join, axis=1)

        ids = df['item'].unique().tolist()
        mapping = {ids[i]: i for i in range(len(ids))}

        df['item_id'] = df['item'].apply(lambda x: mapping[x])

        return df

    @staticmethod
    def add_interaction_id(df):
        """_summary_

        Args:
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        df['interaction_id'] = range(df.shape[0])
        return df

    # ################
    # Column functions
    # ################

    @staticmethod
    def bundle_period(diff):
        """_summary_

        Args:
            diff (_type_): _description_

        Returns:
            _type_: _description_
        """
        diff = float(diff)
        if int(diff) < 0:
            return "<0"
        elif int(diff) <= 7:
            return diff
        elif 7 < int(diff) <= 14:
            return "<14"
        elif 14 < int(diff) <= 30:
            return "<30"
        elif 30 < int(diff) <= 60:
            return "<60"
        elif 60 < int(diff) <= 180:
            return "<180"
        elif int(diff) > 180:
            return ">180"

    @staticmethod
    def bundle_price(price):
        """_summary_

        Args:
            diff (_type_): _description_

        Returns:
            _type_: _description_
        """
        mod = 300.0
        return int((price + mod / 2) / mod) * mod

    @staticmethod
    def map_date_to_season(date):
        """_summary_

        Args:
            diff (_type_): _description_

        Returns:
            _type_: _description_
        """
        day = int(date[8:10])
        month = int(date[5:7])
        if (month == 12 and day >= 21) or (month == 1) or (month == 2) or (month == 3 and day <= 19):
            return "Winter"
        if (month == 3 and day >= 20) or (month == 4) or (month == 5) or (month == 6 and day <= 20):
            return "Spring"
        if (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day <= 22):
            return "Summer"
        if (month == 9 and day >= 23) or (month == 10) or (month == 11) or (month == 12 and day <= 20):
            return "Autumn"

    @staticmethod
    def map_value_to_bucket(value, buckets):
        """_summary_

        Args:
            diff (_type_): _description_

        Returns:
            _type_: _description_
        """
        if value == "":
            return str(buckets[0]).replace(", ", "-")
        for bucket in buckets:
            if bucket[0] <= value <= bucket[1]:
                return str(bucket).replace(", ", "-")

    def map_date_to_term(self, date):
        """_summary_

        Args:
            diff (_type_): _description_

        Returns:
            _type_: _description_
        """

        m = int(date[5:7])
        d = int(date[8:10])
        term = None

        for arrival_term in self.arrival_terms:
            if arrival_term == "Easter":
                year = int(date[:4])
                easter_date = easter(year)
                easter_start = easter_date + timedelta(days=-4)
                easter_end = easter_date + timedelta(days=1)
                esm = easter_start.month
                esd = easter_start.day
                eem = easter_end.month
                eed = easter_end.day
                if ((m > esm) or (m == esm and d >= esd)) and ((m < eem) or (m == eem and d <= eed)):
                    term = arrival_term
                    break

            elif arrival_term == "NewYear":
                sm = self.arrival_terms[arrival_term][0]['start']['m']
                sd = self.arrival_terms[arrival_term][0]['start']['d']
                em = self.arrival_terms[arrival_term][0]['end']['m']
                ed = self.arrival_terms[arrival_term][0]['end']['d']
                if ((m > sm) or (m == sm and d >= sd)) or ((m < em) or (m == em and d <= ed)):
                    term = arrival_term
                    break

            else:
                is_match = False

                for i in range(len(self.arrival_terms[arrival_term])):
                    sm = self.arrival_terms[arrival_term][i]['start']['m']
                    sd = self.arrival_terms[arrival_term][i]['start']['d']
                    em = self.arrival_terms[arrival_term][i]['end']['m']
                    ed = self.arrival_terms[arrival_term][i]['end']['d']
                    if ((m > sm) or (m == sm and d >= sd)) and ((m < em) or (m == em and d <= ed)):
                        term = arrival_term
                        is_match = True
                        break

                if is_match:
                    break

        return term

    def map_dates_list_to_terms(self, dates):
        """_summary_

        Args:
            diff (_type_): _description_

        Returns:
            _type_: _description_
        """

        terms = []
        for date in dates:
            term = self.map_date_to_term(date)
            terms.append(term)

        return terms

    @staticmethod
    def filter_out_historical_dates(date_list):
        """
        Filters out past dates from a list of dates.
        """
        future_dates = []

        for date in date_list:
            if date >= datetime.now():
                future_dates.append(date.strftime("%Y-%m-%d"))

        return future_dates
