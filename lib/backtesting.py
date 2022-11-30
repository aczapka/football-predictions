from model import Model


class Backtesting(Model):
    """Class to iteratively train and predict multiple past seasons."""

    def __init__(self):
        super().__init__()
        self.seasons_complete = None
        self.seasons_half = None
        self.read_season_dates()

    def walk_forward(self, max_iterations=400):
        """Method to perform walk forward analysis.

         First, train with subset i of the dataset and predict for subset i+1.
         Then, proceed to train with the subset i+1 and predict for i+2. And so on.
         This way, it is simulated how the predictions would have turned out in the past without overfitting.
         It is necessary to specify which subset is considered (full seasons, half seasons...)
        """
        num_periods = len(self.seasons_complete)
        periods = self.seasons_complete

        for i_period in range(num_periods - 1):
            # Retrieve the date ranges of the subsets of the dataset
            start_date_train = periods[i_period + 1]['start_date']
            end_date_train = periods[i_period + 1]['end_date']
            train_period_name = periods[i_period + 1]['name']
            start_date_predict = periods[i_period]['start_date']
            end_date_predict = periods[i_period]['end_date']
            predict_period_name = periods[i_period]['name']

            print(f"\n\n------------------------------------------------\n\n\n"
                  f"Training with period {train_period_name} "
                  f"and predicting for period {predict_period_name}")

            # Train
            self.clip_complete_dataset(date_start=start_date_train, date_end=end_date_train)
            self.train(max_iterations=max_iterations)
            self.save_log_parameters(file_suffix='train_' + train_period_name)
            self.save_parameters(file_suffix='train_' + train_period_name)

            # Predict
            self.clip_complete_dataset(date_start=start_date_predict, date_end=end_date_predict)
            self.predict_past()
            self.save_prediction_past(file_suffix='train_' + train_period_name + '_predict_' + predict_period_name)

        return None

    def read_season_dates(self):
        """Date ranges of dataset subsets."""
        self.seasons_complete = [
            {
                'name': '2021-2022',
                'start_date': '2021-08-13',
                'end_date': '2022-05-22'
            },
            {
                'name': '2020-2021',
                'start_date': '2020-09-12',
                'end_date': '2021-05-23'
            },
            {
                'name': '2019-2020',
                'start_date': '2019-08-16',
                'end_date': '2020-07-19'
            },
            {
                'name': '2018-2019',
                'start_date': '2018-08-17',
                'end_date': '2019-05-19'
            },
            {
                'name': '2017-2018',
                'start_date': '2017-08-18',
                'end_date': '2018-05-20'
            },
            {
                'name': '2016-2017',
                'start_date': '2016-08-19',
                'end_date': '2017-05-21'
            },
            {
                'name': '2015-2016',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2014-2015',
                'start_date': '2014-08-23',
                'end_date': '2015-05-23'
            },
            {
                'name': '2013-2014',
                'start_date': '2013-08-17',
                'end_date': '2014-05-18'
            },
            {
                'name': '2012-2013',
                'start_date': '2012-08-18',
                'end_date': '2013-06-01'
            },
        ]

        self.seasons_half = [  # TODO: find the correct dates
            {
                'name': '2021-2022-2',
                'start_date': '2021-08-13',
                'end_date': '2022-05-22'
            },
            {
                'name': '2021-2022-1',
                'start_date': '2021-08-13',
                'end_date': '2022-05-22'
            },
            {
                'name': '2020-2021-2',
                'start_date': '2020-09-12',
                'end_date': '2021-05-23'
            },
            {
                'name': '2020-2021-1',
                'start_date': '2020-09-12',
                'end_date': '2021-05-23'
            },
            {
                'name': '2019-2020-2',
                'start_date': '2019-08-16',
                'end_date': '2020-07-19'
            },
            {
                'name': '2019-2020-1',
                'start_date': '2019-08-16',
                'end_date': '2020-07-19'
            },
            {
                'name': '2018-2019-2',
                'start_date': '2018-08-17',
                'end_date': '2019-05-19'
            },
            {
                'name': '2018-2019-1',
                'start_date': '2018-08-17',
                'end_date': '2019-05-19'
            },
            {
                'name': '2017-2018-2',
                'start_date': '2017-08-18',
                'end_date': '2018-05-20'
            },
            {
                'name': '2017-2018-1',
                'start_date': '2017-08-18',
                'end_date': '2018-05-20'
            },
            {
                'name': '2016-2017-2',
                'start_date': '2016-08-19',
                'end_date': '2017-05-21'
            },
            {
                'name': '2016-2017-1',
                'start_date': '2016-08-19',
                'end_date': '2017-05-21'
            },
            {
                'name': '2015-2016-2',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2015-2016-1',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2014-2015-1',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2014-2015-2',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2013-2014-1',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2013-2014-2',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2012-2013-1',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
            {
                'name': '2012-2013-2',
                'start_date': '2015-08-21',
                'end_date': '2016-05-15'
            },
        ]

        return None
