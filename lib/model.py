from dataset import Dataset
import numpy as np
import json
from scipy.optimize import minimize
from scipy.stats import poisson


class Model(Dataset):
    """Class with the bivariate Poisson prediction model."""

    def __init__(self):
        super().__init__()
        self.optimization = None
        self.log_parameters = None
        self.prediction_past = None
        self.prediction_future = None

    def calculate_match_result_probability(self,
                                           log_attack_home,
                                           log_defence_home,
                                           log_attack_away,
                                           log_defence_away,
                                           log_home_effect,
                                           log_dependence,
                                           max_goals=10):
        """
        Given two teams and their (log)parameters, calculates the probability of each possible combination of goals.
        """

        # Calculate the probability of each result (of each possible combination of goals)
        probabilistic_prediction = np.full((max_goals + 1, max_goals + 1), np.nan)
        for goals_home in range(max_goals + 1):
            for goals_away in range(max_goals + 1):
                log_likelihood = self.log_likelihood(goals_home, goals_away, log_attack_home, log_defence_home,
                                                     log_attack_away, log_defence_away, log_home_effect, log_dependence)
                likelihood = np.exp(log_likelihood)
                probabilistic_prediction[goals_home, goals_away] = likelihood

        # Calculate the probability of victory, defeat and draw by adding probabilities
        probability_home = np.round(np.sum(np.tril(probabilistic_prediction, -1)), 3)
        probability_draw = np.round(np.sum(np.diag(probabilistic_prediction)), 3)
        probability_away = np.round(np.sum(np.triu(probabilistic_prediction, 1)), 3)

        return {
            'probability_home': probability_home,
            'probability_draw': probability_draw,
            'probability_away': probability_away
        }

    def log_likelihood(self, goals_home, goals_away, log_attack_home, log_defence_home, log_attack_away,
                       log_defence_away, log_home_effect, log_dependence):
        """Calculate the log-likelihood of a result in a match given some log-parameters.

         Calculates the log-likelihood of a match obtaining a result of (home_goals, away_goals) and
         taking into account that the teams that play are characterized by their home and away attack,
         home and away defense, by the factor of playing at home and by the dependence correction factor.
         This log-likelihood function is the loss function used for parameter optimization.
        """

        log_parameter_lambda = log_attack_home + log_defence_away + log_home_effect  # home
        log_parameter_mu = log_attack_away + log_defence_home  # away
        log_parameter_tau = self.calculate_log_tau(goals_home, goals_away, log_parameter_lambda, log_parameter_mu,
                                                   log_dependence)  # dependence (draw correction)

        log_likelihood = (log_parameter_tau + poisson.logpmf(goals_home, np.exp(log_parameter_lambda)) +
                          poisson.logpmf(goals_away, np.exp(log_parameter_mu))
                         )  # logarithm of a product is sum of logarithms

        return log_likelihood

    def total_minus_log_likelihood(self, log_parameters_array):
        """Computes the joint (minus)(log)likelihood of all matches in the dataset.

         Given some parameters, iterate the dataset to calculate the likelihood of the outcome of each match.
         The total likelihood is the sum of all the likelihoods because they are actually logarithms.
        """

        log_parameters = self.transform_log_parameters_from_array_to_dict(log_parameters_array=log_parameters_array)

        log_likelihoods = []
        for match in self.dataset.itertuples():
            log_likelihood = self.log_likelihood(goals_home=match.goals_home,
                                                 goals_away=match.goals_away,
                                                 log_attack_home=log_parameters['log_attack'][match.team_home],
                                                 log_defence_home=log_parameters['log_defence'][match.team_home],
                                                 log_attack_away=log_parameters['log_attack'][match.team_away],
                                                 log_defence_away=log_parameters['log_defence'][match.team_away],
                                                 log_home_effect=log_parameters['log_home_effect'][match.team_home],
                                                 log_dependence=log_parameters['log_dependence'])
            log_likelihoods.append(log_likelihood)

        total_log_likelihood = sum(log_likelihoods)  # since they are logarithms, add instead of multiplying
        total_log_likelihood = -1 * total_log_likelihood  # minimizing the function will be maximizing the likelihood

        return total_log_likelihood

    def train(self, max_iterations=300):
        """Train the model: optimize the parameters."""

        print("Training the model: optimizing log-parameters")

        # Initialize the log-parameters with coherent random numbers. Scipy minimize requires an array.
        log_parameters_initial_array = np.concatenate([
            np.random.uniform(0.8, 1.5, self.dataset_num_teams),  # attack
            np.random.uniform(-1.7, -0.5, self.dataset_num_teams),  # defence
            np.random.uniform(0, 0.6, self.dataset_num_teams),  # home effect
            np.array([-2.5]),  # dependence
        ])

        # Optimization
        optimization = minimize(
            self.total_minus_log_likelihood,  # function to minimize
            log_parameters_initial_array,
            method='trust-constr',
            options={
                'disp': True,
                'maxiter': max_iterations,
                'verbose': 3
            },
            # constraints=[{'type':'eq','fun':lambda x:sum(x[:20])-20}],
        )

        self.optimization = optimization
        self.log_parameters = self.transform_log_parameters_from_array_to_dict(optimization.x)
        return None

    def save_log_parameters(self, file_suffix):
        """Save log-parameters as JSON."""

        if self.log_parameters is None:
            print("There are no log-parameters to save")
            return None

        print("Saving log-parameters")
        with open('./output/parameters/log_parameters_' + file_suffix + '.json', 'w') as convert_file:
            convert_file.write(json.dumps(self.log_parameters))
        return None

    def save_parameters(self, file_suffix):
        """Transform log-parameters into parameters and save them as JSON."""

        if self.log_parameters is None:
            print("There are no parameters to save")
            return None

        print("Saving parameters")
        parameters = {}
        for log_parameter in self.log_parameters:
            parameter_name = log_parameter[4:]
            parameters[parameter_name] = {}

            if parameter_name != 'dependence':
                for team in self.log_parameters[log_parameter]:
                    log_value = self.log_parameters[log_parameter][team]
                    value = np.exp(log_value)
                    parameters[parameter_name][team] = value
            else:
                log_value = self.log_parameters[log_parameter]
                value = np.exp(log_value)
                parameters[parameter_name] = value

        with open('./output/parameters/parameters_' + file_suffix + '.json', 'w') as convert_file:
            convert_file.write(json.dumps(parameters))
        return None

    def predict_past(self):
        """Calculate prediction for dates in the past (the actual result is already known)."""

        print("Predicting matches in the past")

        probability_home = []
        probability_draw = []
        probability_away = []
        predictions = []
        observations = []
        is_true = []

        for match in self.dataset.itertuples():
            # Calculate categorical observation
            if match.goals_home > match.goals_away:
                observations.append('home')
            elif match.goals_home == match.goals_away:
                observations.append('draw')
            elif match.goals_away > match.goals_home:
                observations.append('away')
            else:
                observations.append('unknown')

            # Calculate probabilistic prediction
            try:
                probabilistic_prediction = self.calculate_match_result_probability(
                    log_attack_home=self.log_parameters['log_attack'][match.team_home],
                    log_defence_home=self.log_parameters['log_attack'][match.team_away],
                    log_attack_away=self.log_parameters['log_defence'][match.team_home],
                    log_defence_away=self.log_parameters['log_defence'][match.team_away],
                    log_home_effect=self.log_parameters['log_home_effect'][match.team_home],
                    log_dependence=self.log_parameters['log_dependence'],
                    max_goals=6)

                proba_home = probabilistic_prediction['probability_home']
                proba_draw = probabilistic_prediction['probability_draw']
                proba_away = probabilistic_prediction['probability_away']

                probability_home.append(proba_home)
                probability_draw.append(proba_draw)
                probability_away.append(proba_away)

                # Calculate categorical prediction
                if proba_home > proba_draw and proba_home > proba_away:
                    predictions.append('home')
                elif proba_draw > proba_home and proba_draw > proba_away:
                    predictions.append('draw')
                elif proba_away > proba_home and proba_away > proba_draw:
                    predictions.append('away')
                else:
                    predictions.append('unknown')

                # Check if prediction is right
                if predictions[-1] == observations[-1]:
                    is_true.append(1)
                else:
                    is_true.append(0)

            except KeyError:  # Team that has just entered/exited the division and cannot be predicted
                probability_home.append(np.nan)
                probability_draw.append(np.nan)
                probability_away.append(np.nan)
                predictions.append(np.nan)
                is_true.append(np.nan)

        # Save final result
        dataset_prediction = self.dataset.copy()
        dataset_prediction['probability_home'] = probability_home
        dataset_prediction['probability_draw'] = probability_draw
        dataset_prediction['probability_away'] = probability_away
        dataset_prediction['prediction'] = predictions
        dataset_prediction['observation'] = observations
        dataset_prediction['is_true'] = is_true
        self.prediction_past = dataset_prediction
        return None

    def predict_future(self, df_predictors):
        """Calculate prediction for dates in the futur (the actual result is unknown)."""

        print("Predicting matches in the future")

        probability_home = []
        probability_draw = []
        probability_away = []
        predictions = []
        is_true = []

        for match in df_predictors.itertuples():
            # Calculate probabilistic prediction
            try:
                probabilistic_prediction = self.calculate_match_result_probability(
                    log_attack_home=self.log_parameters['log_attack'][match.team_home],
                    log_defence_home=self.log_parameters['log_attack'][match.team_away],
                    log_attack_away=self.log_parameters['log_defence'][match.team_home],
                    log_defence_away=self.log_parameters['log_defence'][match.team_away],
                    log_home_effect=self.log_parameters['log_home_effect'][match.team_home],
                    log_dependence=self.log_parameters['log_dependence'],
                    max_goals=6)

                proba_home = probabilistic_prediction['probability_home']
                proba_draw = probabilistic_prediction['probability_draw']
                proba_away = probabilistic_prediction['probability_away']

                probability_home.append(proba_home)
                probability_draw.append(proba_draw)
                probability_away.append(proba_away)

                # Calculate categorical prediction
                if proba_home > proba_draw and proba_home > proba_away:
                    predictions.append('home')
                elif proba_draw > proba_home and proba_draw > proba_away:
                    predictions.append('draw')
                elif proba_away > proba_home and proba_away > proba_draw:
                    predictions.append('away')
                else:
                    predictions.append('unknown')

            except KeyError:  # Team that has just entered/exited the division and cannot be predicted
                probability_home.append(np.nan)
                probability_draw.append(np.nan)
                probability_away.append(np.nan)
                predictions.append(np.nan)
                is_true.append(np.nan)

        # Save final result
        dataset_prediction = df_predictors.copy()
        dataset_prediction['probability_home'] = probability_home
        dataset_prediction['probability_draw'] = probability_draw
        dataset_prediction['probability_away'] = probability_away
        dataset_prediction['prediction'] = predictions

        self.prediction_future = dataset_prediction
        return None

    def save_prediction_past(self, file_suffix):
        """Save the calculated past prediction in a CSV file."""

        if self.prediction_past is not None:
            print("Saving past prediction")
            self.prediction_past.to_csv('./output/predictions_past/predictions_' + file_suffix + '.csv',
                                        index=False,
                                        encoding='UTF-8',
                                        sep=';')
        else:
            print("There is no prediction to save")
        return None

    def save_prediction_future(self, file_suffix):
        """Save the calculated future prediction in a CSV file."""

        if self.prediction_future is not None:
            print("Saving future prediction")
            self.prediction_future.to_csv('./output/predictions_future/predictions' + file_suffix + '.csv',
                                          index=False,
                                          encoding='UTF-8',
                                          sep=';')
        else:
            print("There is no prediction to save")
        return None

    def transform_log_parameters_from_array_to_dict(self, log_parameters_array):
        """
        Transforms the array of parameters that scipy minimize needs to use into a dictionary of parameters that is
        easier to use and read.
        """
        log_parameters_attack = dict(zip(self.dataset_teams, log_parameters_array[:self.dataset_num_teams]))
        log_parameters_defence = dict(
            zip(self.dataset_teams, log_parameters_array[self.dataset_num_teams:(2 * self.dataset_num_teams)]))
        log_parameters_home_effect = dict(
            zip(self.dataset_teams, log_parameters_array[(2 * self.dataset_num_teams):(3 * self.dataset_num_teams)]))
        log_parameters_dependence = log_parameters_array[-1]

        log_parameters_dict = dict(log_attack=log_parameters_attack,
                                   log_defence=log_parameters_defence,
                                   log_home_effect=log_parameters_home_effect,
                                   log_dependence=log_parameters_dependence)
        return log_parameters_dict

    @staticmethod
    def calculate_log_tau(goals_home, goals_away, log_parameter_lambda, log_parameter_mu, log_dependence):
        """
        Calculate a factor that compensates the assumption of independence between Poisson distributions of each team
        when it is not incorrect.
        The rho correction factor, when positive, "thickens the diagonal" increasing the probability of a tie.
        It is used for calculating the tau factor which modifies the Poisson distributions.
        """

        # Coherence limits (depend on the team pair).
        # - Lower bound: Being logarithms, the actual value of rho is always greater than zero. No need to check.
        # - Upper bound: We leave a margin of 1% so as not to get too close (at the limit, the algorithm fails).
        log_upper_bound = np.min([-1 * log_parameter_mu, -1 * log_parameter_lambda])
        log_upper_bound = log_upper_bound - np.abs(log_upper_bound) * 0.01
        if log_dependence >= log_upper_bound:
            log_dependence = log_upper_bound

        # Calculate tau based on: rho, lambda, mu and goals.
        if goals_home == 0 and goals_away == 0:
            log_tau = np.log(1 + np.exp(log_parameter_lambda + log_parameter_mu + log_dependence))
        elif goals_home == 0 and goals_away == 1:
            log_tau = np.log(1 - np.exp(log_parameter_lambda + log_dependence))
        elif goals_home == 1 and goals_away == 0:
            log_tau = np.log(1 - np.exp(log_parameter_mu + log_dependence))
        elif goals_home == 1 and goals_away == 1:
            log_tau = np.log(1 + np.exp(log_dependence))
        else:
            log_tau = 0

        return log_tau
