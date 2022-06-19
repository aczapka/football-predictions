Football Predictions
====================

This model is designed to make football predictions based on historical data using Poisson regression.

Obtaining historical data for training
------------------

First, the model needs a historical dataset which includes the following information about each match:

* Date of the match
* Home team
* Away team
* Home goals
* Away goals

In this implementation, data from the Spanish first division (_Liga de Primera División_) will be downloaded.

Model training
----------------------

The model used is a bivariate Poisson family with the following parameters:

* Each team's attack.
* Each team's defence.
* Home team's "playing at home" effect.
* Global dependence correction.

The training consist in maximizing the likelihood of the parametric model.
This has been implemented in terms of logarithms to make the calculation easier.


Model predictions
----------------------

Once the model has been trained, it can predict matches that occurred in the past or that will be played in the future.

The predict in the past feature can be used for model validation.
In order to avoid overfitting issues and to simulate real world performance,
a walk forward analysis method has been implemented.

Future improvements
----------------------

The idea is to analyze the output of this model in order to understand its strong and weak points.
These predictions can be used as an input to a more complex metamodel which further improves the accuracy.