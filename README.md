Football Predictions
====================

Description
----------------------
This project tries to implement a successful football betting strategy.

In order to achieve that, it uses the following information from Spanish First Division:

* Historical data of match full time results.
* Historical data of bookmakers odds.
* A bivariate Poisson regression prediction model.
* Additional information.

All this data is finally combined in a neural network metamodel specifically tuned to forecast draws with high precision.

For more detailed information about the idea behind this project, please check `report.pdf`.

Dependencies and installation
----------------------
To use this code, just 
1. Download it.
2. Install Python along with the libraries listed in `requirements.txt`.
3. Mark the `lib` directory as sources root.

Common usage
----------------------
The first step is to execute `main.py`. This will take care of all the Poisson regression set up:

* The `dataset.py` module will download and save all the historical data needed.
* The `model.py` module will set up the model, train it and save the optimized parameters.
* The `backtesting.py` module will split the historical data and train the model iteratively in a realistic "walk forward analysis" fashion.

After this, all the predictions of the model will be stored in the `output/predictions_past` folder.

Now, it is time to play around with this new information and try to set a successful metamodel. This is performed with several Jupyter notebooks in the `metamodel` folder. They are designed in this order:

1. `01_download_dataset_all_columns.ipynb`: This code downloads the raw dataset.
1. `02_retrieve_bookmaker_predictions.ipynb`: In this notebook, to odds historical data of one of the bookmakers is
   extracted from the dataset.
1. `03_join_predictions.ipynb`: This script joins the odds of the bookmaker (which is their probabilistic prediction) with the Poisson regression probabilistic prediction generated with the code of this project.
1. `04_data_enrichment.ipynb`: The two probabilistic predictions are now being enriched with some additional data.
1. `05_betting_strategy.ipynb`: Now that all data is prepared, this notebook performs an analysis searching for a successful betting strategy. The conclusion is: trying to predict the draws.
1. `06_neural_network_metamodel.ipynb`: A neural network metamodel is set up and trained in order to predict the draws
   with the highest precision possible using all the data collected in the past steps. Then, it is evaluated to find if this betting strategy is profitable.

Future improvements
----------------------
This project is in an early stage of development and it does not even have tests. Bugs and errors are to be expected. Use it at your own risk and within the applicable legal limits. Please check `DISCLAIMER.md`.

For sure, the algorithm could be improved including more variables, tuning the model and the metamodel, optimizing, hyperparameters, trying to use other dataset, etc.

The validation of the results is also poor right now.

Author, license and acknowledgements 
----------------------
Author: Alberto Czapka
https://www.linkedin.com/in/alberto-czapka/

This project is licensed under the MIT license, please check `LICENSE.md`.

Theoretical background:
Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. Journal of the Royal Statistical Society. Series C (Applied Statistics), 46(2), 265–280. http://www.jstor.org/stable/2986290