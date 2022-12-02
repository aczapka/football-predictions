Football Predictions
====================

Description
----------------------
This project tries to implement a successful football betting strategy.

### Motivation and key questions
As a summary, we are trying to answer this questions:

* How often do the home and away teams win?
* Are the odds of different bookmakers similar?
* How accurate are the predictions of the bookmakers?
* Have the predictions of the bookmakers changed over time?
* Are the odds of the bookmakers properly estimated?

For a visual summary of this preliminary ideas, please check the notebook: 
`medium_articles_notebooks/medium_can_we_guess_football_results_based_odds_bookmakers.ipynb` 
and the associated Medium article: 
https://medium.com/@aczapka/can-we-guess-football-results-based-on-the-odds-of-the-bookmakers-c91846813262

The last question implies if we can make a better prediction than the bookmaker and this is being further developed in 
this project.

### Data and modeling

In order to achieve that, it uses the following information from Spanish First Division:

* Historical data of match full time results.
* Historical data of bookmakers odds.
* A bivariate Poisson regression prediction model.
* Additional information.

All this data is finally combined in a neural network metamodel specifically tuned to forecast draws with high precision.

For more detailed information about the idea behind this project, please check `report.pdf`.

### Conclusions

As far now, the conclusion is that the predictions of the bookmakers, despite having some flaws, are accurate enough.
There is no easy way to beat them globally and build a successful betting strategy.

One of the lines of work that could be more promising is to focus exclusively on predicting draws with high precision. 
But this is not easy to achieve. 
For now perhaps we have achieved some partial successes whose consistency is yet to be validated.

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