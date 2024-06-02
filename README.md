# EEG Age Estimation - Semester Project
This repository host code that can train xgboost model for age prediction based on EEG data

You can find more about the data used in this project [here](https://www.kaggle.com/datasets/ayurgo/data-eeg-age-v1)

## Quick prediction
You can run a prediction by running the `quickPred.py` file.
It will use the given 40yo eeg csv (`EEG_40yo.csv`) data and pre trained `xgbModel.json` as input.

An example of the output:
```
Creating RawArray with float64 data, n_channels=36, n_times=307750
    Range : 0 ... 307749 =      0.000 ...  1230.996 secs
Ready.
Effective window size : 2.000 (s)
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.4s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.4s finished
Actual age    :  40
Predicted age :  43.53246
Mean Absolute Error (MAE) : 3.532459259033203
```
