# Weather Forecasting with Recurrent Neural Networks

A reproducible pipeline framing multivariate weather forecasting as a sliding-window sequence modeling problem. This project evaluates four recurrent neural network architectures to predict hourly temperatures at 1-hour, 6-hour, and 12-hour horizons, assessing the empirical value of additive temporal attention compared to classical GRU and LSTM baselines.

## Overview
Using 14 meteorological variables from the Jena Climate 2009-2016 dataset, the pipeline downsamples 10-minute sensor readings to hourly resolution. Models observe a 48-hour historical window to output temperature forecasts. The study demonstrates that minimal GRU and Seq2Seq architectures outperform the standard LSTM and the proposed Attention-LSTM, highlighting a "reconstruction paradox" where uniform attention weights fail to provide additional predictive signal over standard recurrent state transitions.

## Project Structure
- `Weather Forecasting with Recurrent Neural Networks.ipynb`: Jupyter notebook containing data acquisition, chronological splitting, model definitions, and training loops.
- `Report_4.pdf`: Comprehensive analysis covering exploratory data analysis, architectural comparisons, attention weight distribution, and failure-case analysis.

## Key Phases
1. **Data Preprocessing & Split:** Resampled 420,224 rows of 10-minute data to 70,041 hourly means. Applied a strict chronological split (70% train, 15% validation, 15% test) to prevent future data leakage, fitting the standard scaler on the training partition only.
2. **Sequence Modeling:** Configured supervised sliding-window datasets (k=48 hours). Trained four models with a hidden dimension of 64 using MSE loss and early stopping: a 2-layer LSTM, a 2-layer GRU, a 1-layer GRU Seq2Seq, and an Attention-LSTM.
3. **Attention Analysis:** Extracted attention weights over the 48-hour history, revealing a nearly flat/uniform distribution (approx. 0.021) that functioned effectively as a soft-averaging component rather than isolating specific informative time-steps.
4. **Failure Analysis:** Evaluated maximum error windows (residuals > 3-4 °C). Results indicated that all models failed simultaneously on the same windows due to exogenous weather regime shifts (e.g., front passages) not captured within the 48-hour context.

## Key Technologies
- **Python:** Data processing and modeling.
- **PyTorch:** Custom implementation, training, and optimization of RNN, GRU, and Attention mechanisms.
- **Pandas:** Time-series resampling, gap-filling, and sentinel value clipping.
- **Matplotlib/Seaborn:** Visualization of meteorological correlations, diurnal/annual temperature cycles, and test-set error curves.

## Metrics & Findings
| Model | 1h MAE (°C) | 12h MAE (°C) | Mean RMSE (°C) | Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| **Seq2Seq** | 0.466 | 1.590 | 1.485 | 28,289 |
| **GRU** | 0.467 | 1.589 | 1.488 | 41,100 |
| **LSTM** | 0.600 | 1.626 | 1.539 | 54,540 |
| **Attention-LSTM** | 0.551 | 1.683 | 1.553 | 59,532 |

## Author
**Sepehr Barekati**
