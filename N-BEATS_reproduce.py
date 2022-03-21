# -*- coding: utf-8 -*-

import os
from glob import glob
import numpy as np
import pandas as pd
import torch as t
from torch import optim

from collections import OrderedDict
from typing import Tuple

from datetime import datetime as dt
from dataclasses import dataclass

os.environ['KMP_DUPLICATE_LIB_OK']='True'

@dataclass
class M4Config:
    pathDatasetOrg: str
    pathDatasetDump: str
    pathResult: str
    
    seasonal_patterns: list
    horizons: list
    horizons_map: dict
    frequencies: list
    frequency_map: dict

    history_size: dict        
    iterations: dict
    layer_size: int
    layers: int
    stacks: int
    
    batch_size: int
    learning_rate: float
  
    # Ensemble parameters
    repeats: int
    lookbacks: list
    losses: list   


    def __init__(self):
        self.pathDatasetOrg = r"C:\Users\taeni\Documents\GitHub\N-BEATS\datasets\m4/"
        self.pathDatasetDump = r"C:\Users\taeni\Documents\Project train/dataset/"
        self.pathResult = r"C:\Users\taeni\Documents\Project train/result/"
    
        self.seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly',
                                  'Weekly', 'Daily', 'Hourly']
        self.horizons = [6, 8, 18, 13, 14, 48]
        self.horizons_map = {
            'Yearly': 6,
            'Quarterly': 8,
            'Monthly': 18,
            'Weekly': 13,
            'Daily': 14,
            'Hourly': 48
        }
        self.frequencies = [1, 4, 12, 1, 1, 24]
        self.frequency_map = {
            'Yearly': 1,
            'Quarterly': 4,
            'Monthly': 12,
            'Weekly': 1,
            'Daily': 1,
            'Hourly': 24
        }    

        self.history_size = {
            'Yearly': 1.5,
            'Quarterly': 1.5,
            'Monthly': 1.5,
            'Weekly': 10,
            'Daily': 10,
            'Hourly': 10
        }
    
        self.iterations = {
            'Yearly': 15000,
            'Quarterly': 15000,
            'Monthly': 15000,
            'Weekly': 5000,
            'Daily': 5000,
            'Hourly': 5000
        }

        # generic
        self.layers = 4
        self.stacks = 30
        self.layer_size = 512
        # trend
        self.trend_blocks = 3
        self.trend_layers = 4
        self.trend_layer_size = 256
        self.trend_degree_of_polynomial = 2
        # seasonality
        self.seasonality_blocks = 3
        self.seasonality_layers = 4
        self.seasonality_layer_size = 2048
        self.seasonality_num_of_harmonics = 1        
       
        # build 
        self.batch_size = 1024
        self.learning_rate = 0.001
        
        # Ensemble parameters
        self.repeats = 10
        self.lookbacks = [2, 3, 4, 5, 6, 7]
        self.losses = ['MASE', 'MAPE', 'SMAPE']
        

class M4Dataset:
    info = pd.DataFrame()
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    
    trainset: np.ndarray
    testset: np.ndarray
    
    def __init__(self, path_org, path_dump):
        self.pathDatasetOrg = path_org
        self.pathDatasetDump = path_dump
        
        info = pd.read_csv(path_org + "m4_info.csv") 
        self.info = info
        self.ids = info.M4id.values
        self.groups = info.SP.values
        self.frequencies = info.Frequency.values
        self.horizons = info.Horizon.values

        if not os.path.isfile(path_dump + "train.npz"):
            self.build_cache('*-train.csv', os.path.join(path_dump, 'train.npz'))
        else:
            print("Skip train dataset process... train.npz")
        
        if not os.path.isfile(path_dump + "test.npz"):
            self.build_cache('*-test.csv', os.path.join(path_dump, 'test.npz'))
        else:
            print("Skip test dataset process... test.npz")

        def build_cache(files: str, cache_path: str) -> None:
            ts_dict = OrderedDict(list(zip(info.M4id.values, [[]] * len(info.M4id.values))))
        
            for f in glob(os.path.join(path_org, files)):
                dataset = pd.read_csv(f)
                dataset.set_index(dataset.columns[0], inplace=True)
                for m4id, row in dataset.iterrows():
                    values = row.values
                    ts_dict[m4id] = values[~np.isnan(values)]
            np.array(list(ts_dict.values())).dump(cache_path)
        
        self.trainset = np.load(os.path.join(path_dump, 'train.npz'),
                                allow_pickle=True)        
        self.testset =  np.load(os.path.join(path_dump, 'test.npz'),
                                allow_pickle=True)        
        

###############################################################################
class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast
    
    
class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


class CommonUtil:
    @staticmethod
    def group_values(values: np.ndarray, groups: np.ndarray, group_name: str, ids: np.ndarray) -> (np.ndarray , np.ndarray):
        """
        Filter values array by group indices and clean it from NaNs.
    
        :param values: Values to filter.
        :param groups: Timeseries groups.
        :param group_name: Group name to filter by.
        :return: Filtered and cleaned timeseries - (ids , values).
        """
        ids = np.array([v for v in ids[groups == group_name]])
        values = np.array([v[~np.isnan(v)] for v in values[groups == group_name]])
        return ids, values    
    
    @staticmethod
    def do_sample(ts, insample_size, outsample_size, batch_size, window_sampling_limit):
        insample = np.zeros((batch_size, insample_size))
        insample_mask = np.zeros((batch_size, insample_size))
        outsample = np.zeros((batch_size, outsample_size))
        outsample_mask = np.zeros((batch_size, outsample_size))
        sampled_ts_indices = np.random.randint(len(ts), size=batch_size)
        for i, sampled_index in enumerate(sampled_ts_indices):
            sampled_timeseries = ts[sampled_index]
            cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - window_sampling_limit),
                                           high=len(sampled_timeseries),
                                           size=1)[0]
    
            insample_window = sampled_timeseries[max(0, cut_point - insample_size):cut_point]
            insample[i, -len(insample_window):] = insample_window
            insample_mask[i, -len(insample_window):] = 1.0
            outsample_window = sampled_timeseries[
                               cut_point:min(len(sampled_timeseries), cut_point + outsample_size)]
            outsample[i, :len(outsample_window)] = outsample_window
            outsample_mask[i, :len(outsample_window)] = 1.0
            
        return insample, insample_mask, outsample, outsample_mask

    @staticmethod
    def last_insample_window(ts, insample_size):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.
    
        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(ts), insample_size))
        insample_mask = np.zeros((len(ts), insample_size))
        for i, ts in enumerate(ts):
            ts_last_window = ts[-insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

# common util 2
def default_device() -> t.device:
    """
    PyTorch default device is GPU when available, CPU otherwise.

    :return: Default device.
    """
    return t.device('cuda' if t.cuda.is_available() else 'cpu')

def to_tensor(array: np.ndarray) -> t.Tensor:
    """
    Convert numpy array to tensor on default device.

    :param array: Numpy array to convert.
    :return: PyTorch tensor on default device.
    """
    return t.tensor(array, dtype=t.float32).to(default_device())

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def mape_loss(forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    weights = divide_no_nan(mask, target)
    return t.mean(t.abs((forecast - target) * weights))

def smape_2_loss(forecast, target, mask) -> t.float:
    """
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                      t.abs(forecast.data) + t.abs(target.data)) * mask)

def mase_loss(insample: t.Tensor, freq: int,
              forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param insample: Insample values. Shape: batch, time_i
    :param freq: Frequency value
    :param forecast: Forecast values. Shape: batch, time_o
    :param target: Target values. Shape: batch, time_o
    :param mask: 0/1 mask. Shape: batch, time_o
    :return: Loss value
    """
    masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
    masked_masep_inv = divide_no_nan(mask, masep[:, None])
    return t.mean(t.abs(target - forecast) * masked_masep_inv)

def __loss_fn(loss_name: str):
    def loss(x, freq, forecast, target, target_mask):
        if loss_name == 'MAPE':
            return mape_loss(forecast, target, target_mask)
        elif loss_name == 'MASE':
            return mase_loss(x, freq, forecast, target, target_mask)
        elif loss_name == 'SMAPE':
            return smape_2_loss(forecast, target, target_mask)
        else:
            raise Exception(f'Unknown loss function: {loss_name}')

    return loss





###############################################################################
# init
cutil = CommonUtil()

# M4 Experiments
# Models: seasonal -> lookback -> loss
def m4experiments(cfg: M4Config, dataset: M4Dataset, model_type='generic') -> None:
    trainset = dataset.trainset
    testset = dataset.testset
    for seasonal_pattern in cfg.seasonal_patterns:
        for j, lookback in enumerate(cfg.lookbacks):
            for k, loss in enumerate(cfg.losses):            
                history_size_in_horizons = cfg.history_size[seasonal_pattern]
                horizon = cfg.horizons_map[seasonal_pattern]
                input_size = lookback * horizon    
                timeseries_frequency=cfg.frequency_map[seasonal_pattern]
    
                # Data sampling
                train_ids, train_values = cutil.group_values(trainset, dataset.groups, seasonal_pattern, dataset.ids)
                test_ids, test_values = cutil.group_values(testset, dataset.groups, seasonal_pattern, dataset.ids)
                
                timeseries = [ts for ts in train_values]
                window_sampling_limit = int(history_size_in_horizons * horizon)
                batch_size = cfg.batch_size
                insample_size = input_size
                outsample_size = horizon
    
                if model_type == 'generic':
                    model = NBeats(t.nn.ModuleList([NBeatsBlock(input_size=insample_size,
                                                                theta_size=insample_size + outsample_size,
                                                                basis_function=GenericBasis(backcast_size=insample_size,
                                                                                            forecast_size=outsample_size),
                                                                layers=cfg.layers,
                                                                layer_size=cfg.layer_size)
                                                    for _ in range(cfg.stacks)]))
                elif model_type == 'interpretable':
                    trend_block = NBeatsBlock(input_size=insample_size,
                                              theta_size=2 * (cfg.degree_of_polynomial + 1),
                                              basis_function=TrendBasis(degree_of_polynomial=cfg.degree_of_polynomial,
                                                                        backcast_size=insample_size,
                                                                        forecast_size=outsample_size),
                                              layers=cfg.trend_layers,
                                              layer_size=cfg.trend_layer_size)
                    
                    seasonality_block = NBeatsBlock(input_size=insample_size,
                                                    theta_size=4 * int(
                                                        np.ceil(cfg.num_of_harmonics / 2 * outsample_size) - (cfg.num_of_harmonics - 1)),
                                                    basis_function=SeasonalityBasis(harmonics=cfg.num_of_harmonics,
                                                                                    backcast_size=insample_size,
                                                                                    forecast_size=outsample_size),
                                                    layers=cfg.seasonality_layers,
                                                    layer_size=cfg.seasonality_layer_size)
                    
                    model = NBeats(t.nn.ModuleList([trend_block for _ in range(cfg.trend_blocks)] +
                                                   [seasonality_block for _ in range(cfg.seasonality_blocks)]))
                else:
                    print(f"There is no {model_type} model-!!")
                    return
    
                model = model.to(default_device())
                learning_rate = cfg.learning_rate
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                training_loss_fn = __loss_fn(loss)
                iterations = cfg.iterations[seasonal_pattern]
                iterations = int(iterations / 1000) # Test
                lr_decay_step = iterations // 3            
                if lr_decay_step == 0:
                    lr_decay_step = 1
                    
                forecasts = []
                for i in range(1, iterations + 1):
                    model.train()
                    training_set = cutil.do_sample(timeseries,
                                                   insample_size,
                                                   outsample_size,
                                                   batch_size,
                                                   window_sampling_limit)
                
                    x, x_mask, y, y_mask = map(to_tensor, training_set)
                    optimizer.zero_grad()
                    forecast = model(x, x_mask)
                    training_loss = training_loss_fn(x, timeseries_frequency, forecast, y, y_mask)
                
                    if np.isnan(float(training_loss)):
                        break
                
                    training_loss.backward()
                    t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)
                
                    print(f'iter:{i}/{iterations} \t loss:{training_loss:.3f}')
    
                # Evaluate
                x, x_mask = map(to_tensor,
                                cutil.last_insample_window(timeseries, insample_size))
                model.eval()
                with t.no_grad():
                    forecasts.extend(model(x, x_mask).cpu().detach().numpy())            
                
                forecasts_df = pd.DataFrame(forecasts,
                                            columns=[f'V{i + 1}' for i in range(horizon)])
                forecasts_df.index = train_ids
                forecasts_df.index.name = 'id'
                f = cfg.pathResult
                f += dt.now().strftime("%Y-%m-%d %H%M%S")
                f += f'+{seasonal_pattern}+{lookback}+{loss}+'
                print(f'Dump start: {f}')
                forecasts_df.to_csv(f + 'forecast.csv')

if __name__ == '__main__':
    # Set Config
    m4cfg = M4Config()
    
    # M4 Dataset
    m4dataset = M4Dataset(path_org = m4cfg.pathDatasetOrg,
                        path_dump = m4cfg.pathDatasetDump)
    
    # M4 Experiments
    m4experiments(m4cfg, m4dataset, 'generic')
    







