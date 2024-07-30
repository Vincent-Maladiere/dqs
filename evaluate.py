# %%
from pathlib import Path
from time import time
from tqdm import tqdm
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_ipcw
from pycox.datasets import support, metabric

from hazardous.data._seer import (
    load_seer,
    NUMERIC_COLUMN_NAMES,
    CATEGORICAL_COLUMN_NAMES,
)
from hazardous.survtrace._encoder import SurvFeatureEncoder
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.utils import make_time_grid
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
    brier_score_incidence,
    brier_score_incidence_oracle,
)
from hazardous.metrics._yana import CensoredNegativeLogLikelihoodSimple

from dqs.torch.distribution import DistributionLinear
from dqs.torch.loss import NegativeLogLikelihood

PATH_SCORES = Path("../benchmark/scores")
PATH_SEER = Path("../hazardous/data/seer_cancer_cardio_raw_data.txt")
WEIBULL_PARAMS = {
    "n_events": 1,
    "n_samples": 20_000,
    "censoring_relative_scale": 1.5,
    "complex_features": False,
    "independent_censoring": False,
}
SEEDS = range(5)
N_STEPS_TIME_GRID = 20
MODEL_NAME = "dqs"
N_EPOCH = 100
LEARNING_RATE = .01


class MLP(nn.Module):
    def __init__(self, input_len, n_output):
        super(MLP,self).__init__()

        num_neuron = 128
        self.fc1 = nn.Linear(input_len, num_neuron)
        self.fc2 = nn.Linear(num_neuron, n_output)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


def run_evaluation(dataset_name):
    all_scores = []

    for random_state in tqdm(SEEDS):
        scores = run_seed(dataset_name, random_state)
        all_scores.append(scores)
        
        path_dir = PATH_SCORES / "raw" / MODEL_NAME
        path_dir.mkdir(parents=True, exist_ok=True)
        path_raw_scores = path_dir / f"{dataset_name}.json"
        json.dump(all_scores, open(path_raw_scores, "w"))


def run_seed(dataset_name, random_state):
    bunch, dataset_params = get_dataset(dataset_name, random_state)
    model, fit_time = get_model(bunch)
    
    scores = evaluate(
        model,
        bunch,
        dataset_name,
        dataset_params=dataset_params,
        model_name=MODEL_NAME,
    )
    scores["fit_time"] = fit_time

    return scores


def evaluate(
    model, bunch, dataset_name, dataset_params, model_name,
):
    """Evaluate a model against its test set.
    """
    X_train, y_train = bunch["X_train"], bunch["y_train"]
    y_test = bunch["y_test"]

    n_events = np.unique(y_train["event"]).shape[0] - 1
    is_competing_risk = n_events > 1

    scores = {
        "is_competing_risk": is_competing_risk,
        "n_events": n_events,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_rows": X_train.shape[0],
        "n_cols": X_train.shape[1],
        "censoring_rate": (y_train["event"] == 0).mean(),
        **dataset_params,
    }

    y_pred, predict_time = get_y_pred(model, bunch)
    time_grid = np.asarray(bunch["time_grid"], dtype="float64")

    print(f"{time_grid=}")
    print(f"{y_pred.shape=}")
    print(f"{y_pred.mean(axis=1)}")

    scores["time_grid"] = time_grid.round(4).tolist()
    scores["y_pred"] = y_pred.round(4).tolist()
    scores["predict_time"] = round(predict_time, 2)

    event_specific_ibs, event_specific_brier_scores = [], []
    event_specific_c_index = []

    print("Computing Brier scores, IBS and C-index")

    event_id = 1

    # Brier score and IBS
    if dataset_name == "weibull":
        # Use oracle metrics with the synthetic dataset.
        ibs = integrated_brier_score_incidence_oracle(
            y_train,
            y_test,
            y_pred[event_id],
            time_grid,
            shape_censoring=bunch.shape_censoring.loc[y_test.index],
            scale_censoring=bunch.scale_censoring.loc[y_test.index],
            event_of_interest=event_id,
        )
        brier_scores = brier_score_incidence_oracle(
            y_train,
            y_test,
            y_pred[event_id],
            time_grid,
            shape_censoring=bunch.shape_censoring.loc[y_test.index],
            scale_censoring=bunch.scale_censoring.loc[y_test.index],
            event_of_interest=event_id,  
        )
    else:
        ibs = integrated_brier_score_incidence(
            y_train,
            y_test,
            y_pred[event_id],
            time_grid,
            event_of_interest=event_id,
        )
        brier_scores = brier_score_incidence(
            y_train,
            y_test,
            y_pred[event_id],
            time_grid,
            event_of_interest=event_id,
        )   
        
    event_specific_ibs.append({
        "event": event_id,
        "ibs": round(ibs, 4),
    })
    event_specific_brier_scores.append({
        "event": event_id,
        "time": list(time_grid.round(2)),
        "brier_score": list(brier_scores.round(4)),
    })

    # C-index
    y_train_binary = y_train.copy()
    y_test_binary = y_test.copy()

    y_train_binary["event"] = (y_train["event"] == event_id)
    y_test_binary["event"] = (y_test["event"] == event_id)

    truncation_quantiles = [0.25, 0.5, 0.75]
    taus = np.quantile(time_grid, truncation_quantiles)
    
    print(f"{taus=}")
    taus = tqdm(
        taus,
        desc=f"c-index at tau for event {event_id}",
        total=len(taus),
    )
    c_indices = []
    for tau in taus:
        tau_idx = np.searchsorted(time_grid, tau)
        y_pred_at_t = y_pred[event_id, :, tau_idx]
        ct_index, _, _, _, _ = concordance_index_ipcw(
            make_recarray(y_train_binary),
            make_recarray(y_test_binary),
            y_pred_at_t,
            tau=tau,
        )
        c_indices.append(round(ct_index, 4))

    event_specific_c_index.append({
        "event": event_id,
        "time_quantile": truncation_quantiles,
        "c_index": c_indices,
    })

    scores.update({
        "event_specific_ibs": event_specific_ibs,
        "event_specific_brier_scores": event_specific_brier_scores,
        "event_specific_c_index": event_specific_c_index,
    })

    # Yana loss
    print("Computing Censlog")

    censlog = CensoredNegativeLogLikelihoodSimple().loss(
        y_pred, y_test["duration"], y_test["event"], time_grid
    )
    scores["censlog"] = round(censlog, 4)        

    print(f"{event_specific_ibs=}")
    print(f"{event_specific_c_index}")
    print(f"{censlog=}")

    return scores


def get_dataset(dataset_name, random_state):
    bunch, dataset_params = load_dataset(dataset_name, random_state)
    X, y = bunch["X"], bunch["y"]
    print(f"{X.shape=}, {y.shape=}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y["event"],
        random_state=random_state,
    )

    enc = SurvFeatureEncoder(
        categorical_columns=bunch["categorical_columns"],
        numeric_columns=bunch["numeric_columns"],
    )
    X_train = enc.fit_transform(X_train)
    X_test = enc.transform(X_test)
    print(f"{X_train.shape=}, {X_test.shape=}")

    n_features = X_train.shape[1]
    n_events = len(set(np.unique(y_train["event"])) - {0})

    time_grid = torch.from_numpy(
        make_time_grid(
            y["duration"], n_steps=N_STEPS_TIME_GRID
        ).astype("float32")
    )
    # Make sure the duration always belong to the time_grid range.
    #time_grid[0] -= 1
    time_grid[-1] += 1
    
    X_train = torch.from_numpy(
        X_train.to_numpy().astype("float32")
    )
    event_train = torch.from_numpy(
        y_train["event"].to_numpy()
    )
    duration_train = torch.from_numpy(
        y_train["duration"].to_numpy().astype("float32")
    )

    duration_test = torch.from_numpy(
        y_test["duration"].to_numpy().astype("float32")
    )
    X_test = torch.from_numpy(
        X_test.to_numpy().astype("float32")
    )

    bunch.update({
        "X_train": X_train,
        "y_train": y_train,
        "event_train": event_train,
        "duration_train": duration_train,
        "X_test": X_test,
        "y_test": y_test,
        "duration_test": duration_test,
        "n_features": n_features,
        "n_events": n_events,
        "time_grid": time_grid,
    })
    
    return bunch, dataset_params


def load_dataset(dataset_name, random_state):

    dataset_params = {"random_state": random_state}

    if dataset_name == "seer":
        X, y = load_seer(
            input_path=PATH_SEER,
            survtrace_preprocessing=True,
            return_X_y=True,
        )
        X = X.dropna()
        y = y.iloc[X.index]
        bunch = {
            "X": X,
            "y": y,
            "numeric_columns": NUMERIC_COLUMN_NAMES,
            "categorical_columns": CATEGORICAL_COLUMN_NAMES,
        }

    elif dataset_name == "weibull":
        dataset_params.update(WEIBULL_PARAMS)
        bunch = make_synthetic_competing_weibull(**dataset_params)
        bunch.update({
            "numeric_columns": list(bunch.X.columns),
            "categorical_columns": [],
        })

    elif dataset_name == "support":
        df = support.read_df()
        categorical_features = ["x1", "x2", "x3", "x4", "x5", "x6"]
        numerical_features = ["x0", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
        bunch = _pycox_preprocessing(
            df, categorical_features, numerical_features, dataset_params
        ) 
    
    elif dataset_name == "metabric":
        df = metabric.read_df()
        categorical_features = ["x4", "x5", "x6", "x7"]
        numerical_features = ["x0", "x1", "x2", "x3", "x8"]
        bunch = _pycox_preprocessing(
            df, categorical_features, numerical_features, dataset_params
        ) 

    else:
        raise ValueError(dataset_name)

    return bunch, dataset_params


def _pycox_preprocessing(df, categorical_features, numerical_features, dataset_params):
    X = df.drop(columns=["duration", "event"])
    X[categorical_features] = X[categorical_features].astype("category")
    X[numerical_features] = X[numerical_features].astype("float64")
    y = df[["duration", "event"]]

    return dict(
        X=X,
        y=y,
        categorical_columns=categorical_features,
        numeric_columns=numerical_features,
    )
    

def get_model(bunch):

    tic = time()
    
    dist = DistributionLinear(bunch["time_grid"])
    loss_fn = NegativeLogLikelihood(dist, bunch["time_grid"])

    n_output = len(bunch["time_grid"])
    n_input = bunch["n_features"]
    model = MLP(n_input, n_output)

    X_train = bunch["X_train"]
    duration_train = bunch["duration_train"]
    event_train = bunch["event_train"]

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(N_EPOCH):
        pred = model(X_train)
        loss = loss_fn.loss(pred, duration_train, event_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print('epoch=%d, loss=%f' % (epoch,loss))

    fit_time = time() - tic

    return model, fit_time


def get_y_pred(model, bunch):
    tic = time()
    model.eval()
    
    X_test = bunch["X_test"]

    with torch.no_grad():
        y_pred_densities = model(X_test)
        y_pred = torch.cumsum(y_pred_densities, dim=1)
        # zeros = torch.zeros(y_pred.shape[0], 1)
        # y_pred = torch.cat([zeros, y_pred], axis=1)
        y_pred = y_pred.detach().numpy()

    y_pred = y_pred[None, :, :]
    y_surv = 1 - y_pred
    y_pred = np.concatenate([y_surv, y_pred], axis=0)

    predict_time = time() - tic

    return y_pred, predict_time


def make_recarray(y):
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )

# %%
if __name__ == "__main__":
    run_evaluation("weibull")

# %%
