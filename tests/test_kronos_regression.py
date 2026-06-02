import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from tqdm import tqdm

from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import _normalize_timestamps, calc_time_stamps
from webui import app as webui_app

TEST_DATA_ROOT = Path(__file__).parent / "data"
INPUT_DATA_PATH = TEST_DATA_ROOT / "regression_input.csv"

OUTPUT_DATA_DIR = TEST_DATA_ROOT
TEST_CTX_LEN = [512, 256]
PRED_LEN = 8
REL_TOLERANCE = 1e-5
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]

MSE_SAMPLE_SIZE = 4
MSE_CTX_LEN = [512, 256]
MSE_EXPECTED = [0.008979, 0.003741]
MSE_PRED_LEN = 30
MSE_TOLERANCE = 0.000001
MSE_FEATURE_NAMES = ["open", "high", "low", "close"]

MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MAX_CTX_LEN = 512
SEED = 123
DEVICE = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.mark.parametrize("context_len", TEST_CTX_LEN)
def test_kronos_predictor_regression(context_len):
    set_seed(SEED)

    expected_output_path = OUTPUT_DATA_DIR / f"regression_output_{context_len}.csv"
    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    expected_df = pd.read_csv(expected_output_path, parse_dates=["timestamps"])

    if df.shape[0] < context_len + len(expected_df):
        raise ValueError("Example data does not contain enough rows for the regression test.")

    context_df = df.iloc[:context_len].copy()
    context_features = context_df[FEATURE_NAMES].reset_index(drop=True)
    x_timestamp = context_df["timestamps"].reset_index(drop=True)
    future_timestamp = df["timestamps"].iloc[context_len:context_len + len(expected_df)].reset_index(drop=True)
    expected = expected_df[FEATURE_NAMES].values.astype(np.float32)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    with torch.no_grad():
        pred_df = predictor.predict(
            df=context_features,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamp,
            pred_len=expected.shape[0],
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
        )

    obtained = pred_df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    abs_diff = np.abs(obtained - expected)
    rel_diff = abs_diff / (np.abs(expected) + 1e-9)
    print(f"Abs diff: {np.max(abs_diff)}, Rel diff: {np.max(rel_diff)}")

    np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)


@pytest.mark.parametrize("context_len, expected_mse", zip(MSE_CTX_LEN, MSE_EXPECTED))
def test_kronos_predictor_mse(context_len, expected_mse):
    set_seed(SEED)

    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])
    if df.shape[0] <= context_len + MSE_PRED_LEN:
        raise ValueError("Example data does not contain enough rows for the random sample regression test.")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CTX_LEN)

    valid_region = df.iloc[context_len : df.shape[0] - MSE_PRED_LEN]
    if valid_region.shape[0] < MSE_SAMPLE_SIZE:
        raise ValueError("Not enough data points to draw the requested random samples.")

    sampled_rows = valid_region.sample(n=MSE_SAMPLE_SIZE, random_state=SEED).sort_index()

    mse_values = []
    sample_indices = sampled_rows.index.to_list()
    with torch.no_grad():
        for row_idx in tqdm(sample_indices):
            context_slice = df.iloc[row_idx - context_len : row_idx].copy()
            future_slice = df.iloc[row_idx : row_idx + MSE_PRED_LEN].copy()

            pred_df = predictor.predict(
                df=context_slice[FEATURE_NAMES].reset_index(drop=True),
                x_timestamp=context_slice["timestamps"].reset_index(drop=True),
                y_timestamp=future_slice["timestamps"].reset_index(drop=True),
                pred_len=MSE_PRED_LEN,
                T=1.0,
                top_k=1,
                top_p=1.0,
                verbose=False,
                sample_count=1,
            )

            obtained = pred_df[MSE_FEATURE_NAMES].to_numpy(dtype=np.float32)
            expected = future_slice[MSE_FEATURE_NAMES].to_numpy(dtype=np.float32)
            mse_values.append(float(np.mean((obtained - expected) ** 2)))

    assert len(mse_values) == MSE_SAMPLE_SIZE, f"Expected {MSE_SAMPLE_SIZE} MSE values, got {len(mse_values)}."

    mse = np.mean(mse_values).item()
    mse_diff = mse - expected_mse
    print(f"Average MSE: {mse} (Diff vs expected: {mse_diff:+})")

    assert abs(mse_diff) <= MSE_TOLERANCE, f"MSE {mse} differs from expected {expected_mse}"


class _DummyModule(torch.nn.Module):
    pass


class CapturingPredictor(KronosPredictor):
    def __init__(self):
        super().__init__(_DummyModule(), _DummyModule(), device="cpu", max_context=16)
        self.last_generate_call = None

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        self.last_generate_call = {
            "x_shape": tuple(x.shape),
            "x_stamp_shape": tuple(x_stamp.shape),
            "y_stamp_shape": tuple(y_stamp.shape),
            "pred_len": pred_len,
        }
        return np.zeros((x.shape[0], pred_len, x.shape[-1]), dtype=np.float32)


class RoutePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_p=0.9, sample_count=1):
        normalized_x = pd.Series(pd.to_datetime(x_timestamp)).reset_index(drop=True)
        normalized_y = pd.Series(pd.to_datetime(y_timestamp)).reset_index(drop=True)
        self.calls.append(
            {
                "df": df.copy(),
                "x_timestamp": normalized_x,
                "y_timestamp": normalized_y,
                "pred_len": pred_len,
            }
        )

        result = pd.DataFrame(
            {
                "open": np.linspace(10.0, 10.0 + pred_len - 1, pred_len),
                "high": np.linspace(10.5, 10.5 + pred_len - 1, pred_len),
                "low": np.linspace(9.5, 9.5 + pred_len - 1, pred_len),
                "close": np.linspace(10.2, 10.2 + pred_len - 1, pred_len),
                "volume": np.linspace(1000.0, 1000.0 + pred_len - 1, pred_len),
                "amount": np.linspace(2000.0, 2000.0 + pred_len - 1, pred_len),
            },
            index=pd.DatetimeIndex(normalized_y),
        )
        return result


@pytest.fixture
def sample_market_df():
    periods = 12
    timestamps = pd.date_range("2024-01-01 09:00:00", periods=periods, freq="h")
    base = np.arange(periods, dtype=np.float32)
    return pd.DataFrame(
        {
            "timestamps": timestamps,
            "open": 100 + base,
            "high": 101 + base,
            "low": 99 + base,
            "close": 100.5 + base,
            "volume": 1000 + base,
            "amount": 2000 + base,
        }
    )


def test_normalize_timestamps_and_calc_time_stamps_accept_datetime_index():
    timestamps = pd.date_range("2024-02-01 15:30:00", periods=3, freq="h")

    normalized = _normalize_timestamps(timestamps, "x_timestamp")
    time_df = calc_time_stamps(timestamps)

    assert isinstance(normalized, pd.Series)
    assert normalized.tolist() == list(timestamps)
    assert time_df.to_dict("records") == [
        {"minute": 30, "hour": 15, "weekday": 3, "day": 1, "month": 2},
        {"minute": 30, "hour": 16, "weekday": 3, "day": 1, "month": 2},
        {"minute": 30, "hour": 17, "weekday": 3, "day": 1, "month": 2},
    ]


def test_normalize_timestamps_rejects_invalid_values():
    with pytest.raises(ValueError, match="contains invalid timestamps"):
        _normalize_timestamps(["2024-01-01", "bad-timestamp"], "y_timestamp")


def test_kronos_predictor_predict_normalizes_timestamp_inputs():
    predictor = CapturingPredictor()
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )
    x_timestamp = pd.date_range("2024-03-01 09:00:00", periods=4, freq="h")
    y_timestamp = pd.DatetimeIndex(pd.date_range("2024-03-01 13:00:00", periods=2, freq="h"))

    pred_df = predictor.predict(
        df=df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=2,
        verbose=False,
        sample_count=1,
    )

    assert predictor.last_generate_call == {
        "x_shape": (1, 4, 6),
        "x_stamp_shape": (1, 4, 5),
        "y_stamp_shape": (1, 2, 5),
        "pred_len": 2,
    }
    assert list(pred_df.index) == list(y_timestamp)
    assert list(pred_df.columns) == ["open", "high", "low", "close", "volume", "amount"]


def test_kronos_predictor_predict_rejects_timestamp_length_mismatch():
    predictor = CapturingPredictor()
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )

    with pytest.raises(ValueError, match="x_timestamp length should match df length"):
        predictor.predict(
            df=df,
            x_timestamp=pd.date_range("2024-03-01 09:00:00", periods=3, freq="h"),
            y_timestamp=pd.date_range("2024-03-01 13:00:00", periods=2, freq="h"),
            pred_len=2,
            verbose=False,
            sample_count=1,
        )


def test_api_predict_uses_latest_comparable_window(monkeypatch, sample_market_df):
    route_predictor = RoutePredictor()
    monkeypatch.setattr(webui_app, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(webui_app, "predictor", route_predictor)
    monkeypatch.setattr(webui_app, "load_data_file", lambda _: (sample_market_df.copy(), None))
    monkeypatch.setattr(webui_app, "save_prediction_results", lambda **kwargs: None)

    client = webui_app.app.test_client()
    response = client.post(
        "/api/predict",
        json={"file_path": "/tmp/data.csv", "lookback": 4, "pred_len": 3, "temperature": 1.0, "top_p": 0.9, "sample_count": 1},
    )

    assert response.status_code == 200
    payload = response.get_json()
    expected_x = sample_market_df["timestamps"].iloc[5:9].reset_index(drop=True)
    expected_y = sample_market_df["timestamps"].iloc[9:12].reset_index(drop=True)

    assert route_predictor.calls[0]["x_timestamp"].equals(expected_x)
    assert route_predictor.calls[0]["y_timestamp"].equals(expected_y)
    assert payload["success"] is True
    assert payload["has_comparison"] is True
    assert payload["prediction_type"] == "Kronos model prediction (latest comparable window)"
    assert [item["timestamp"] for item in payload["prediction_results"]] == [ts.isoformat() for ts in expected_y]
    assert [item["timestamp"] for item in payload["actual_data"]] == [ts.isoformat() for ts in expected_y]


def test_api_predict_generates_future_timestamps_without_comparison(monkeypatch, sample_market_df):
    route_predictor = RoutePredictor()
    truncated_df = sample_market_df.iloc[:4].copy()
    monkeypatch.setattr(webui_app, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(webui_app, "predictor", route_predictor)
    monkeypatch.setattr(webui_app, "load_data_file", lambda _: (truncated_df, None))
    monkeypatch.setattr(webui_app, "save_prediction_results", lambda **kwargs: None)

    client = webui_app.app.test_client()
    response = client.post(
        "/api/predict",
        json={"file_path": "/tmp/data.csv", "lookback": 4, "pred_len": 2, "temperature": 1.0, "top_p": 0.9, "sample_count": 1},
    )

    assert response.status_code == 200
    payload = response.get_json()
    expected_future = pd.date_range(truncated_df["timestamps"].iloc[-1] + pd.Timedelta(hours=1), periods=2, freq="h")

    assert route_predictor.calls[0]["x_timestamp"].equals(truncated_df["timestamps"].reset_index(drop=True))
    assert route_predictor.calls[0]["y_timestamp"].equals(pd.Series(expected_future, name="timestamps"))
    assert payload["success"] is True
    assert payload["has_comparison"] is False
    assert payload["actual_data"] == []
    assert [item["timestamp"] for item in payload["prediction_results"]] == [ts.isoformat() for ts in expected_future]


def test_api_predict_respects_selected_start_window(monkeypatch, sample_market_df):
    route_predictor = RoutePredictor()
    monkeypatch.setattr(webui_app, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(webui_app, "predictor", route_predictor)
    monkeypatch.setattr(webui_app, "load_data_file", lambda _: (sample_market_df.copy(), None))
    monkeypatch.setattr(webui_app, "save_prediction_results", lambda **kwargs: None)

    start_timestamp = sample_market_df["timestamps"].iloc[2]
    client = webui_app.app.test_client()
    response = client.post(
        "/api/predict",
        json={
            "file_path": "/tmp/data.csv",
            "lookback": 4,
            "pred_len": 3,
            "start_date": start_timestamp.strftime("%Y-%m-%dT%H:%M"),
            "temperature": 1.0,
            "top_p": 0.9,
            "sample_count": 1,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    expected_x = sample_market_df["timestamps"].iloc[2:6].reset_index(drop=True)
    expected_y = sample_market_df["timestamps"].iloc[6:9].reset_index(drop=True)

    assert route_predictor.calls[0]["x_timestamp"].equals(expected_x)
    assert route_predictor.calls[0]["y_timestamp"].equals(expected_y)
    assert payload["success"] is True
    assert payload["has_comparison"] is True
    assert [item["timestamp"] for item in payload["prediction_results"]] == [ts.isoformat() for ts in expected_y]
    assert [item["timestamp"] for item in payload["actual_data"]] == [ts.isoformat() for ts in expected_y]
