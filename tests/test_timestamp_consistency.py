import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.kronos import _normalize_timestamps, calc_time_stamps


def make_timestamps(start="2024-01-01", periods=10, freq="1H"):
    return pd.date_range(start=start, periods=periods, freq=freq)


# ---------------------------------------------------------------------------
# _normalize_timestamps  tests
# ---------------------------------------------------------------------------

class TestNormalizeTimestamps:
    def test_accepts_series(self):
        ts = pd.Series(make_timestamps())
        result = _normalize_timestamps(ts, "x_timestamp")
        assert isinstance(result, pd.Series)
        assert result.dtype == "datetime64[ns]"
        assert len(result) == 10

    def test_accepts_datetimeindex(self):
        ts = make_timestamps()
        result = _normalize_timestamps(ts, "x_timestamp")
        assert isinstance(result, pd.Series)
        assert result.dtype == "datetime64[ns]"
        assert len(result) == 10

    def test_accepts_numpy_datetime64(self):
        ts = np.array(make_timestamps().to_pydatetime(), dtype="datetime64[ns]")
        result = _normalize_timestamps(ts, "x_timestamp")
        assert isinstance(result, pd.Series)
        assert result.dtype == "datetime64[ns]"
        assert len(result) == 10

    def test_accepts_list_of_strings(self):
        ts = ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"]
        result = _normalize_timestamps(ts, "x_timestamp")
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_accepts_list_of_timestamps(self):
        dti = make_timestamps(periods=5)
        ts = list(dti)
        result = _normalize_timestamps(ts, "x_timestamp")
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_resets_index(self):
        dti = make_timestamps(periods=5)
        result = _normalize_timestamps(dti, "x_timestamp")
        assert (result.index == pd.RangeIndex(start=0, stop=5)).all()

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _normalize_timestamps(pd.Series([], dtype="datetime64[ns]"), "x_timestamp")

    def test_raises_on_wrong_type_int(self):
        with pytest.raises(TypeError, match="must be a pandas Series"):
            _normalize_timestamps(42, "x_timestamp")

    def test_raises_on_wrong_type_float_list(self):
        with pytest.raises(TypeError, match="must be a pandas Series"):
            _normalize_timestamps([1.0, 2.0, 3.0], "x_timestamp")

    def test_raises_on_non_datetime_ndarray(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(TypeError, match="must be datetime-like"):
            _normalize_timestamps(arr, "x_timestamp")

    def test_coerces_string_series(self):
        s = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
        result = _normalize_timestamps(s, "x_timestamp")
        assert result.dtype == "datetime64[ns]"
        assert len(result) == 3


# ---------------------------------------------------------------------------
# calc_time_stamps  tests
# ---------------------------------------------------------------------------

class TestCalcTimeStamps:
    def test_output_shape(self):
        ts = make_timestamps(periods=20)
        result = calc_time_stamps(ts)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (20, 5)
        expected_cols = ["minute", "hour", "weekday", "day", "month"]
        assert list(result.columns) == expected_cols

    def test_output_values_single(self):
        ts = pd.Series(pd.to_datetime(["2024-03-15 14:30:00"]))
        result = calc_time_stamps(ts)
        assert result.loc[0, "minute"] == 30
        assert result.loc[0, "hour"] == 14
        assert result.loc[0, "weekday"] == 4
        assert result.loc[0, "day"] == 15
        assert result.loc[0, "month"] == 3

    def test_output_values_range(self):
        ts = pd.date_range("2024-01-01 00:00:00", periods=5, freq="3H")
        result = calc_time_stamps(ts)
        assert result["minute"].tolist() == [0, 0, 0, 0, 0]
        assert result["hour"].tolist() == [0, 3, 6, 9, 12]
        assert result["day"].tolist() == [1, 1, 1, 1, 1]
        assert result["month"].tolist() == [1, 1, 1, 1, 1]

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            calc_time_stamps(pd.Series([], dtype="datetime64[ns]"))

    def test_raises_on_non_datetime(self):
        with pytest.raises(TypeError):
            calc_time_stamps([1, 2, 3])

    def test_works_with_datetimeindex_input(self):
        dti = make_timestamps(periods=30)
        result = calc_time_stamps(dti)
        assert result.shape == (30, 5)

    def test_works_with_ndarray_input(self):
        arr = np.array(make_timestamps(periods=15).to_pydatetime(), dtype="datetime64[ns]")
        result = calc_time_stamps(arr)
        assert result.shape == (15, 5)


# ---------------------------------------------------------------------------
# Flask /api/predict route tests
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_csv(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=600, freq="1H")
    df = pd.DataFrame({
        "timestamps": timestamps,
        "open": np.random.randn(600).cumsum() + 100,
        "high": np.random.randn(600).cumsum() + 101,
        "low": np.random.randn(600).cumsum() + 99,
        "close": np.random.randn(600).cumsum() + 100,
        "volume": np.abs(np.random.randn(600) * 1000),
        "amount": np.abs(np.random.randn(600) * 10000),
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), df


@pytest.fixture
def app_client(monkeypatch):
    import webui.app as app_module

    app_module.MODEL_AVAILABLE = True

    class FakePredictor:
        def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0,
                    top_p=0.9, sample_count=1, verbose=True):
            pred_data = {
                "open": np.linspace(100, 110, pred_len),
                "high": np.linspace(101, 111, pred_len),
                "low": np.linspace(99, 109, pred_len),
                "close": np.linspace(100, 110, pred_len),
                "volume": np.ones(pred_len),
                "amount": np.ones(pred_len),
            }
            return pd.DataFrame(pred_data, index=y_timestamp)

    app_module.predictor = FakePredictor()

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client

    app_module.predictor = None
    app_module.MODEL_AVAILABLE = False


class TestPredictRouteTimestampConsistency:

    def test_prediction_timestamps_match_y_timestamp(self, app_client, temp_csv):
        csv_path, df = temp_csv
        lookback = 400
        pred_len = 120

        resp = app_client.post("/api/predict", json={
            "file_path": csv_path,
            "lookback": lookback,
            "pred_len": pred_len,
            "temperature": 1.0,
            "top_p": 0.9,
            "sample_count": 1,
        })
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True

        results = data["prediction_results"]
        assert len(results) == pred_len

        expected_timestamps = df["timestamps"].iloc[lookback:lookback + pred_len]
        for i, r in enumerate(results):
            ts_str = r["timestamp"]
            expected_ts = pd.Timestamp(expected_timestamps.iloc[i])
            assert pd.Timestamp(ts_str) == expected_ts, (
                f"Mismatch at index {i}: got {ts_str}, expected {expected_ts}"
            )

    def test_actual_data_timestamps_match_window(self, app_client, temp_csv):
        csv_path, df = temp_csv
        lookback = 400
        pred_len = 120

        resp = app_client.post("/api/predict", json={
            "file_path": csv_path,
            "lookback": lookback,
            "pred_len": pred_len,
            "temperature": 1.0,
            "top_p": 0.9,
            "sample_count": 1,
        })
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["has_comparison"] is True

        actual = data["actual_data"]
        assert len(actual) == pred_len

        expected_timestamps = df["timestamps"].iloc[lookback:lookback + pred_len]
        for i, a in enumerate(actual):
            ts_str = a["timestamp"]
            expected_ts = pd.Timestamp(expected_timestamps.iloc[i])
            assert pd.Timestamp(ts_str) == expected_ts, (
                f"Mismatch at index {i}: got {ts_str}, expected {expected_ts}"
            )

    def test_response_has_required_fields(self, app_client, temp_csv):
        csv_path, _ = temp_csv

        resp = app_client.post("/api/predict", json={
            "file_path": csv_path,
            "lookback": 400,
            "pred_len": 120,
        })
        assert resp.status_code == 200
        data = json.loads(resp.data)

        for field in ["success", "prediction_results", "actual_data",
                       "has_comparison", "chart", "message"]:
            assert field in data, f"Missing required field: {field}"

    def test_prediction_and_actual_timestamps_are_equal(self, app_client, temp_csv):
        csv_path, _ = temp_csv
        lookback = 400
        pred_len = 120

        resp = app_client.post("/api/predict", json={
            "file_path": csv_path,
            "lookback": lookback,
            "pred_len": pred_len,
        })
        assert resp.status_code == 200
        data = json.loads(resp.data)

        pred_timestamps = [r["timestamp"] for r in data["prediction_results"]]
        actual_timestamps = [a["timestamp"] for a in data["actual_data"]]

        assert pred_timestamps == actual_timestamps, (
            "Prediction timestamps and actual data timestamps must be identical"
        )


# ---------------------------------------------------------------------------
# KronosPredictor timestamp passthrough test
# ---------------------------------------------------------------------------

class TestPredictorTimestampPassthrough:

    def test_predictor_uses_y_timestamp_as_index(self):
        from model.kronos import KronosPredictor

        x_ts = pd.date_range("2024-01-01", periods=100, freq="1H")
        y_ts = pd.date_range("2024-01-05", periods=8, freq="1H")

        x_df = pd.DataFrame({
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 101,
            "low": np.random.randn(100).cumsum() + 99,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.abs(np.random.randn(100) * 1000),
            "amount": np.abs(np.random.randn(100) * 10000),
        })

        class FakeTokenizer:
            def __init__(self):
                self.s1_bits = 8
                self.codebook_dim = 16

            def encode(self, x, half=False):
                return (torch.zeros(1, 100, dtype=torch.long),
                        torch.zeros(1, 100, dtype=torch.long))

            def decode(self, x, half=False):
                return torch.randn(1, 108, 6)

            def to(self, device):
                return self

        class FakeModel:
            def __init__(self):
                self.d_model = 64
                self.n_heads = 4
                self.s1_bits = 8
                self.s2_bits = 8

            def decode_s1(self, s1_ids, s2_ids, stamp=None):
                B, T = s1_ids.shape
                logits = torch.randn(B, T, 256)
                context = torch.randn(B, T, 64)
                return logits, context

            def decode_s2(self, context, s1_ids, padding_mask=None):
                B, T, _ = context.shape
                return torch.randn(B, T, 256)

            def to(self, device):
                return self

        import torch
        predictor = KronosPredictor(FakeModel(), FakeTokenizer(), device="cpu", max_context=512)

        import types
        def mock_generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p,
                          sample_count, verbose):
            B = x.shape[0]
            return np.random.randn(B, pred_len, 6).astype(np.float32)

        predictor.generate = types.MethodType(mock_generate, predictor)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=8,
        )

        assert isinstance(pred_df, pd.DataFrame)
        assert pred_df.index.equals(pd.DatetimeIndex(y_ts)), (
            "predictor.predict() must use y_timestamp as result index"
        )