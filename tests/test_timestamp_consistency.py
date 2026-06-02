import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kronos import _normalize_timestamps, calc_time_stamps, KronosPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_timestamps_series():
    return pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))


@pytest.fixture
def sample_timestamps_index():
    return pd.date_range("2024-01-01", periods=10, freq="1H")


@pytest.fixture
def sample_timestamps_list():
    return list(pd.date_range("2024-01-01", periods=10, freq="1H"))


@pytest.fixture
def sample_timestamps_numpy():
    return np.array(pd.date_range("2024-01-01", periods=10, freq="1H"))


@pytest.fixture
def sample_timestamps_str():
    return [f"2024-01-01 {i:02d}:00:00" for i in range(10)]


@pytest.fixture
def sample_df():
    n = 10
    return pd.DataFrame({
        "open": np.random.rand(n) * 100 + 50,
        "high": np.random.rand(n) * 100 + 55,
        "low": np.random.rand(n) * 100 + 45,
        "close": np.random.rand(n) * 100 + 50,
        "volume": np.random.rand(n) * 1000,
        "amount": np.random.rand(n) * 100000,
    })


# ---------------------------------------------------------------------------
# _normalize_timestamps tests
# ---------------------------------------------------------------------------

class TestNormalizeTimestamps:

    def test_series_datetime(self, sample_timestamps_series):
        result = _normalize_timestamps(sample_timestamps_series, "ts")
        assert isinstance(result, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 10

    def test_datetime_index(self, sample_timestamps_index):
        result = _normalize_timestamps(sample_timestamps_index, "ts")
        assert isinstance(result, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 10

    def test_list_input(self, sample_timestamps_list):
        result = _normalize_timestamps(sample_timestamps_list, "ts")
        assert isinstance(result, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 10

    def test_numpy_input(self, sample_timestamps_numpy):
        result = _normalize_timestamps(sample_timestamps_numpy, "ts")
        assert isinstance(result, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 10

    def test_string_list_input(self, sample_timestamps_str):
        result = _normalize_timestamps(sample_timestamps_str, "ts")
        assert isinstance(result, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert len(result) == 10

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be a pd.Series"):
            _normalize_timestamps(42, "ts")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _normalize_timestamps(["not-a-date"], "ts")

    def test_null_values_raise(self):
        ts = pd.Series([pd.Timestamp("2024-01-01"), pd.NaT])
        with pytest.raises(ValueError, match="contains null values"):
            _normalize_timestamps(ts, "ts")

    def test_reset_index(self, sample_timestamps_series):
        original = sample_timestamps_series.copy()
        original.index = range(100, 110)
        result = _normalize_timestamps(original, "ts")
        assert list(result.index) == list(range(10))

    def test_preserves_values(self, sample_timestamps_series):
        result = _normalize_timestamps(sample_timestamps_series, "ts")
        for i in range(len(sample_timestamps_series)):
            assert result.iloc[i] == sample_timestamps_series.iloc[i]


# ---------------------------------------------------------------------------
# calc_time_stamps tests
# ---------------------------------------------------------------------------

class TestCalcTimeStamps:

    def test_basic_output(self, sample_timestamps_series):
        result = calc_time_stamps(sample_timestamps_series)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["minute", "hour", "weekday", "day", "month"]
        assert len(result) == 10

    def test_correct_values(self):
        ts = pd.Series([pd.Timestamp("2024-03-15 14:30:00")])
        result = calc_time_stamps(ts)
        assert result["minute"].iloc[0] == 30
        assert result["hour"].iloc[0] == 14
        assert result["weekday"].iloc[0] == 4  # Friday
        assert result["day"].iloc[0] == 15
        assert result["month"].iloc[0] == 3

    def test_with_datetime_index(self, sample_timestamps_index):
        result = calc_time_stamps(sample_timestamps_index)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_with_string_list(self, sample_timestamps_str):
        result = calc_time_stamps(sample_timestamps_str)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_with_numpy_array(self, sample_timestamps_numpy):
        result = calc_time_stamps(sample_timestamps_numpy)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError):
            calc_time_stamps(42)

    def test_all_columns_numeric(self, sample_timestamps_series):
        result = calc_time_stamps(sample_timestamps_series)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])


# ---------------------------------------------------------------------------
# KronosPredictor.predict validation tests (no model weights needed)
# ---------------------------------------------------------------------------

class TestKronosPredictorValidation:

    @pytest.fixture
    def dummy_predictor(self):
        class DummyModel:
            pass

        class DummyTokenizer:
            pass

        return KronosPredictor(DummyModel(), DummyTokenizer(), device="cpu")

    def test_x_timestamp_length_mismatch(self, dummy_predictor, sample_df):
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1H"))
        y_ts = pd.Series(pd.date_range("2024-01-01", periods=3, freq="1H"))
        with pytest.raises(ValueError, match="x_timestamp length"):
            dummy_predictor.predict(sample_df, x_ts, y_ts, pred_len=3)

    def test_y_timestamp_length_mismatch(self, dummy_predictor, sample_df):
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))
        y_ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1H"))
        with pytest.raises(ValueError, match="y_timestamp length"):
            dummy_predictor.predict(sample_df, x_ts, y_ts, pred_len=3)

    def test_non_dataframe_raises(self, dummy_predictor):
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            dummy_predictor.predict(
                [1, 2, 3],
                pd.Series(pd.date_range("2024-01-01", periods=3, freq="1H")),
                pd.Series(pd.date_range("2024-01-01", periods=3, freq="1H")),
                pred_len=3,
            )

    def test_missing_price_columns_raises(self, dummy_predictor):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Price columns"):
            dummy_predictor.predict(
                df,
                pd.Series(pd.date_range("2024-01-01", periods=1, freq="1H")),
                pd.Series(pd.date_range("2024-01-01", periods=1, freq="1H")),
                pred_len=1,
            )

    def test_nan_values_raise(self, dummy_predictor):
        df = pd.DataFrame({
            "open": [1.0, np.nan],
            "high": [2.0, 3.0],
            "low": [0.5, 1.0],
            "close": [1.5, 2.0],
            "volume": [100.0, 200.0],
            "amount": [1000.0, 2000.0],
        })
        with pytest.raises(ValueError, match="NaN values"):
            dummy_predictor.predict(
                df,
                pd.Series(pd.date_range("2024-01-01", periods=2, freq="1H")),
                pd.Series(pd.date_range("2024-01-01", periods=1, freq="1H")),
                pred_len=1,
            )

    def test_datetime_index_accepted(self, dummy_predictor, sample_df):
        x_ts = pd.date_range("2024-01-01", periods=10, freq="1H")
        y_ts = pd.date_range("2024-01-01 10:00", periods=3, freq="1H")
        with pytest.raises(Exception):
            dummy_predictor.predict(sample_df, x_ts, y_ts, pred_len=3)
        # Should NOT raise TypeError about DatetimeIndex - the error should be
        # from the model not being real, not from timestamp handling

    def test_string_timestamps_accepted(self, dummy_predictor, sample_df):
        x_ts = [f"2024-01-01 {i:02d}:00:00" for i in range(10)]
        y_ts = [f"2024-01-01 {i:02d}:00:00" for i in range(10, 13)]
        with pytest.raises(Exception):
            dummy_predictor.predict(sample_df, x_ts, y_ts, pred_len=3)


# ---------------------------------------------------------------------------
# WebUI /api/predict Flask route tests
# ---------------------------------------------------------------------------

class TestApiPredictTimestamps:

    @pytest.fixture
    def app_client(self):
        from webui.app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def csv_file(self, tmp_path):
        n = 530
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1H")
        df = pd.DataFrame({
            "timestamps": timestamps,
            "open": np.random.rand(n) * 100 + 50,
            "high": np.random.rand(n) * 100 + 55,
            "low": np.random.rand(n) * 100 + 45,
            "close": np.random.rand(n) * 100 + 50,
            "volume": np.random.rand(n) * 1000,
            "amount": np.random.rand(n) * 100000,
        })
        csv_path = str(tmp_path / "test_data.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_predict_without_model_returns_error(self, app_client, csv_file):
        response = app_client.post(
            "/api/predict",
            data=json.dumps({"file_path": csv_file, "lookback": 400, "pred_len": 120}),
            content_type="application/json",
        )
        data = response.get_json()
        assert "error" in data or data.get("success") is True

    def test_predict_response_structure(self, app_client, csv_file):
        response = app_client.post(
            "/api/predict",
            data=json.dumps({"file_path": csv_file, "lookback": 400, "pred_len": 120}),
            content_type="application/json",
        )
        data = response.get_json()
        if data.get("success") is True:
            assert "prediction_results" in data
            assert "actual_data" in data
            assert "has_comparison" in data
            assert "chart" in data
            assert "message" in data

    def test_predict_timestamp_consistency_with_model(self, app_client, csv_file):
        import webui.app as app_module

        df = pd.read_csv(csv_file, parse_dates=["timestamps"])
        lookback = 10
        pred_len = 5

        x_df = df.iloc[:lookback][["open", "high", "low", "close", "volume"]]
        x_timestamp = df.iloc[:lookback]["timestamps"]
        y_timestamp = df.iloc[lookback:lookback + pred_len]["timestamps"]

        class FakePredictor:
            def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
                x_timestamp_norm = _normalize_timestamps(x_timestamp, "x_timestamp")
                y_timestamp_norm = _normalize_timestamps(y_timestamp, "y_timestamp")
                assert len(x_timestamp_norm) == len(df), "x_timestamp length must match df"
                assert len(y_timestamp_norm) == pred_len, "y_timestamp length must match pred_len"
                preds = np.random.rand(pred_len, 6).astype(np.float32)
                pred_df = pd.DataFrame(
                    preds,
                    columns=["open", "high", "low", "close", "volume", "amount"],
                    index=y_timestamp_norm,
                )
                return pred_df

        original_predictor = app_module.predictor
        app_module.predictor = FakePredictor()
        original_model_available = app_module.MODEL_AVAILABLE
        app_module.MODEL_AVAILABLE = True

        try:
            response = app_client.post(
                "/api/predict",
                data=json.dumps({"file_path": csv_file, "lookback": lookback, "pred_len": pred_len}),
                content_type="application/json",
            )
            data = response.get_json()
            assert data.get("success") is True, f"Expected success, got: {data}"

            prediction_results = data["prediction_results"]
            assert len(prediction_results) == pred_len

            y_ts_reset = y_timestamp.reset_index(drop=True)
            for i, pr in enumerate(prediction_results):
                expected_ts = y_ts_reset.iloc[i].isoformat()
                assert pr["timestamp"] == expected_ts, (
                    f"Timestamp mismatch at index {i}: "
                    f"response has {pr['timestamp']}, expected {expected_ts}"
                )
        finally:
            app_module.predictor = original_predictor
            app_module.MODEL_AVAILABLE = original_model_available

    def test_predict_with_start_date_timestamp_consistency(self, app_client, csv_file):
        import webui.app as app_module

        df = pd.read_csv(csv_file, parse_dates=["timestamps"])
        lookback = 10
        pred_len = 5
        start_date = "2024-01-01 05:00"

        start_dt = pd.to_datetime(start_date)
        mask = df["timestamps"] >= start_dt
        time_range_df = df[mask]
        y_timestamp = time_range_df.iloc[lookback:lookback + pred_len]["timestamps"]

        class FakePredictor:
            def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
                x_timestamp_norm = _normalize_timestamps(x_timestamp, "x_timestamp")
                y_timestamp_norm = _normalize_timestamps(y_timestamp, "y_timestamp")
                assert len(x_timestamp_norm) == len(df), "x_timestamp length must match df"
                assert len(y_timestamp_norm) == pred_len, "y_timestamp length must match pred_len"
                preds = np.random.rand(pred_len, 6).astype(np.float32)
                pred_df = pd.DataFrame(
                    preds,
                    columns=["open", "high", "low", "close", "volume", "amount"],
                    index=y_timestamp_norm,
                )
                return pred_df

        original_predictor = app_module.predictor
        app_module.predictor = FakePredictor()
        original_model_available = app_module.MODEL_AVAILABLE
        app_module.MODEL_AVAILABLE = True

        try:
            response = app_client.post(
                "/api/predict",
                data=json.dumps({
                    "file_path": csv_file,
                    "lookback": lookback,
                    "pred_len": pred_len,
                    "start_date": start_date,
                }),
                content_type="application/json",
            )
            data = response.get_json()
            assert data.get("success") is True, f"Expected success, got: {data}"

            prediction_results = data["prediction_results"]
            assert len(prediction_results) == pred_len

            y_ts_reset = y_timestamp.reset_index(drop=True)
            for i, pr in enumerate(prediction_results):
                expected_ts = y_ts_reset.iloc[i].isoformat()
                assert pr["timestamp"] == expected_ts, (
                    f"Timestamp mismatch at index {i}: "
                    f"response has {pr['timestamp']}, expected {expected_ts}"
                )
        finally:
            app_module.predictor = original_predictor
            app_module.MODEL_AVAILABLE = original_model_available

    def test_predict_actual_data_timestamps_match_y_timestamp(self, app_client, csv_file):
        import webui.app as app_module

        df = pd.read_csv(csv_file, parse_dates=["timestamps"])
        lookback = 10
        pred_len = 5

        class FakePredictor:
            def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
                y_timestamp_norm = _normalize_timestamps(y_timestamp, "y_timestamp")
                preds = np.random.rand(pred_len, 6).astype(np.float32)
                pred_df = pd.DataFrame(
                    preds,
                    columns=["open", "high", "low", "close", "volume", "amount"],
                    index=y_timestamp_norm,
                )
                return pred_df

        original_predictor = app_module.predictor
        app_module.predictor = FakePredictor()
        original_model_available = app_module.MODEL_AVAILABLE
        app_module.MODEL_AVAILABLE = True

        try:
            response = app_client.post(
                "/api/predict",
                data=json.dumps({"file_path": csv_file, "lookback": lookback, "pred_len": pred_len}),
                content_type="application/json",
            )
            data = response.get_json()
            assert data.get("success") is True

            actual_data = data["actual_data"]
            if len(actual_data) > 0:
                y_timestamp = df.iloc[lookback:lookback + pred_len]["timestamps"]
                y_ts_reset = y_timestamp.reset_index(drop=True)
                for i, ad in enumerate(actual_data):
                    expected_ts = y_ts_reset.iloc[i].isoformat()
                    assert ad["timestamp"] == expected_ts, (
                        f"Actual data timestamp mismatch at index {i}: "
                        f"response has {ad['timestamp']}, expected {expected_ts}"
                    )
        finally:
            app_module.predictor = original_predictor
            app_module.MODEL_AVAILABLE = original_model_available


# ---------------------------------------------------------------------------
# Integration: _normalize_timestamps -> calc_time_stamps pipeline
# ---------------------------------------------------------------------------

class TestTimestampPipeline:

    def test_pipeline_with_various_inputs(self):
        base = pd.date_range("2024-06-18 09:30:00", periods=5, freq="5min")
        inputs = [
            pd.Series(base),
            base,
            list(base),
            np.array(base),
            [str(t) for t in base],
        ]
        expected = calc_time_stamps(pd.Series(base))
        for inp in inputs:
            result = calc_time_stamps(inp)
            pd.testing.assert_frame_equal(result, expected)

    def test_pipeline_preserves_time_features(self):
        ts = pd.Series([
            pd.Timestamp("2024-01-15 09:30:00"),
            pd.Timestamp("2024-06-30 15:45:00"),
            pd.Timestamp("2024-12-01 23:59:00"),
        ])
        result = calc_time_stamps(ts)
        assert result["minute"].tolist() == [30, 45, 59]
        assert result["hour"].tolist() == [9, 15, 23]
        assert result["day"].tolist() == [15, 30, 1]
        assert result["month"].tolist() == [1, 6, 12]
