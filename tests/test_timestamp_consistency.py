import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from model.kronos import _normalize_timestamps, calc_time_stamps
from webui.app import app, load_data_file

def test_normalize_timestamps():
    # Test DatetimeIndex
    dt_index = pd.date_range("2024-01-01", periods=5, freq="D")
    normalized = _normalize_timestamps(dt_index, "x")
    assert isinstance(normalized, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(normalized)

    # Test List
    dt_list = ["2024-01-01", "2024-01-02", "2024-01-03"]
    normalized = _normalize_timestamps(dt_list, "x")
    assert isinstance(normalized, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(normalized)

    # Test numpy array
    dt_np = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    normalized = _normalize_timestamps(dt_np, "x")
    assert isinstance(normalized, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(normalized)

    # Test Series with datetime
    dt_series = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D"))
    normalized = _normalize_timestamps(dt_series, "x")
    assert isinstance(normalized, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(normalized)

    # Test Series with string
    dt_series_str = pd.Series(["2024-01-01", "2024-01-02"])
    normalized = _normalize_timestamps(dt_series_str, "x")
    assert isinstance(normalized, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(normalized)

def test_calc_time_stamps():
    # It should not raise an error with DatetimeIndex
    dt_index = pd.date_range("2024-01-01 10:30", periods=5, freq="h")
    time_df = calc_time_stamps(dt_index)
    assert "minute" in time_df.columns
    assert "hour" in time_df.columns
    assert "weekday" in time_df.columns
    assert "day" in time_df.columns
    assert "month" in time_df.columns
    assert time_df["minute"].iloc[0] == 30
    assert time_df["hour"].iloc[0] == 10

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_predict_latest_data(client, tmp_path):
    # Create dummy data
    data_path = tmp_path / "dummy.csv"
    df = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-01", periods=500, freq="h"),
        "open": np.random.randn(500),
        "high": np.random.randn(500),
        "low": np.random.randn(500),
        "close": np.random.randn(500),
        "volume": np.random.randn(500)
    })
    df.to_csv(data_path, index=False)

    # Mock Kronos Predictor to avoid loading real model in tests
    with patch("webui.app.MODEL_AVAILABLE", True), \
         patch("webui.app.predictor") as mock_predictor:
        
        # Mock the predictor output
        mock_pred_df = pd.DataFrame({
            "open": [1.0]*120,
            "high": [1.1]*120,
            "low": [0.9]*120,
            "close": [1.05]*120,
            "volume": [100]*120,
            "amount": [1000]*120
        }, index=pd.date_range(df["timestamps"].iloc[-1] + pd.Timedelta(hours=1), periods=120, freq="h"))
        mock_predictor.predict.return_value = mock_pred_df

        response = client.post("/api/predict", json={
            "file_path": str(data_path),
            "lookback": 400,
            "pred_len": 120
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "prediction_results" in data
        assert len(data["prediction_results"]) == 120
        assert "actual_data" in data
        assert len(data["actual_data"]) == 0
        assert data["has_comparison"] is False
        assert "chart" in data
        assert "message" in data

def test_api_predict_custom_date(client, tmp_path):
    # Create dummy data
    data_path = tmp_path / "dummy.csv"
    df = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-01", periods=600, freq="h"),
        "open": np.random.randn(600),
        "high": np.random.randn(600),
        "low": np.random.randn(600),
        "close": np.random.randn(600),
        "volume": np.random.randn(600)
    })
    df.to_csv(data_path, index=False)

    with patch("webui.app.MODEL_AVAILABLE", True), \
         patch("webui.app.predictor") as mock_predictor:
        
        # Mock the predictor output
        mock_pred_df = pd.DataFrame({
            "open": [1.0]*120,
            "high": [1.1]*120,
            "low": [0.9]*120,
            "close": [1.05]*120,
            "volume": [100]*120,
            "amount": [1000]*120
        })
        mock_predictor.predict.return_value = mock_pred_df

        start_date = "2024-01-02 00:00:00"
        response = client.post("/api/predict", json={
            "file_path": str(data_path),
            "lookback": 400,
            "pred_len": 120,
            "start_date": start_date
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "prediction_results" in data
        assert len(data["prediction_results"]) == 120
        assert "actual_data" in data
        assert len(data["actual_data"]) == 120
        assert data["has_comparison"] is True
        assert "chart" in data
        assert "message" in data
