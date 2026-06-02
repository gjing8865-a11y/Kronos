import io
import sys
import contextlib
import traceback

sys.path.insert(0, "/app/Kronos")

output_lines = []

def log(msg):
    output_lines.append(msg)
    print(msg)

log("=== Running timestamp consistency tests ===")

try:
    import numpy as np
    import pandas as pd
    from model.kronos import _normalize_timestamps, calc_time_stamps, KronosPredictor
    log("OK: imports successful")
except Exception as e:
    log(f"FAIL: imports failed: {e}")
    traceback.print_exc()

# Test _normalize_timestamps
passed = 0
failed = 0
total = 0

def run_test(name, func):
    global passed, failed, total
    total += 1
    try:
        func()
        passed += 1
        log(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        log(f"  FAIL: {name}: {e}")

log("\n--- _normalize_timestamps tests ---")

def test_series_datetime():
    ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))
    result = _normalize_timestamps(ts, "ts")
    assert isinstance(result, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert len(result) == 10
run_test("series_datetime", test_series_datetime)

def test_datetime_index():
    ts = pd.date_range("2024-01-01", periods=10, freq="1H")
    result = _normalize_timestamps(ts, "ts")
    assert isinstance(result, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert len(result) == 10
run_test("datetime_index", test_datetime_index)

def test_list_input():
    ts = list(pd.date_range("2024-01-01", periods=10, freq="1H"))
    result = _normalize_timestamps(ts, "ts")
    assert isinstance(result, pd.Series)
    assert len(result) == 10
run_test("list_input", test_list_input)

def test_numpy_input():
    ts = np.array(pd.date_range("2024-01-01", periods=10, freq="1H"))
    result = _normalize_timestamps(ts, "ts")
    assert isinstance(result, pd.Series)
    assert len(result) == 10
run_test("numpy_input", test_numpy_input)

def test_string_list_input():
    ts = [f"2024-01-01 {i:02d}:00:00" for i in range(10)]
    result = _normalize_timestamps(ts, "ts")
    assert isinstance(result, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert len(result) == 10
run_test("string_list_input", test_string_list_input)

def test_invalid_type_raises():
    try:
        _normalize_timestamps(42, "ts")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
run_test("invalid_type_raises", test_invalid_type_raises)

def test_invalid_string_raises():
    try:
        _normalize_timestamps(["not-a-date"], "ts")
        assert False, "Should have raised ValueError"
    except (ValueError, Exception):
        pass
run_test("invalid_string_raises", test_invalid_string_raises)

def test_null_values_raise():
    ts = pd.Series([pd.Timestamp("2024-01-01"), pd.NaT])
    try:
        _normalize_timestamps(ts, "ts")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
run_test("null_values_raise", test_null_values_raise)

def test_reset_index():
    ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))
    ts.index = range(100, 110)
    result = _normalize_timestamps(ts, "ts")
    assert list(result.index) == list(range(10))
run_test("reset_index", test_reset_index)

def test_preserves_values():
    ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))
    result = _normalize_timestamps(ts, "ts")
    for i in range(len(ts)):
        assert result.iloc[i] == ts.iloc[i]
run_test("preserves_values", test_preserves_values)

log("\n--- calc_time_stamps tests ---")

def test_calc_basic_output():
    ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))
    result = calc_time_stamps(ts)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["minute", "hour", "weekday", "day", "month"]
    assert len(result) == 10
run_test("calc_basic_output", test_calc_basic_output)

def test_calc_correct_values():
    ts = pd.Series([pd.Timestamp("2024-03-15 14:30:00")])
    result = calc_time_stamps(ts)
    assert result["minute"].iloc[0] == 30
    assert result["hour"].iloc[0] == 14
    assert result["weekday"].iloc[0] == 4
    assert result["day"].iloc[0] == 15
    assert result["month"].iloc[0] == 3
run_test("calc_correct_values", test_calc_correct_values)

def test_calc_with_datetime_index():
    ts = pd.date_range("2024-01-01", periods=10, freq="1H")
    result = calc_time_stamps(ts)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
run_test("calc_with_datetime_index", test_calc_with_datetime_index)

def test_calc_with_string_list():
    ts = [f"2024-01-01 {i:02d}:00:00" for i in range(10)]
    result = calc_time_stamps(ts)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
run_test("calc_with_string_list", test_calc_with_string_list)

def test_calc_with_numpy_array():
    ts = np.array(pd.date_range("2024-01-01", periods=10, freq="1H"))
    result = calc_time_stamps(ts)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
run_test("calc_with_numpy_array", test_calc_with_numpy_array)

def test_calc_invalid_input_raises():
    try:
        calc_time_stamps(42)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
run_test("calc_invalid_input_raises", test_calc_invalid_input_raises)

def test_calc_all_columns_numeric():
    ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))
    result = calc_time_stamps(ts)
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col])
run_test("calc_all_columns_numeric", test_calc_all_columns_numeric)

log("\n--- KronosPredictor validation tests ---")

class DummyModel:
    pass

class DummyTokenizer:
    pass

predictor = KronosPredictor(DummyModel(), DummyTokenizer(), device="cpu")

def test_predict_x_timestamp_length_mismatch():
    df = pd.DataFrame({
        "open": np.random.rand(10) * 100 + 50,
        "high": np.random.rand(10) * 100 + 55,
        "low": np.random.rand(10) * 100 + 45,
        "close": np.random.rand(10) * 100 + 50,
        "volume": np.random.rand(10) * 1000,
        "amount": np.random.rand(10) * 100000,
    })
    x_ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1H"))
    y_ts = pd.Series(pd.date_range("2024-01-01", periods=3, freq="1H"))
    try:
        predictor.predict(df, x_ts, y_ts, pred_len=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "x_timestamp length" in str(e)
run_test("predict_x_timestamp_length_mismatch", test_predict_x_timestamp_length_mismatch)

def test_predict_y_timestamp_length_mismatch():
    df = pd.DataFrame({
        "open": np.random.rand(10) * 100 + 50,
        "high": np.random.rand(10) * 100 + 55,
        "low": np.random.rand(10) * 100 + 45,
        "close": np.random.rand(10) * 100 + 50,
        "volume": np.random.rand(10) * 1000,
        "amount": np.random.rand(10) * 100000,
    })
    x_ts = pd.Series(pd.date_range("2024-01-01", periods=10, freq="1H"))
    y_ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="1H"))
    try:
        predictor.predict(df, x_ts, y_ts, pred_len=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "y_timestamp length" in str(e)
run_test("predict_y_timestamp_length_mismatch", test_predict_y_timestamp_length_mismatch)

def test_predict_non_dataframe_raises():
    try:
        predictor.predict(
            [1, 2, 3],
            pd.Series(pd.date_range("2024-01-01", periods=3, freq="1H")),
            pd.Series(pd.date_range("2024-01-01", periods=3, freq="1H")),
            pred_len=3,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be a pandas DataFrame" in str(e)
run_test("predict_non_dataframe_raises", test_predict_non_dataframe_raises)

def test_predict_missing_price_columns_raises():
    df = pd.DataFrame({"a": [1], "b": [2]})
    try:
        predictor.predict(
            df,
            pd.Series(pd.date_range("2024-01-01", periods=1, freq="1H")),
            pd.Series(pd.date_range("2024-01-01", periods=1, freq="1H")),
            pred_len=1,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Price columns" in str(e)
run_test("predict_missing_price_columns_raises", test_predict_missing_price_columns_raises)

def test_predict_nan_values_raise():
    df = pd.DataFrame({
        "open": [1.0, np.nan],
        "high": [2.0, 3.0],
        "low": [0.5, 1.0],
        "close": [1.5, 2.0],
        "volume": [100.0, 200.0],
        "amount": [1000.0, 2000.0],
    })
    try:
        predictor.predict(
            df,
            pd.Series(pd.date_range("2024-01-01", periods=2, freq="1H")),
            pd.Series(pd.date_range("2024-01-01", periods=1, freq="1H")),
            pred_len=1,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "NaN values" in str(e)
run_test("predict_nan_values_raise", test_predict_nan_values_raise)

log("\n--- Pipeline tests ---")

def test_pipeline_with_various_inputs():
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
run_test("pipeline_with_various_inputs", test_pipeline_with_various_inputs)

def test_pipeline_preserves_time_features():
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
run_test("pipeline_preserves_time_features", test_pipeline_preserves_time_features)

log("\n--- Flask route tests ---")

def test_flask_predict_timestamp_consistency():
    import json
    from webui.app import app
    app.config["TESTING"] = True
    import webui.app as app_module

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

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    lookback = 10
    pred_len = 5
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
        with app.test_client() as client:
            response = client.post(
                "/api/predict",
                data=json.dumps({"file_path": csv_path, "lookback": lookback, "pred_len": pred_len}),
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

    import os
    os.unlink(csv_path)

run_test("flask_predict_timestamp_consistency", test_flask_predict_timestamp_consistency)

def test_flask_predict_with_start_date():
    import json
    from webui.app import app
    app.config["TESTING"] = True
    import webui.app as app_module

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

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

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
        with app.test_client() as client:
            response = client.post(
                "/api/predict",
                data=json.dumps({
                    "file_path": csv_path,
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

    import os
    os.unlink(csv_path)

run_test("flask_predict_with_start_date", test_flask_predict_with_start_date)

def test_flask_predict_actual_data_timestamps():
    import json
    from webui.app import app
    app.config["TESTING"] = True
    import webui.app as app_module

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

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

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
        with app.test_client() as client:
            response = client.post(
                "/api/predict",
                data=json.dumps({"file_path": csv_path, "lookback": lookback, "pred_len": pred_len}),
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

    import os
    os.unlink(csv_path)

run_test("flask_predict_actual_data_timestamps", test_flask_predict_actual_data_timestamps)

log(f"\n{'='*60}")
log(f"Results: {passed} passed, {failed} failed, {total} total")
log(f"{'='*60}")

with open("/app/Kronos/tests/_inline_test_output.txt", "w") as f:
    f.write("\n".join(output_lines))
