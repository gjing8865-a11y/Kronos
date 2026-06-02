import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def _get_prediction_timestamps(pred_df, historical_df=None):
    if pred_df is None or len(pred_df) == 0:
        return []

    if isinstance(pred_df.index, pd.DatetimeIndex):
        return list(pred_df.index)

    if historical_df is not None and 'timestamps' in historical_df.columns and len(historical_df) > 0:
        base_timestamps = pd.to_datetime(historical_df['timestamps']).reset_index(drop=True)
        if len(base_timestamps) >= 2:
            time_diff = base_timestamps.iloc[-1] - base_timestamps.iloc[-2]
        else:
            time_diff = pd.Timedelta(hours=1)
        return list(pd.date_range(start=base_timestamps.iloc[-1] + time_diff, periods=len(pred_df), freq=time_diff))

    return list(range(len(pred_df)))


def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    available_lookback = max(0, min(lookback, len(df) - historical_start_idx))
    historical_df = df.iloc[historical_start_idx:historical_start_idx + available_lookback]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))

    pred_timestamps = _get_prediction_timestamps(pred_df, historical_df)
    if pred_df is not None and len(pred_df) > 0:
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))

    actual_timestamps = []
    if actual_df is not None and len(actual_df) > 0:
        if 'timestamps' in actual_df.columns:
            actual_timestamps = list(pd.to_datetime(actual_df['timestamps']))
        else:
            actual_timestamps = pred_timestamps[:len(actual_df)] if pred_timestamps else list(range(len(actual_df)))

        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))

    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )

    all_timestamps = []
    if 'timestamps' in historical_df.columns and len(historical_df) > 0:
        all_timestamps.extend(pd.to_datetime(historical_df['timestamps']).tolist())
    all_timestamps.extend(pred_timestamps)
    all_timestamps.extend(actual_timestamps)

    if all_timestamps and all(isinstance(ts, (pd.Timestamp, datetime.datetime, np.datetime64)) for ts in all_timestamps):
        all_timestamps = sorted(pd.to_datetime(all_timestamps).tolist())
        fig.update_xaxes(
            range=[all_timestamps[0], all_timestamps[-1]],
            rangeslider_visible=False,
            type='date'
        )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500


def _infer_time_diff(timestamps):
    normalized = pd.Series(pd.to_datetime(timestamps)).dropna().reset_index(drop=True)
    if len(normalized) >= 2:
        return normalized.iloc[-1] - normalized.iloc[-2]
    return pd.Timedelta(hours=1)


def _generate_future_timestamps(timestamps, pred_len):
    normalized = pd.Series(pd.to_datetime(timestamps)).dropna().reset_index(drop=True)
    if len(normalized) == 0:
        return pd.Series(dtype='datetime64[ns]')

    time_diff = _infer_time_diff(normalized)
    return pd.Series(
        pd.date_range(start=normalized.iloc[-1] + time_diff, periods=pred_len, freq=time_diff),
        name='timestamps'
    )


def _build_prediction_inputs(df, required_cols, lookback, pred_len, start_date=None):
    if start_date:
        start_dt = pd.to_datetime(start_date)
        time_range_df = df[df['timestamps'] >= start_dt]
        if len(time_range_df) < lookback + pred_len:
            raise ValueError(
                f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'
            )

        historical_window = time_range_df.iloc[:lookback].copy()
        actual_df = time_range_df.iloc[lookback:lookback + pred_len].copy()
        historical_start_idx = int(historical_window.index[0]) if len(historical_window) > 0 else 0
        start_timestamp = historical_window['timestamps'].iloc[0]
        end_timestamp = actual_df['timestamps'].iloc[-1]
        time_span = end_timestamp - start_timestamp
        prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
        y_timestamp = actual_df['timestamps'].reset_index(drop=True)
    elif len(df) >= lookback + pred_len:
        window_df = df.iloc[-(lookback + pred_len):].copy()
        historical_window = window_df.iloc[:lookback].copy()
        actual_df = window_df.iloc[lookback:lookback + pred_len].copy()
        historical_start_idx = int(historical_window.index[0]) if len(historical_window) > 0 else max(len(df) - lookback, 0)
        prediction_type = "Kronos model prediction (latest comparable window)"
        y_timestamp = actual_df['timestamps'].reset_index(drop=True)
    else:
        historical_window = df.iloc[-lookback:].copy()
        actual_df = None
        historical_start_idx = int(historical_window.index[0]) if len(historical_window) > 0 else 0
        prediction_type = "Kronos model prediction (latest data)"
        y_timestamp = _generate_future_timestamps(historical_window['timestamps'], pred_len)

    x_df = historical_window[required_cols].reset_index(drop=True)
    x_timestamp = historical_window['timestamps'].reset_index(drop=True)

    return x_df, x_timestamp, y_timestamp, actual_df, historical_start_idx, prediction_type


@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        start_date = data.get('start_date')

        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400

        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400

        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400

        if not (MODEL_AVAILABLE and predictor is not None):
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400

        required_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            required_cols.append('volume')

        try:
            x_df, x_timestamp, y_timestamp, actual_df, historical_start_idx, prediction_type = _build_prediction_inputs(
                df=df,
                required_cols=required_cols,
                lookback=lookback,
                pred_len=pred_len,
                start_date=start_date
            )

            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count
            )
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500

        actual_data = []
        if actual_df is not None and len(actual_df) > 0:
            for _, row in actual_df.iterrows():
                actual_data.append({
                    'timestamp': row['timestamps'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 0,
                    'amount': float(row['amount']) if 'amount' in row else 0
                })

        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)

        prediction_results = []
        prediction_timestamps = _get_prediction_timestamps(pred_df)
        for i, (_, row) in enumerate(pred_df.iterrows()):
            timestamp_value = prediction_timestamps[i] if i < len(prediction_timestamps) else f"T{i}"
            if hasattr(timestamp_value, 'isoformat'):
                timestamp_value = timestamp_value.isoformat()
            prediction_results.append({
                'timestamp': timestamp_value,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })

        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")

        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
