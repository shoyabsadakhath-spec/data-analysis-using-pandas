import os
import uuid
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import plotly
import plotly.express as px
import tempfile

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

sessions = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_data():
    session_id = session.get('data_session_id')
    if not session_id or session_id not in sessions:
        return None
    return sessions[session_id]

def save_session_data(session_data):
    session_id = session.get('data_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['data_session_id'] = session_id
    sessions[session_id] = session_data

def clear_session():
    session_id = session.get('data_session_id')
    if session_id and session_id in sessions:
        del sessions[session_id]
    session.pop('data_session_id', None)

def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native for JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def perform_cleaning(df, params):
    df_clean = df.copy()
    if params.get('trim_whitespace', True):
        str_cols = df_clean.select_dtypes(include=['object']).columns
        df_clean[str_cols] = df_clean[str_cols].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    if params.get('remove_duplicates', False):
        df_clean = df_clean.drop_duplicates()
    if params.get('handle_missing', False):
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        num_strategy = params.get('missing_numeric_strategy', 'mean')
        for col in numeric_cols:
            if num_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            elif num_strategy == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif num_strategy == 'median':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        cat_strategy = params.get('missing_categorical_strategy', 'mode')
        for col in cat_cols:
            if cat_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            elif cat_strategy == 'mode':
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
    return df_clean

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload CSV.'}), 400
    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({'error': 'CSV file is empty.'}), 400
        session_data = {
            'original_df': df.to_dict('records'),
            'cleaned_df': df.to_dict('records'),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'shape': df.shape,
            'cleaning_params': {}
        }
        save_session_data(session_data)
        preview = df.head(5).to_dict('records')
        return jsonify({
            'message': 'File uploaded successfully',
            'preview': preview,
            'columns': session_data['columns'],
            'shape': session_data['shape'],
            'dtypes': session_data['dtypes']
        })
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

@app.route('/api/clean', methods=['POST'])
def clean_data():
    session_data = get_session_data()
    if not session_data:
        return jsonify({'error': 'No data uploaded. Please upload a CSV first.'}), 400
    try:
        params = request.json or {}
        original_df = pd.DataFrame(session_data['original_df'])
        cleaned_df = perform_cleaning(original_df, params)
        session_data['cleaned_df'] = cleaned_df.to_dict('records')
        session_data['cleaning_params'] = params
        session_data['shape'] = cleaned_df.shape
        save_session_data(session_data)
        before_preview = original_df.head(5).to_dict('records')
        after_preview = cleaned_df.head(5).to_dict('records')
        response_data = {
            'message': 'Cleaning applied successfully',
            'before_preview': before_preview,
            'after_preview': after_preview,
            'shape_before': original_df.shape,
            'shape_after': cleaned_df.shape,
            'missing_before': original_df.isnull().sum().to_dict(),
            'missing_after': cleaned_df.isnull().sum().to_dict(),
            'duplicates_removed': int(original_df.duplicated().sum() - cleaned_df.duplicated().sum())
        }
        response_data = convert_to_serializable(response_data)
        return jsonify(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Cleaning failed: {str(e)}'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    session_data = get_session_data()
    if not session_data:
        return jsonify({'error': 'No data available'}), 400
    df = pd.DataFrame(session_data['cleaned_df'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    stats = {}
    if numeric_cols:
        stats['numeric'] = df[numeric_cols].describe(percentiles=[.25, .5, .75]).to_dict()
        mode_series = df[numeric_cols].mode().iloc[0] if not df[numeric_cols].mode().empty else None
        stats['numeric_extra'] = {
            'mode': mode_series.to_dict() if mode_series is not None else {},
            'skewness': df[numeric_cols].skew().to_dict(),
            'kurtosis': df[numeric_cols].kurtosis().to_dict()
        }
    else:
        stats['numeric'] = {}
    if categorical_cols:
        freq_tables = {}
        for col in categorical_cols:
            freq = df[col].value_counts().head(10).to_dict()
            freq_tables[col] = freq
        stats['categorical'] = freq_tables
    else:
        stats['categorical'] = {}
    stats['overview'] = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'missing_total': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum())
    }
    stats = convert_to_serializable(stats)
    return jsonify(stats)

@app.route('/api/insights', methods=['GET'])
def get_insights():
    session_data = get_session_data()
    if not session_data:
        return jsonify({'error': 'No data available'}), 400
    df = pd.DataFrame(session_data['cleaned_df'])
    insights = []
    missing_counts = df.isnull().sum()
    high_missing = missing_counts[missing_counts > 0.2 * len(df)]
    if not high_missing.empty:
        for col, count in high_missing.items():
            insights.append({
                'title': f'High missing values in "{col}"',
                'description': f'{count} missing values ({count/len(df)*100:.1f}%). Consider imputation or dropping.',
                'priority': 'high',
                'category': 'data_quality'
            })
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        insights.append({
            'title': 'Duplicate rows detected',
            'description': f'Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}% of data).',
            'priority': 'medium',
            'category': 'data_quality'
        })
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            direction = 'right‑skewed' if skew > 0 else 'left‑skewed'
            insights.append({
                'title': f'"{col}" is {direction}',
                'description': f'Skewness = {skew:.2f}. Consider transformation for symmetric distribution.',
                'priority': 'medium',
                'category': 'distribution'
            })
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if not outliers.empty:
            insights.append({
                'title': f'Outliers found in "{col}"',
                'description': f'{len(outliers)} potential outliers ({len(outliers)/len(df)*100:.1f}%). Use IQR method for detection.',
                'priority': 'low',
                'category': 'outlier'
            })
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:
            insights.append({
                'title': f'"{col}" is nearly unique',
                'description': f'{df[col].nunique()} distinct values out of {len(df)} rows. May be an identifier.',
                'priority': 'low',
                'category': 'categorical'
            })
        elif unique_ratio < 0.05:
            top_val = df[col].mode()[0]
            insights.append({
                'title': f'"{col}" has a dominant value',
                'description': f"'{top_val}' appears {df[col].value_counts().iloc[0]} times ({df[col].value_counts(normalize=True).iloc[0]*100:.1f}%).",
                'priority': 'medium',
                'category': 'categorical'
            })
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        if high_corr:
            top = high_corr[0]
            insights.append({
                'title': f'Strong correlation: {top[0]} ↔ {top[1]}',
                'description': f'Pearson correlation = {top[2]:.2f}. Variables move together strongly.',
                'priority': 'high',
                'category': 'correlation'
            })
    insights = convert_to_serializable(insights)
    return jsonify(insights)

@app.route('/api/correlation', methods=['GET'])
def get_correlation():
    session_data = get_session_data()
    if not session_data:
        return jsonify({'error': 'No data available'}), 400
    df = pd.DataFrame(session_data['cleaned_df'])
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return jsonify({'error': 'Need at least two numeric columns for correlation'}), 400
    corr_matrix = numeric_df.corr().round(3)
    result = {
        'columns': numeric_df.columns.tolist(),
        'correlation': corr_matrix.to_dict()
    }
    result = convert_to_serializable(result)
    return jsonify(result)

@app.route('/api/visualize', methods=['POST'])
def visualize():
    session_data = get_session_data()
    if not session_data:
        return jsonify({'error': 'No data available'}), 400
    df = pd.DataFrame(session_data['cleaned_df'])
    req = request.json
    chart_type = req.get('type')
    x_col = req.get('x')
    y_col = req.get('y')
    color_col = req.get('color')
    fig = None
    if chart_type == 'histogram':
        if not x_col or x_col not in df.columns:
            return jsonify({'error': 'Invalid x column'}), 400
        fig = px.histogram(df, x=x_col, title=f'Distribution of {x_col}', nbins=30)
    elif chart_type == 'bar':
        if not x_col or x_col not in df.columns:
            return jsonify({'error': 'Invalid x column'}), 400
        if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f'{y_col} by {x_col}')
        else:
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, 'count']
            fig = px.bar(counts, x=x_col, y='count', title=f'Frequency of {x_col}')
    elif chart_type == 'scatter':
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return jsonify({'error': 'Invalid x or y columns'}), 400
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f'{y_col} vs {x_col}')
    elif chart_type == 'correlation_heatmap':
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return jsonify({'error': 'Need at least two numeric columns'}), 400
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation Heatmap')
    else:
        return jsonify({'error': 'Unsupported chart type'}), 400
    graph_json = json.loads(plotly.io.to_json(fig))
    return jsonify({'graph': graph_json})

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    session_data = get_session_data()
    if not session_data:
        return jsonify({'error': 'No data available'}), 400
    df = pd.DataFrame(session_data['cleaned_df'])
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='cleaned_data.csv')

@app.route('/api/export/json', methods=['GET'])
def export_json():
    session_data = get_session_data()
    if not session_data:
        return jsonify({'error': 'No data available'}), 400
    df = pd.DataFrame(session_data['cleaned_df'])
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'summary': df.describe(include='all').to_dict(),
        'missing': df.isnull().sum().to_dict()
    }
    stats = convert_to_serializable(stats)
    return jsonify(stats)

@app.route('/api/session', methods=['DELETE'])
def delete_session():
    clear_session()
    return jsonify({'message': 'Session cleared'})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)