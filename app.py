import io
import base64
import json
import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from utils.predictor import validate, build_feature_vector, predict_crop, FEATURE_ORDER

REGIONS = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
    'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
    'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
    'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
    'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

app = Flask(
    __name__,
    static_folder='static',
    static_url_path='/static',
    template_folder='templates'
)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')

# Load pesticide recommendations, crop details, and region map
pesticide_df    = pd.read_csv(os.path.join(data_dir, 'pesticides_recommendation.csv'))
crop_details_df = pd.read_csv(os.path.join(data_dir, 'crop_details.csv'))
with open(os.path.join(data_dir, 'crop_region.json')) as f:
    CROP_REGION_MAP = json.load(f)


def check_region_suitability(crop, region):
    """Return (is_suitable, alternative_crops)."""
    suitable_regions = CROP_REGION_MAP.get(crop.lower(), [])
    if region in suitable_regions:
        return True, []
    # Suggest crops that do grow well in this region
    alternatives = [
        c.title() for c, regions in CROP_REGION_MAP.items()
        if region in regions and c != crop.lower()
    ][:3]
    return False, alternatives

def get_pesticide_recommendation(crop_name):
    if not crop_name:
        return []
    crop_name_clean = str(crop_name).strip().lower()
    results = pesticide_df[pesticide_df['crop'].astype(str).str.strip().str.lower() == crop_name_clean]
    return results.to_dict(orient='records')


def get_crop_details(crop_name):
    if not crop_name:
        return None
    crop_name_clean = str(crop_name).strip().lower()
    row = crop_details_df[crop_details_df['crop'].astype(str).str.strip().str.lower() == crop_name_clean]
    if row.empty:
        return None
    details = row.iloc[0].to_dict()
    # Build harvest time based on growth days
    details['harvest_time'] = f"Approximately {details['growth_days']} days after planting"
    details['display_crop'] = details.get('crop', crop_name).title()
    details['fertilizer_suggestion'] = details.get('fertilizer', 'NPK balanced')
    details['market_demand'] = details.get('market_demand', 'Stable')
    details['soil_npk'] = {
        'N': details.get('ideal_n'),
        'P': details.get('ideal_p'),
        'K': details.get('ideal_k')
    }
    return details


def compute_soil_progress(features, details):
    if not details or not details.get('soil_npk'):
        return {}

    if not hasattr(features, 'get'):
        try:
            feature_list = features.tolist()
        except Exception:
            feature_list = list(features)
        features = dict(zip(FEATURE_ORDER, feature_list))

    def safe_percent(value, ideal):
        try:
            return max(0, min(100, round((float(value) / float(ideal)) * 100)))
        except Exception:
            return 0

    return {
        'N': safe_percent(features.get('N', 0), details['soil_npk'].get('N', 1)),
        'P': safe_percent(features.get('P', 0), details['soil_npk'].get('P', 1)),
        'K': safe_percent(features.get('K', 0), details['soil_npk'].get('K', 1))
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', regions=REGIONS)

    region = request.form.get('region', '').strip()
    if not region or region not in REGIONS:
        return render_template('index.html', regions=REGIONS, error="Please select a valid region.")

    features, error = validate(request.form)
    if error:
        logging.error("Validation error: %s", error)
        return render_template('index.html', regions=REGIONS, error=error)

    try:
        feature_vector = build_feature_vector(features)
        crop = predict_crop(feature_vector)
        pesticides = get_pesticide_recommendation(crop)
        details = get_crop_details(crop)
        soil_progress = compute_soil_progress(features, details)
        is_suitable, alternatives = check_region_suitability(crop, region)
        logging.info("Predicted crop: %s | Region: %s | Suitable: %s", crop, region, is_suitable)
        return render_template(
            'result.html',
            crop=crop,
            crop_details=details,
            pesticides=pesticides,
            soil_progress=soil_progress,
            features=features,
            region=region,
            region_suitable=is_suitable,
            region_alternatives=alternatives
        )
    except Exception as e:
        logging.exception("Prediction failed")
        return render_template('error.html', message=str(e) or "Prediction failed. Please try again.")


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        region = data.get('region', '').strip()
        if not region or region not in REGIONS:
            return jsonify({'error': 'Invalid or missing region'}), 400
        features, error = validate({k: str(v) for k, v in data.items()})
        if error:
            return jsonify({'error': error}), 400
        crop = predict_crop(build_feature_vector(features))
        is_suitable, alternatives = check_region_suitability(crop, region)
        return jsonify({
            'crop': crop,
            'region': region,
            'region_suitable': is_suitable,
            'region_alternatives': alternatives,
            'pesticides': get_pesticide_recommendation(crop)
        })
    except Exception as e:
        logging.error("API error: %s", e)
        return jsonify({'error': str(e)}), 500


def _chart_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0e1a26')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


@app.route('/results')
def results():
    from sklearn.metrics import confusion_matrix

    data = pd.read_csv(os.path.join(base_dir, 'data', 'Crop_recommendation.csv'))
    if 'soil_moisture' not in data.columns:
        rng_sm = np.random.default_rng(42)
        base = (data['humidity'] * 0.7) + (data['rainfall'] / 30 * 0.3)
        data['soil_moisture'] = (base + rng_sm.normal(0, 8, size=len(data))).clip(0, 100)

    X = data[FEATURE_ORDER]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # apply same noise as train_model.py so evaluation matches saved model
    rng2 = np.random.default_rng(7)
    X_train = X_train.copy()
    X_test  = X_test.copy()
    for col in FEATURE_ORDER:
        std = X_train[col].std()
        X_train[col] += rng2.normal(0, std * 0.22, size=len(X_train))
        X_test[col]  += rng2.normal(0, std * 0.22, size=len(X_test))

    scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
    model  = joblib.load(os.path.join(base_dir, 'models', 'model.pkl'))

    X_test_scaled  = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    y_pred         = model.predict(X_test_scaled)

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    train_acc = round(model.score(X_train_scaled, y_train) * 100, 2)
    report   = classification_report(y_test, y_pred, output_dict=True)

    def _fmt(d):
        return {'precision': round(d['precision'],4), 'recall': round(d['recall'],4), 'f1_score': round(d['f1-score'],4)}

    macro    = _fmt(report['macro avg'])
    weighted = _fmt(report['weighted avg'])

    # per-class table
    classes_all = sorted([k for k in report if k not in ('accuracy','macro avg','weighted avg')])
    class_table = [{
        'name': c.title(),
        'precision': round(report[c]['precision'], 3),
        'recall':    round(report[c]['recall'],    3),
        'f1':        round(report[c]['f1-score'],  3),
        'support':   int(report[c]['support'])
    } for c in classes_all]

    # dataset stats
    dataset_stats = {
        'total':    len(data),
        'train':    len(X_train),
        'test':     len(X_test),
        'classes':  y.nunique(),
        'features': len(FEATURE_ORDER)
    }

    # ── Chart 1: Summary metrics bar ──
    met_labels = ['Accuracy','Macro\nPrecision','Macro\nRecall','Macro\nF1',
                  'Weighted\nPrecision','Weighted\nRecall','Weighted\nF1']
    met_values = [accuracy/100, macro['precision'], macro['recall'], macro['f1_score'],
                  weighted['precision'], weighted['recall'], weighted['f1_score']]
    met_colors = ['#7ed957','#5bc0de','#5bc0de','#5bc0de','#f7b731','#f7b731','#f7b731']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#0b1929'); ax.set_facecolor('#0b1929')
    x_pos = np.arange(len(met_labels))
    bars = ax.bar(x_pos, met_values, color=met_colors, width=0.55, zorder=3)
    ax.set_xticks(x_pos); ax.set_xticklabels(met_labels, color='#c8d8e8', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_tick_params(labelcolor='#c8d8e8')
    ax.bar_label(bars, fmt='%.3f', padding=4, color='#f6f8fa', fontsize=9, fontweight='bold')
    ax.set_ylabel('Score', color='#c8d8e8', fontsize=10)
    ax.set_title('Overall Model Performance', color='#7ed957', fontsize=12, pad=12, fontweight='bold')
    ax.grid(axis='y', color=(1,1,1,0.07), zorder=0)
    for sp in ax.spines.values(): sp.set_color((1,1,1,0.1))
    fig.tight_layout()
    bar_chart = _chart_to_b64(fig)

    # ── Chart 2: Per-class F1 horizontal bar ──
    f1s_sorted, cls_sorted = zip(*sorted(zip(
        [report[c]['f1-score'] for c in classes_all], classes_all)))
    bar_colors2 = ['#7ed957' if v>=0.9 else '#f7b731' if v>=0.75 else '#ff6b6b' for v in f1s_sorted]

    fig2, ax2 = plt.subplots(figsize=(9, max(5, len(cls_sorted)*0.38)))
    fig2.patch.set_facecolor('#0b1929'); ax2.set_facecolor('#0b1929')
    bars2 = ax2.barh([c.title() for c in cls_sorted], f1s_sorted, color=bar_colors2, height=0.6, zorder=3)
    ax2.set_xlim(0, 1.15)
    ax2.bar_label(bars2, fmt='%.3f', padding=4, color='#f6f8fa', fontsize=8)
    ax2.tick_params(colors='#c8d8e8', labelsize=8)
    ax2.set_xlabel('F1 Score', color='#c8d8e8', fontsize=10)
    ax2.set_title('Per-Class F1 Score', color='#7ed957', fontsize=12, pad=12, fontweight='bold')
    ax2.grid(axis='x', color=(1,1,1,0.07), zorder=0)
    for sp in ax2.spines.values(): sp.set_color((1,1,1,0.1))
    fig2.tight_layout()
    f1_chart = _chart_to_b64(fig2)

    # ── Chart 3: Confusion matrix heatmap ──
    cm      = confusion_matrix(y_test, y_pred, labels=classes_all)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig3, ax3 = plt.subplots(figsize=(14, 11))
    fig3.patch.set_facecolor('#0b1929'); ax3.set_facecolor('#0b1929')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGn',
                xticklabels=[c.title() for c in classes_all],
                yticklabels=[c.title() for c in classes_all],
                ax=ax3, linewidths=0.4, linecolor=(1,1,1,0.06),
                cbar_kws={'shrink': 0.7})
    ax3.set_xlabel('Predicted Label', color='#c8d8e8', fontsize=11, labelpad=10)
    ax3.set_ylabel('True Label',      color='#c8d8e8', fontsize=11, labelpad=10)
    ax3.set_title('Normalised Confusion Matrix', color='#7ed957', fontsize=13, pad=14, fontweight='bold')
    ax3.tick_params(colors='#c8d8e8', labelsize=7)
    ax3.figure.axes[-1].tick_params(colors='#c8d8e8')
    fig3.tight_layout()
    cm_chart = _chart_to_b64(fig3)

    # ── Chart 4: Feature importance (mean |coef| across classes) ──
    importances = np.abs(model.coef_).mean(axis=0)
    feat_pairs  = sorted(zip(importances, FEATURE_ORDER), reverse=True)
    imp_vals, imp_names = zip(*feat_pairs)
    feat_colors = ['#7ed957','#5bc0de','#f7b731','#ff6b6b','#c084fc','#fb923c','#38bdf8','#a3e635']

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    fig4.patch.set_facecolor('#0b1929'); ax4.set_facecolor('#0b1929')
    bars4 = ax4.bar(imp_names, imp_vals, color=feat_colors[:len(imp_names)], width=0.55, zorder=3)
    ax4.bar_label(bars4, fmt='%.3f', padding=4, color='#f6f8fa', fontsize=9)
    ax4.tick_params(colors='#c8d8e8', labelsize=9)
    ax4.set_ylabel('Mean |Coefficient|', color='#c8d8e8', fontsize=10)
    ax4.set_title('Feature Importance (Logistic Regression)', color='#7ed957', fontsize=12, pad=12, fontweight='bold')
    ax4.grid(axis='y', color=(1,1,1,0.07), zorder=0)
    for sp in ax4.spines.values(): sp.set_color((1,1,1,0.1))
    fig4.tight_layout()
    feat_chart = _chart_to_b64(fig4)

    return render_template(
        'results.html',
        accuracy=accuracy,
        train_acc=train_acc,
        macro=macro,
        weighted=weighted,
        bar_chart=bar_chart,
        f1_chart=f1_chart,
        cm_chart=cm_chart,
        feat_chart=feat_chart,
        class_table=class_table,
        dataset_stats=dataset_stats,
        total_samples=len(y_test)
    )


if __name__ == '__main__':
    app.run(debug=False)
