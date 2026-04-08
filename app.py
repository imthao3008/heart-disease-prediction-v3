import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling  import SMOTE

# ── Cấu hình trang ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dự đoán Bệnh Tim | UCI Heart Disease",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS tùy chỉnh ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .label { color: #a0a8c8; font-size: 13px; margin-bottom: 6px; }
    .metric-card .value { color: #e8ecff; font-size: 26px; font-weight: 700; }
    .metric-card .sub   { color: #6b7ab8; font-size: 12px; margin-top: 4px; }
    .risk-high {
        background: linear-gradient(135deg, #3d1a1a, #5c2020);
        border: 2px solid #e05555;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #0d2e1a, #14432a);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .risk-gray {
        background: linear-gradient(135deg, #2a2510, #3d3515);
        border: 2px solid #f39c12;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .confidence-bar-wrap {
        background: #1a1f35;
        border: 1px solid #2a2f4a;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .confidence-bar-label {
        color: #a0a8c8;
        font-size: 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }
    .confidence-bar-track {
        background: #2a2f4a;
        border-radius: 6px;
        height: 14px;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    .gray-zone-warning {
        background: linear-gradient(135deg, #2a2510, #3d3515);
        border: 1px solid #f39c12;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #f5c842;
        margin: 10px 0;
    }
    .ood-warning {
        background: linear-gradient(135deg, #1a1020, #2a1535);
        border: 1px solid #9b59b6;
        border-left: 4px solid #9b59b6;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #c39bd3;
        margin: 10px 0;
    }
    .section-title {
        color: #7c8fdb;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2a2f4a;
    }
    .stTabs [data-baseweb="tab"] { color: #a0a8c8; font-size: 14px; }
    .stTabs [aria-selected="true"] { color: #7c8fdb; border-bottom: 2px solid #7c8fdb; }
    .info-box {
        background: #1a1f35;
        border-left: 3px solid #7c8fdb;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #c0c8e8;
        margin: 8px 0;
    }
    .feature-tag {
        display: inline-block;
        background: #1e2540;
        border: 1px solid #3d4a7a;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        color: #8a9bd4;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

SEED = 42
BG   = '#0f1117'
COLORS = ['#7c8fdb','#e05555','#2ecc71','#f39c12','#9b59b6']
MODEL_COLORS = {
    'Logistic Regression': '#7c8fdb',
    'SVM':                 '#e05555',
    'Decision Tree':       '#f39c12',
    'Random Forest':       '#2ecc71',
    'XGBoost':             '#9b59b6'
}

NUM_FEATURES = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
BIN_FEATURES = ['sex', 'fbs', 'exang']
CAT_FEATURES = ['cp', 'restecg', 'slope', 'thal']

BEST_PARAMS = {
    'Logistic Regression': {'C': 0.1,  'penalty': 'l2',      'solver': 'liblinear'},
    'SVM':                 {'C': 1,    'gamma': 'scale'},
    'Decision Tree':       {'max_depth': 5, 'min_samples_split': 10, 'criterion': 'entropy'},
    'Random Forest':       {'n_estimators': 300, 'max_depth': 20, 'max_features': 'sqrt'},
    'XGBoost':             {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8}
}

# ── Load & cache dữ liệu + mô hình ────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Đang huấn luyện 5 mô hình ML...")
def load_and_train():
    df = pd.read_csv('heart_disease_uci.csv')
    df['target'] = (df['num'] > 0).astype(int)

    X = df.drop(columns=['id', 'dataset', 'num', 'target'])
    y = df['target']

    X_proc = X.copy()
    for col in ['fbs', 'exang']:
        X_proc[col] = X_proc[col].map({True: 1, False: 0, 'True': 1, 'False': 0})
    X_proc['sex'] = X_proc['sex'].map({'Male': 1, 'Female': 0})
    X_proc = pd.get_dummies(X_proc, columns=CAT_FEATURES, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=SEED, stratify=y
    )

    imp_num = SimpleImputer(strategy='median')
    num_cols = [c for c in NUM_FEATURES if c in X_train.columns]
    X_train[num_cols] = imp_num.fit_transform(X_train[num_cols])
    X_test[num_cols]  = imp_num.transform(X_test[num_cols])

    imp_cat = SimpleImputer(strategy='most_frequent')
    other_cols = [c for c in X_train.columns if c not in num_cols]
    X_train[other_cols] = imp_cat.fit_transform(X_train[other_cols])
    X_test[other_cols]  = imp_cat.transform(X_test[other_cols])

    scaler = StandardScaler()
    X_train_sc = X_train.copy()
    X_test_sc  = X_test.copy()
    X_train_sc[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_sc[num_cols]  = scaler.transform(X_test[num_cols])

    smote = SMOTE(random_state=SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

    feature_names = list(X_test_sc.columns)
    X_train_arr = X_train_sm.values
    X_test_arr  = X_test_sc.values

    # Huấn luyện 5 mô hình với best params
    models_def = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED,
                                                   **BEST_PARAMS['Logistic Regression']),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=SEED,
                                   **BEST_PARAMS['SVM']),
        'Decision Tree':       DecisionTreeClassifier(random_state=SEED,
                                                      **BEST_PARAMS['Decision Tree']),
        'Random Forest':       RandomForestClassifier(random_state=SEED,
                                                      **BEST_PARAMS['Random Forest']),
        'XGBoost':             XGBClassifier(eval_metric='logloss', use_label_encoder=False,
                                             random_state=SEED, verbosity=0,
                                             **BEST_PARAMS['XGBoost'])
    }

    best_models = {}
    test_results = {}
    for name, model in models_def.items():
        model.fit(X_train_arr, y_train_sm)
        best_models[name] = model
        y_pred = model.predict(X_test_arr)
        y_prob = model.predict_proba(X_test_arr)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        test_results[name] = {
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred),
            'F1-score':  f1_score(y_test, y_pred),
            'AUC-ROC':   roc_auc_score(y_test, y_prob),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            'FN_rate': fn / (fn + tp),
            'y_prob': y_prob, 'y_pred': y_pred
        }

    # SHAP values
    shap_values_dict = {}
    for name, model in best_models.items():
        if name in ['Random Forest', 'XGBoost', 'Decision Tree']:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test_arr)
            if isinstance(sv, list): sv = sv[1]
            sv = np.array(sv, dtype=float)
            if sv.ndim == 3: sv = sv[:, :, 1]
        else:
            background = shap.sample(X_train_arr, 100, random_state=SEED)
            explainer  = shap.KernelExplainer(model.predict_proba, background)
            sv = explainer.shap_values(X_test_arr[:50], nsamples=100)
            if isinstance(sv, list): sv = sv[1]
            sv = np.array(sv, dtype=float)
            if sv.ndim == 3: sv = sv[:, :, 1]
        shap_values_dict[name] = sv

    # Fix LR direction
    sv_lr = shap_values_dict['Logistic Regression']
    top1  = np.argmax(np.abs(sv_lr).mean(axis=0))
    if best_models['Logistic Regression'].coef_[0][top1] * sv_lr[:, top1].mean() < 0:
        shap_values_dict['Logistic Regression'] = -sv_lr

    return (df, best_models, test_results, shap_values_dict,
            feature_names, X_test_arr, X_test_sc, y_test,
            imp_num, imp_cat, scaler, num_cols, other_cols, X_proc.columns.tolist())


def preprocess_input(user_input, imp_num, imp_cat, scaler, num_cols, other_cols, all_columns):
    """Xử lý đầu vào người dùng theo đúng pipeline huấn luyện."""
    row = pd.DataFrame([user_input])

    # Binary encode
    for col in ['fbs', 'exang']:
        row[col] = row[col].map({True: 1, False: 0, 1: 1, 0: 0})
    row['sex'] = row['sex'].map({'Male': 1, 'Female': 0})

    # OHE
    row = pd.get_dummies(row, columns=CAT_FEATURES, drop_first=False)

    # Align columns
    for col in all_columns:
        if col not in row.columns:
            row[col] = 0
    row = row[[c for c in all_columns if c in row.columns]]

    # Impute
    nc = [c for c in num_cols if c in row.columns]
    oc = [c for c in row.columns if c not in nc]
    row[nc] = imp_num.transform(row[nc])
    row[oc] = imp_cat.transform(row[oc])

    # Scale
    row_sc = row.copy()
    row_sc[nc] = scaler.transform(row[nc])

    return row_sc.values


def fig_to_buf(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# ── LOAD ────────────────────────────────────────────────────────────────────────
(df, best_models, test_results, shap_values_dict,
 feature_names, X_test_arr, X_test_sc, y_test,
 imp_num, imp_cat, scaler, num_cols, other_cols, all_columns) = load_and_train()

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Disease AI")
    st.markdown("**Đồ án tốt nghiệp** · Vật lý kỹ thuật")
    st.markdown("---")
    st.markdown('<div class="section-title">📋 Thông tin bệnh nhân</div>', unsafe_allow_html=True)

    age      = st.slider("Tuổi", 20, 80, 54)
    sex      = st.selectbox("Giới tính", ["Male", "Female"])
    cp       = st.selectbox("Loại đau ngực (cp)", [
        "asymptomatic", "typical angina", "atypical angina", "non-anginal"
    ])
    trestbps = st.slider("Huyết áp nghỉ (mmHg)", 80, 210, 130)
    chol     = st.slider("Cholesterol (mg/dL)", 100, 610, 240)
    fbs      = st.radio("Đường huyết lúc đói > 120 mg/dL", [False, True],
                        format_func=lambda x: "Có" if x else "Không")

    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Kết quả xét nghiệm</div>', unsafe_allow_html=True)

    restecg  = st.selectbox("Điện tâm đồ lúc nghỉ", [
        "normal", "lv hypertrophy", "st-t abnormality"
    ])
    thalch   = st.slider("Nhịp tim tối đa (bpm)", 60, 210, 150)
    exang    = st.radio("Đau ngực khi gắng sức", [False, True],
                        format_func=lambda x: "Có" if x else "Không")
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope    = st.selectbox("Độ dốc ST khi gắng sức", ["flat", "upsloping", "downsloping"])
    ca       = st.slider("Số nhánh mạch vành hẹp (ca)", 0, 3, 0)
    thal     = st.selectbox("Kết quả Thalassemia", [
        "normal", "reversable defect", "fixed defect"
    ])

    st.markdown("---")
    selected_model = st.selectbox("🤖 Chọn mô hình dự đoán",
                                   list(best_models.keys()), index=4)

    predict_btn = st.button("🔍 Phân tích & Dự đoán", use_container_width=True, type="primary")

# ── HEADER ──────────────────────────────────────────────────────────────────────
st.markdown("# 🫀 Hệ thống Dự đoán Bệnh Tim Mạch")
st.markdown("**UCI Heart Disease Dataset** · 920 mẫu · 5 mô hình ML · Phân tích SHAP")
st.markdown("---")

# ── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Dự đoán bệnh nhân",
    "📊 So sánh mô hình",
    "🧠 Phân tích SHAP",
    "📈 Dữ liệu & EDA"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DỰ ĐOÁN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    user_input = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': float(ca), 'thal': thal
    }

    # Tóm tắt thông số nhập vào
    st.markdown('<div class="section-title">📋 Thông số bệnh nhân đã nhập</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    params_display = [
        ("Tuổi", age, "năm"),
        ("Giới tính", "Nam" if sex == "Male" else "Nữ", ""),
        ("Cholesterol", chol, "mg/dL"),
        ("Huyết áp", trestbps, "mmHg"),
        ("Nhịp tim tối đa", thalch, "bpm"),
        ("ST Depression", oldpeak, "mm"),
        ("Nhánh hẹp (ca)", ca, ""),
    ]
    for col_ui, (label, val, unit) in zip([c1,c2,c3,c4,c5,c6,c7], params_display):
        col_ui.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{unit}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    if predict_btn:
        X_input = preprocess_input(user_input, imp_num, imp_cat, scaler,
                                   num_cols, other_cols, all_columns)
        model   = best_models[selected_model]
        prob    = model.predict_proba(X_input)[0][1]
        pred    = int(prob >= 0.5)

        # ── Phân loại vùng tin cậy ─────────────────────────────────────────────
        is_gray_zone = 0.4 <= prob <= 0.6

        # Phân loại mức tin cậy
        dist_from_center = abs(prob - 0.5)
        if dist_from_center >= 0.35:
            confidence_label = "Rất cao"
            confidence_color = "#2ecc71"
            confidence_pct   = 95
        elif dist_from_center >= 0.25:
            confidence_label = "Cao"
            confidence_color = "#7c8fdb"
            confidence_pct   = 80
        elif dist_from_center >= 0.15:
            confidence_label = "Trung bình"
            confidence_color = "#f39c12"
            confidence_pct   = 60
        else:
            confidence_label = "Thấp — Vùng xám"
            confidence_color = "#e05555"
            confidence_pct   = 35

        # Kiểm tra out-of-distribution
        ood_warnings = []
        if chol > 603:   ood_warnings.append(f"Cholesterol = {chol} mg/dL (max dataset: 603)")
        if trestbps > 200: ood_warnings.append(f"Huyết áp = {trestbps} mmHg (max dataset: 200)")
        if thalch > 202: ood_warnings.append(f"Nhịp tim tối đa = {thalch} bpm (max dataset: 202)")
        if oldpeak > 6.2: ood_warnings.append(f"ST Depression = {oldpeak} mm (max dataset: 6.2)")
        if age < 29:     ood_warnings.append(f"Tuổi = {age} (min dataset: 29)")
        if age > 77:     ood_warnings.append(f"Tuổi = {age} (max dataset: 77)")

        # ── Kết quả chính ──────────────────────────────────────────────────────
        col_res, col_gauge, col_info = st.columns([1, 1, 1])

        with col_res:
            if is_gray_zone:
                st.markdown(f"""
                <div class="risk-gray">
                    <div style="font-size:48px">⚡</div>
                    <div style="color:#f39c12; font-size:20px; font-weight:700; margin:8px 0">
                        VÙNG XÁM
                    </div>
                    <div style="color:#f5c842; font-size:36px; font-weight:800">
                        {prob*100:.1f}%
                    </div>
                    <div style="color:#d4a843; font-size:13px; margin-top:6px">
                        Xác suất mắc bệnh tim
                    </div>
                    <div style="color:#a07830; font-size:11px; margin-top:4px">
                        {'CÓ BỆNH (gần ngưỡng)' if pred==1 else 'KHÔNG BỆNH (gần ngưỡng)'}
                    </div>
                </div>""", unsafe_allow_html=True)
            elif pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div style="font-size:48px">⚠️</div>
                    <div style="color:#e05555; font-size:22px; font-weight:700; margin:8px 0">
                        NGUY CƠ CAO
                    </div>
                    <div style="color:#ff8888; font-size:36px; font-weight:800">
                        {prob*100:.1f}%
                    </div>
                    <div style="color:#cc6666; font-size:13px; margin-top:6px">
                        Xác suất mắc bệnh tim
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div style="font-size:48px">✅</div>
                    <div style="color:#2ecc71; font-size:22px; font-weight:700; margin:8px 0">
                        NGUY CƠ THẤP
                    </div>
                    <div style="color:#5aff9a; font-size:36px; font-weight:800">
                        {prob*100:.1f}%
                    </div>
                    <div style="color:#2ecc71; font-size:13px; margin-top:6px">
                        Xác suất mắc bệnh tim
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Thanh độ tin cậy ───────────────────────────────────────────────
            st.markdown(f"""
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-label">
                    <span>🎯 Độ tin cậy dự đoán</span>
                    <span style="color:{confidence_color}; font-weight:700">
                        {confidence_label}
                    </span>
                </div>
                <div class="confidence-bar-track">
                    <div class="confidence-bar-fill" style="
                        width:{confidence_pct}%;
                        background:linear-gradient(90deg, {confidence_color}88, {confidence_color});
                    "></div>
                </div>
                <div style="display:flex; justify-content:space-between;
                            color:#555577; font-size:10px; margin-top:4px">
                    <span>Thấp</span>
                    <span>Trung bình</span>
                    <span>Cao</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Cảnh báo vùng xám ──────────────────────────────────────────────
            if is_gray_zone:
                st.markdown(f"""
                <div class="gray-zone-warning">
                    <b>⚡ Cảnh báo: Vùng xám ({prob*100:.1f}%)</b><br>
                    Xác suất nằm trong khoảng 40–60% — mô hình <b>không chắc chắn</b>.
                    Kết quả này <b>không nên dùng để ra quyết định lâm sàng</b>
                    mà cần bổ sung thêm xét nghiệm chuyên sâu (ECG, siêu âm tim,
                    nghiệm pháp gắng sức).
                </div>""", unsafe_allow_html=True)

            # ── Cảnh báo out-of-distribution ───────────────────────────────────
            if ood_warnings:
                ood_text = "<br>".join([f"• {w}" for w in ood_warnings])
                st.markdown(f"""
                <div class="ood-warning">
                    <b>🔬 Cảnh báo: Giá trị ngoài phân phối huấn luyện</b><br>
                    {ood_text}<br>
                    <span style="font-size:11px; color:#9b8ab0; margin-top:4px; display:block">
                    Mô hình chưa học từ các giá trị này → dự đoán có thể không đáng tin cậy.
                    </span>
                </div>""", unsafe_allow_html=True)

        with col_gauge:
            # Gauge chart với vùng xám
            fig_g, ax_g = plt.subplots(figsize=(4, 3), subplot_kw=dict(polar=True),
                                        facecolor=BG)
            theta = np.linspace(np.pi, 0, 300)
            # Vùng xanh (0–40%)
            ax_g.fill_between(theta[180:], 0.7, 1.0, color='#2ecc71', alpha=0.3)
            # Vùng vàng xám (40–60%)
            ax_g.fill_between(theta[120:180], 0.7, 1.0, color='#f39c12', alpha=0.4)
            # Vùng đỏ (60–100%)
            ax_g.fill_between(theta[:120], 0.7, 1.0, color='#e05555', alpha=0.3)
            # Kim chỉ
            needle = np.pi - prob * np.pi
            needle_color = '#f39c12' if is_gray_zone else ('white' if pred==0 else '#ff6b6b')
            ax_g.annotate('', xy=(needle, 0.85), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color=needle_color, lw=2.5))
            ax_g.set_ylim(0, 1)
            ax_g.set_yticks([])
            ax_g.set_xticks([np.pi, np.pi*3/4, np.pi/2, np.pi/4, 0])
            ax_g.set_xticklabels(['0%', '25%', '50%', '75%', '100%'],
                                  color='#a0a8c8', fontsize=9)
            ax_g.set_facecolor(BG)
            ax_g.grid(False)
            ax_g.spines['polar'].set_visible(False)
            ax_g.set_theta_zero_location('E')
            ax_g.set_theta_direction(-1)
            ax_g.set_thetamin(0)
            ax_g.set_thetamax(180)
            # Nhãn vùng
            ax_g.text(np.pi*5/6, 0.62, 'Thấp', color='#2ecc71',
                     fontsize=7, ha='center', va='center')
            ax_g.text(np.pi/2,   0.62, 'Xám',  color='#f39c12',
                     fontsize=7, ha='center', va='center')
            ax_g.text(np.pi/6,   0.62, 'Cao',  color='#e05555',
                     fontsize=7, ha='center', va='center')
            ax_g.text(0, 0, f"{prob*100:.0f}%", ha='center', va='center',
                     color=needle_color, fontsize=20, fontweight='bold',
                     transform=ax_g.transData)
            st.pyplot(fig_g, use_container_width=True)
            plt.close()

            # Bảng giải thích xác suất
            st.markdown("""
            <div style="background:#1a1f35; border-radius:8px; padding:10px 14px;
                        font-size:11px; color:#a0a8c8; margin-top:4px">
                <div style="margin-bottom:4px; color:#7c8fdb; font-weight:600">
                    📊 Ý nghĩa xác suất
                </div>
                <div>🟢 &lt;40% — Nguy cơ thấp, tin cậy cao</div>
                <div>🟡 40–60% — Vùng xám, cần xét nghiệm thêm</div>
                <div>🔴 &gt;60% — Nguy cơ cao, tin cậy cao</div>
                <div style="margin-top:6px; color:#555577; font-size:10px">
                    Ngưỡng quyết định: prob ≥ 0.5 → Có bệnh
                </div>
            </div>""", unsafe_allow_html=True)

        with col_info:
            st.markdown(f"""
            <div class="info-box">
                <b>🤖 Mô hình:</b> {selected_model}<br><br>
                <b>📊 Hiệu năng (Test):</b><br>
                • Accuracy: {test_results[selected_model]['Accuracy']:.3f}<br>
                • F1-score: {test_results[selected_model]['F1-score']:.3f}<br>
                • AUC-ROC:  {test_results[selected_model]['AUC-ROC']:.3f}<br>
                • FN Rate:  {test_results[selected_model]['FN_rate']:.1%}
            </div>""", unsafe_allow_html=True)

            # Yếu tố nguy cơ phát hiện
            risk_factors = []
            if chol > 240:   risk_factors.append("⚠️ Cholesterol cao")
            if oldpeak > 1:  risk_factors.append("⚠️ ST depression đáng kể")
            if exang:        risk_factors.append("⚠️ Đau ngực khi gắng sức")
            if cp == "asymptomatic": risk_factors.append("⚠️ Thiếu máu thầm lặng")
            if thalch < 120: risk_factors.append("⚠️ Nhịp tim tối đa thấp")
            if ca > 0:       risk_factors.append(f"⚠️ {ca} nhánh mạch vành hẹp")
            if chol > 603:   risk_factors.append("🔬 Chol ngoài phạm vi dataset")
            if age > 77:     risk_factors.append("🔬 Tuổi ngoài phạm vi dataset")

            if risk_factors:
                st.markdown("**🔴 Yếu tố cần lưu ý:**")
                for rf in risk_factors:
                    color = '#9b59b6' if '🔬' in rf else '#e05555'
                    st.markdown(f"<span style='color:{color}; font-size:13px'>{rf}</span>",
                                unsafe_allow_html=True)

            # Giải thích xác suất
            st.markdown(f"""
            <div class="info-box" style="margin-top:10px; font-size:12px">
                <b>❓ {prob*100:.1f}% được tính như thế nào?</b><br><br>
                300 cây quyết định mỗi cây "vote" một giá trị nhỏ
                → Cộng tất cả lại (raw score)<br>
                → Qua hàm sigmoid:<br>
                <span style="color:#7c8fdb; font-family:monospace">
                &nbsp;&nbsp;P = 1/(1+e⁻ˢ) = {prob:.3f}
                </span>
            </div>""", unsafe_allow_html=True)

        # ── SHAP Waterfall cho bệnh nhân này ───────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">🔬 Giải thích dự đoán (SHAP Waterfall)</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        Biểu đồ dưới đây giải thích tại sao mô hình <b>{selected_model}</b> đưa ra dự đoán này.
        Thanh <span style="color:#e05555">đỏ</span> = đặc trưng làm tăng nguy cơ •
        Thanh <span style="color:#5aabff">xanh</span> = đặc trưng làm giảm nguy cơ
        </div>""", unsafe_allow_html=True)

        try:
            if selected_model in ['Random Forest', 'XGBoost', 'Decision Tree']:
                exp = shap.TreeExplainer(best_models[selected_model])
                sv_single = exp.shap_values(X_input)
                if isinstance(sv_single, list): sv_single = sv_single[1]
                sv_single = np.array(sv_single, dtype=float).flatten()
                base_val  = float(np.array(exp.expected_value).flat[
                    1 if hasattr(exp.expected_value, '__len__') and
                    len(np.array(exp.expected_value).flat) > 1 else 0])
            else:
                background = shap.sample(
                    np.vstack([X_test_arr] * 2)[:100], 100, random_state=SEED)
                exp = shap.KernelExplainer(
                    best_models[selected_model].predict_proba, background)
                sv_raw    = exp.shap_values(X_input, nsamples=100)
                if isinstance(sv_raw, list): sv_raw = sv_raw[1]
                sv_single = np.array(sv_raw, dtype=float).flatten()
                base_val  = float(np.array(exp.expected_value).flat[
                    1 if hasattr(exp.expected_value, '__len__') and
                    len(np.array(exp.expected_value).flat) > 1 else 0])

            # Vẽ waterfall thủ công
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap':    sv_single
            }).reindex(pd.Series(sv_single).abs().sort_values(ascending=False).index)
            top_n    = 12
            shap_top = shap_df.head(top_n).iloc[::-1]

            fig_wf, ax_wf = plt.subplots(figsize=(9, 5), facecolor=BG)
            ax_wf.set_facecolor(BG)
            bar_colors = ['#e05555' if v > 0 else '#5aabff' for v in shap_top['shap']]
            bars = ax_wf.barh(range(len(shap_top)), shap_top['shap'],
                              color=bar_colors, alpha=0.85, height=0.65, edgecolor='none')
            ax_wf.set_yticks(range(len(shap_top)))
            ax_wf.set_yticklabels(shap_top['feature'], color='#c0c8e8', fontsize=10)
            ax_wf.axvline(0, color='#555577', linewidth=1)
            ax_wf.set_xlabel('SHAP value', color='#a0a8c8', fontsize=10)
            ax_wf.set_title(f'Top {top_n} đặc trưng ảnh hưởng đến dự đoán — {selected_model}',
                           color='#e8ecff', fontsize=11, pad=12)
            ax_wf.tick_params(colors='#a0a8c8')
            ax_wf.spines[['top','right','left']].set_visible(False)
            ax_wf.spines['bottom'].set_color('#333355')
            for bar, val in zip(bars, shap_top['shap']):
                ax_wf.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                          f'{val:+.3f}', va='center',
                          ha='left' if val >= 0 else 'right',
                          color='#e8ecff', fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig_wf, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"Không thể tính SHAP cho bệnh nhân này: {e}")

    else:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding:30px">
            <div style="font-size:40px; margin-bottom:12px">👈</div>
            <div style="font-size:16px; color:#c0c8e8">
                Nhập thông số bệnh nhân trong thanh bên trái<br>
                rồi nhấn <b>Phân tích & Dự đoán</b>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SO SÁNH MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Bảng so sánh hiệu năng trên tập Test</div>',
                unsafe_allow_html=True)

    # Bảng kết quả
    rows = []
    for name, r in test_results.items():
        rows.append({
            'Mô hình':   name,
            'Accuracy':  f"{r['Accuracy']:.4f}",
            'Precision': f"{r['Precision']:.4f}",
            'Recall':    f"{r['Recall']:.4f}",
            'F1-score':  f"{r['F1-score']:.4f}",
            'AUC-ROC':   f"{r['AUC-ROC']:.4f}",
            'FN Rate':   f"{r['FN_rate']:.1%}",
        })
    df_table = pd.DataFrame(rows).set_index('Mô hình')
    st.dataframe(df_table, use_container_width=True)

    col_bar, col_roc = st.columns(2)

    with col_bar:
        st.markdown('<div class="section-title">📊 So sánh 5 chỉ số</div>',
                    unsafe_allow_html=True)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']
        model_names = list(test_results.keys())
        x = np.arange(len(metrics))
        width = 0.15

        fig_bar, ax_bar = plt.subplots(figsize=(9, 5), facecolor=BG)
        ax_bar.set_facecolor(BG)
        for i, (name, color) in enumerate(MODEL_COLORS.items()):
            vals = [test_results[name][m] for m in metrics]
            ax_bar.bar(x + i*width - 2*width, vals, width, label=name,
                      color=color, alpha=0.85, edgecolor='none')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metrics, color='#c0c8e8', fontsize=10)
        ax_bar.set_ylim(0.6, 1.0)
        ax_bar.set_ylabel('Score', color='#a0a8c8')
        ax_bar.tick_params(colors='#a0a8c8')
        ax_bar.spines[['top','right']].set_visible(False)
        ax_bar.spines[['bottom','left']].set_color('#333355')
        ax_bar.legend(fontsize=8, framealpha=0.2,
                     labelcolor='white', facecolor='#1a1f35')
        ax_bar.set_title('Hiệu năng 5 mô hình trên tập Test',
                        color='#e8ecff', fontsize=11, pad=10)
        ax_bar.yaxis.grid(True, color='#2a2f4a', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close()

    with col_roc:
        st.markdown('<div class="section-title">📈 ROC Curve</div>',
                    unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax_roc.set_facecolor(BG)
        for name, color in MODEL_COLORS.items():
            y_prob = test_results[name]['y_prob']
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = test_results[name]['AUC-ROC']
            ax_roc.plot(fpr, tpr, color=color, lw=2,
                       label=f"{name[:3]} (AUC={auc:.3f})")
        ax_roc.plot([0,1],[0,1], '--', color='#555577', lw=1)
        ax_roc.set_xlabel('False Positive Rate', color='#a0a8c8')
        ax_roc.set_ylabel('True Positive Rate',  color='#a0a8c8')
        ax_roc.set_title('ROC Curve – 5 mô hình', color='#e8ecff', fontsize=11)
        ax_roc.tick_params(colors='#a0a8c8')
        ax_roc.spines[['top','right']].set_visible(False)
        ax_roc.spines[['bottom','left']].set_color('#333355')
        ax_roc.legend(fontsize=9, framealpha=0.2,
                     labelcolor='white', facecolor='#1a1f35')
        ax_roc.yaxis.grid(True, color='#2a2f4a', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)
        plt.close()

    # Confusion Matrix
    st.markdown("---")
    st.markdown('<div class="section-title">🔲 Confusion Matrix — 5 mô hình</div>',
                unsafe_allow_html=True)

    fig_cm, axes_cm = plt.subplots(1, 5, figsize=(22, 4), facecolor=BG)
    for ax, (name, color) in zip(axes_cm, MODEL_COLORS.items()):
        r  = test_results[name]
        cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                   cmap=sns.light_palette(color, as_cmap=True),
                   linewidths=1, linecolor=BG,
                   annot_kws={'size': 14, 'color': 'white', 'weight': 'bold'},
                   cbar=False)
        ax.set_title(f"{name}\nFN={r['FN']} ({r['FN_rate']:.0%})",
                    color='#e8ecff', fontsize=10, pad=8)
        ax.set_xlabel('Dự đoán', color='#a0a8c8', fontsize=9)
        ax.set_ylabel('Thực tế',  color='#a0a8c8', fontsize=9)
        ax.set_xticklabels(['Không bệnh','Có bệnh'], color='#c0c8e8', fontsize=8)
        ax.set_yticklabels(['Không bệnh','Có bệnh'], color='#c0c8e8',
                           fontsize=8, rotation=0)
        ax.set_facecolor(BG)
        ax.tick_params(colors='#a0a8c8')
    fig_cm.patch.set_facecolor(BG)
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True)
    plt.close()

    # FN Rate comparison
    st.markdown("---")
    st.markdown('<div class="section-title">🏥 FN Rate — Góc nhìn lâm sàng</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>FN Rate (False Negative Rate)</b> = tỷ lệ bệnh nhân có bệnh bị bỏ sót.
    Đây là chỉ số quan trọng nhất trong lâm sàng — FN thấp hơn = ít bỏ sót ca bệnh hơn.
    </div>""", unsafe_allow_html=True)

    fn_data = {n: test_results[n]['FN_rate'] for n in model_names}
    fig_fn, ax_fn = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    ax_fn.set_facecolor(BG)
    bars_fn = ax_fn.bar(fn_data.keys(), fn_data.values(),
                        color=[MODEL_COLORS[n] for n in fn_data],
                        alpha=0.85, edgecolor='none', width=0.5)
    for bar, val in zip(bars_fn, fn_data.values()):
        ax_fn.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                  f'{val:.1%}', ha='center', color='#e8ecff', fontsize=10,
                  fontweight='bold')
    ax_fn.set_ylabel('FN Rate', color='#a0a8c8')
    ax_fn.set_title('False Negative Rate — Bỏ sót ca bệnh (thấp hơn = tốt hơn)',
                   color='#e8ecff', fontsize=11)
    ax_fn.tick_params(colors='#a0a8c8', axis='x', rotation=15)
    ax_fn.tick_params(colors='#a0a8c8', axis='y')
    ax_fn.spines[['top','right']].set_visible(False)
    ax_fn.spines[['bottom','left']].set_color('#333355')
    ax_fn.yaxis.grid(True, color='#2a2f4a', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig_fn, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🧠 Phân tích SHAP — Tầm quan trọng đặc trưng</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>SHAP (SHapley Additive exPlanations)</b> định lượng đóng góp của từng đặc trưng lâm sàng
    vào kết quả dự đoán. SHAP dương → tăng nguy cơ • SHAP âm → giảm nguy cơ.
    </div>""", unsafe_allow_html=True)

    # Mean |SHAP| Bar Plot
    st.markdown("#### 📊 Mean |SHAP| — Top 10 đặc trưng quan trọng nhất")
    fig_shap, axes_shap = plt.subplots(1, 5, figsize=(25, 5.5), facecolor=BG)
    for ax, (name, color) in zip(axes_shap, MODEL_COLORS.items()):
        sv = shap_values_dict[name]
        mean_abs = np.abs(sv).mean(axis=0)
        top10_idx = np.argsort(mean_abs)[-10:]
        top10_vals = mean_abs[top10_idx]
        top10_feats = [feature_names[i] for i in top10_idx]
        ax.set_facecolor(BG)
        ax.barh(range(10), top10_vals, color=color, alpha=0.8, edgecolor='none')
        ax.set_yticks(range(10))
        ax.set_yticklabels(top10_feats, color='#c0c8e8', fontsize=8.5)
        ax.set_title(name, color='#e8ecff', fontsize=10, pad=8)
        ax.set_xlabel('Mean |SHAP|', color='#a0a8c8', fontsize=8)
        ax.tick_params(colors='#a0a8c8')
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color('#333355')
        ax.xaxis.grid(True, color='#2a2f4a', linewidth=0.5)
    fig_shap.patch.set_facecolor(BG)
    plt.suptitle('Mean |SHAP| — Top 10 đặc trưng theo từng mô hình',
                color='#e8ecff', fontsize=12, y=1.01)
    plt.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close()

    # Bảng thứ hạng Top 10 nhất quán
    st.markdown("---")
    st.markdown("#### 🏆 Top 10 đặc trưng nhất quán qua 5 mô hình")
    rankings = {}
    for name, sv in shap_values_dict.items():
        mean_abs = np.abs(np.array(sv, dtype=float)).mean(axis=0)
        n_feats  = min(len(feature_names), len(mean_abs))
        rank_arr = pd.Series(mean_abs[:n_feats],
                             index=feature_names[:n_feats]).rank(ascending=False)
        rankings[name] = rank_arr

    rank_df   = pd.DataFrame(rankings)
    mean_rank = rank_df.mean(axis=1).sort_values()
    top10     = mean_rank.head(10)

    rank_display = pd.DataFrame({
        'Đặc trưng':      top10.index,
        'Avg Rank':       top10.values.round(1),
        'LR':             [rank_df.loc[f,'Logistic Regression'] if f in rank_df.index else '-' for f in top10.index],
        'SVM':            [rank_df.loc[f,'SVM'] if f in rank_df.index else '-' for f in top10.index],
        'DT':             [rank_df.loc[f,'Decision Tree'] if f in rank_df.index else '-' for f in top10.index],
        'RF':             [rank_df.loc[f,'Random Forest'] if f in rank_df.index else '-' for f in top10.index],
        'XGBoost':        [rank_df.loc[f,'XGBoost'] if f in rank_df.index else '-' for f in top10.index],
    })
    rank_display.index = range(1, 11)
    st.dataframe(rank_display, use_container_width=True)

    # Ma trận Spearman
    st.markdown("---")
    st.markdown("#### 🔗 Ma trận tương quan Spearman — Đồng thuận giữa các mô hình")
    st.markdown("""
    <div class="info-box">
    Hệ số Spearman ρ cao → 2 mô hình đồng thuận cao về thứ hạng đặc trưng.
    ρ = 1.0 = hoàn toàn đồng thuận.
    </div>""", unsafe_allow_html=True)

    from scipy.stats import spearmanr
    model_names_shap = list(shap_values_dict.keys())
    spearman_mat = np.zeros((5, 5))
    for i, n1 in enumerate(model_names_shap):
        for j, n2 in enumerate(model_names_shap):
            r1 = rank_df[n1].values
            r2 = rank_df[n2].values
            rho, _ = spearmanr(r1, r2)
            spearman_mat[i, j] = rho

    fig_sp, ax_sp = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax_sp.set_facecolor(BG)
    short_names = ['LR', 'SVM', 'DT', 'RF', 'XGB']
    im = ax_sp.imshow(spearman_mat, cmap='Blues', vmin=0.85, vmax=1.0)
    ax_sp.set_xticks(range(5))
    ax_sp.set_yticks(range(5))
    ax_sp.set_xticklabels(short_names, color='#c0c8e8', fontsize=11)
    ax_sp.set_yticklabels(short_names, color='#c0c8e8', fontsize=11)
    for i in range(5):
        for j in range(5):
            ax_sp.text(j, i, f'{spearman_mat[i,j]:.3f}',
                      ha='center', va='center',
                      color='white' if spearman_mat[i,j] > 0.93 else '#1a1f35',
                      fontsize=11, fontweight='bold')
    ax_sp.set_title('Tương quan Spearman giữa thứ hạng SHAP của 5 mô hình',
                   color='#e8ecff', fontsize=11, pad=12)
    plt.colorbar(im, ax=ax_sp, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig_sp, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📈 Khám phá dữ liệu (EDA)</div>',
                unsafe_allow_html=True)

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.markdown(f"""<div class="metric-card">
        <div class="label">Tổng mẫu</div><div class="value">920</div>
        <div class="sub">4 trung tâm y tế</div></div>""", unsafe_allow_html=True)
    col_stat2.markdown(f"""<div class="metric-card">
        <div class="label">Có bệnh</div>
        <div class="value" style="color:#e05555">{(df['target']==1).sum()}</div>
        <div class="sub">{(df['target']==1).mean():.1%}</div></div>""", unsafe_allow_html=True)
    col_stat3.markdown(f"""<div class="metric-card">
        <div class="label">Không bệnh</div>
        <div class="value" style="color:#2ecc71">{(df['target']==0).sum()}</div>
        <div class="sub">{(df['target']==0).mean():.1%}</div></div>""", unsafe_allow_html=True)
    col_stat4.markdown(f"""<div class="metric-card">
        <div class="label">Đặc trưng</div><div class="value">13</div>
        <div class="sub">→ 22 sau OHE</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.markdown("#### 📊 Phân phối Cholesterol theo nhãn")
        fig_c, ax_c = plt.subplots(figsize=(6, 4), facecolor=BG)
        ax_c.set_facecolor(BG)
        for label, color, name in [(0,'#2ecc71','Không bệnh'),(1,'#e05555','Có bệnh')]:
            data = df[df['target']==label]['chol'].dropna()
            ax_c.hist(data, bins=30, alpha=0.6, color=color, label=name, edgecolor='none')
        ax_c.axvline(240, color='#f39c12', linestyle='--', lw=1.5, label='Ngưỡng 240 mg/dL')
        ax_c.set_xlabel('Cholesterol (mg/dL)', color='#a0a8c8')
        ax_c.set_ylabel('Số bệnh nhân', color='#a0a8c8')
        ax_c.tick_params(colors='#a0a8c8')
        ax_c.spines[['top','right']].set_visible(False)
        ax_c.spines[['bottom','left']].set_color('#333355')
        ax_c.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor='#1a1f35')
        plt.tight_layout()
        st.pyplot(fig_c, use_container_width=True)
        plt.close()

    with col_eda2:
        st.markdown("#### 📊 Phân phối Tuổi theo nhãn")
        fig_a, ax_a = plt.subplots(figsize=(6, 4), facecolor=BG)
        ax_a.set_facecolor(BG)
        for label, color, name in [(0,'#2ecc71','Không bệnh'),(1,'#e05555','Có bệnh')]:
            data = df[df['target']==label]['age'].dropna()
            ax_a.hist(data, bins=20, alpha=0.6, color=color, label=name, edgecolor='none')
        ax_a.set_xlabel('Tuổi', color='#a0a8c8')
        ax_a.set_ylabel('Số bệnh nhân', color='#a0a8c8')
        ax_a.tick_params(colors='#a0a8c8')
        ax_a.spines[['top','right']].set_visible(False)
        ax_a.spines[['bottom','left']].set_color('#333355')
        ax_a.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor='#1a1f35')
        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=True)
        plt.close()

    col_eda3, col_eda4 = st.columns(2)

    with col_eda3:
        st.markdown("#### 📊 Tỷ lệ bệnh theo loại đau ngực")
        cp_rate = df.groupby('cp')['target'].mean().sort_values(ascending=True)
        fig_cp, ax_cp = plt.subplots(figsize=(6, 4), facecolor=BG)
        ax_cp.set_facecolor(BG)
        colors_cp = ['#2ecc71' if v < 0.5 else '#e05555' for v in cp_rate.values]
        ax_cp.barh(cp_rate.index, cp_rate.values, color=colors_cp, alpha=0.85, edgecolor='none')
        ax_cp.axvline(0.5, color='#f39c12', linestyle='--', lw=1.5)
        ax_cp.set_xlabel('Tỷ lệ có bệnh', color='#a0a8c8')
        ax_cp.tick_params(colors='#a0a8c8')
        ax_cp.spines[['top','right']].set_visible(False)
        ax_cp.spines[['bottom','left']].set_color('#333355')
        for i, (v, label) in enumerate(zip(cp_rate.values, cp_rate.index)):
            ax_cp.text(v+0.01, i, f'{v:.1%}', va='center', color='#e8ecff', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_cp, use_container_width=True)
        plt.close()

    with col_eda4:
        st.markdown("#### 📊 Oldpeak theo nhãn (Boxplot)")
        fig_op, ax_op = plt.subplots(figsize=(6, 4), facecolor=BG)
        ax_op.set_facecolor(BG)
        d0 = df[df['target']==0]['oldpeak'].dropna()
        d1 = df[df['target']==1]['oldpeak'].dropna()
        bp = ax_op.boxplot([d0, d1], patch_artist=True,
                          medianprops={'color': 'white', 'linewidth': 2},
                          whiskerprops={'color': '#a0a8c8'},
                          capprops={'color': '#a0a8c8'},
                          flierprops={'marker': 'o', 'markerfacecolor': '#555577',
                                     'markersize': 3, 'alpha': 0.5})
        bp['boxes'][0].set_facecolor('#2ecc71'); bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('#e05555'); bp['boxes'][1].set_alpha(0.5)
        ax_op.set_xticklabels(['Không bệnh', 'Có bệnh'], color='#c0c8e8')
        ax_op.set_ylabel('ST Depression (mm)', color='#a0a8c8')
        ax_op.axhline(1, color='#f39c12', linestyle='--', lw=1.5,
                     label='Ngưỡng lâm sàng 1mm')
        ax_op.tick_params(colors='#a0a8c8')
        ax_op.spines[['top','right']].set_visible(False)
        ax_op.spines[['bottom','left']].set_color('#333355')
        ax_op.legend(fontsize=9, framealpha=0.2, labelcolor='white', facecolor='#1a1f35')
        plt.tight_layout()
        st.pyplot(fig_op, use_container_width=True)
        plt.close()

    # Tỷ lệ thiếu
    st.markdown("---")
    st.markdown("#### 📋 Tỷ lệ giá trị thiếu theo đặc trưng")
    missing = df.drop(columns=['id','dataset','num','target']).isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) > 0:
        fig_miss, ax_miss = plt.subplots(figsize=(8, 3), facecolor=BG)
        ax_miss.set_facecolor(BG)
        colors_m = ['#e05555' if v > 50 else '#f39c12' if v > 10 else '#7c8fdb'
                   for v in missing.values]
        ax_miss.barh(missing.index, missing.values, color=colors_m,
                    alpha=0.85, edgecolor='none')
        ax_miss.set_xlabel('Tỷ lệ thiếu (%)', color='#a0a8c8')
        ax_miss.tick_params(colors='#a0a8c8')
        ax_miss.spines[['top','right']].set_visible(False)
        ax_miss.spines[['bottom','left']].set_color('#333355')
        for i, v in enumerate(missing.values):
            ax_miss.text(v+0.3, i, f'{v:.1f}%', va='center',
                        color='#e8ecff', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_miss, use_container_width=True)
        plt.close()

    # Thông số best params
    st.markdown("---")
    st.markdown("#### ⚙️ Siêu tham số tối ưu (GridSearchCV)")
    params_rows = []
    for name, params in BEST_PARAMS.items():
        params_rows.append({'Mô hình': name, 'Siêu tham số tối ưu': str(params)})
    st.dataframe(pd.DataFrame(params_rows).set_index('Mô hình'),
                use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555577; font-size:12px; padding:10px">
    🫀 Heart Disease Prediction System · UCI Heart Disease Dataset (920 samples) ·
    Đồ án Tốt nghiệp · Nguyễn Thị Phương Thảo · MSSV 1915190 · ĐH Bách Khoa TP.HCM
</div>""", unsafe_allow_html=True)
