import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, auc
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import zipfile
import time
import requests
import shap
import umap

# النصوص والإعدادات الثابتة
GITHUB_URL = "https://raw.githubusercontent.com/midosaad254/sonar-rock-vs-mine/main/Sonar%20data.csv"
TEXTS = {
    "ar": {
        "title": "نظام الكشف عن الألغام والصخور بالسونار",
        "sidebar_settings": "إعدادات النظام",
        "theme": "الثيم",
        "light": "فاتح",
        "dark": "داكن",
        "model_type": "اختر النماذج للتدريب",
        "use_smote": "استخدام SMOTE لمعالجة عدم توازن البيانات",
        "use_umap": "استخدام UMAP لتقليل الأبعاد",
        "remove_outliers": "إزالة القيم الشاذة",
        "data_loading": "جارٍ تحميل البيانات...",
        "data_error": "خطأ في تحميل البيانات. تأكد من الاتصال بالإنترنت.",
        "components": "عدد المكونات بعد UMAP",
        "train_eval": "التدريب والتقييم",
        "start_training": "بدء التدريب",
        "training_progress": "جارٍ تدريب النموذج: {}",
        "training_done": "تم تدريب نموذج {} بنجاح!",
        "time_taken": "الوقت المستغرق: {:.2f} ثانية",
        "train_report": "تقرير التدريب لـ {}",
        "test_report": "تقرير الاختبار لـ {}",
        "saved": "تم حفظ النموذج كـ 'sonar_{}_{}.pkl'",
        "plots": "الرسومات التفاعلية",
        "confusion_matrix": "مصفوفة الارتباك",
        "roc_curve": "منحنى ROC",
        "signal_plot": "إشارة سونار عينة",
        "download": "📥 تنزيل النتائج",
        "download_file": "sonar_results_{}.zip",
        "no_model": "يرجى تدريب النموذج أولاً!",
        "summary": "الملخص",
        "accuracy": "الدقة",
        "predict_new": "التنبؤ ببيانات جديدة",
        "upload_new_data": "ارفع ملف CSV للتنبؤ",
        "predict_button": "تنبؤ",
        "prediction_results": "نتائج التنبؤ",
        "manual_input": "أدخل 60 قيمة مفصولة بفواصل",
        "alarm_mine": "⚠️ تحذير: تم اكتشاف لغم!",
        "no_mine": "لا ألغام تم اكتشافها",
        "eda": "تحليل البيانات الاستكشافي",
    },
    "en": {
        "title": "Sonar Mine vs Rock Detection System",
        "sidebar_settings": "System Settings",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "model_type": "Select Models for Training",
        "use_smote": "Use SMOTE to handle data imbalance",
        "use_umap": "Use UMAP for dimensionality reduction",
        "remove_outliers": "Remove outliers",
        "data_loading": "Loading data...",
        "data_error": "Error loading data. Check your internet connection.",
        "components": "Number of Components after UMAP",
        "train_eval": "Training & Evaluation",
        "start_training": "Start Training",
        "training_progress": "Training model: {}",
        "training_done": "Model {} trained successfully!",
        "time_taken": "Time taken: {:.2f} seconds",
        "train_report": "Training Report for {}",
        "test_report": "Test Report for {}",
        "saved": "Model saved as 'sonar_{}_{}.pkl'",
        "plots": "Interactive Plots",
        "confusion_matrix": "Confusion Matrix",
        "roc_curve": "ROC Curve",
        "signal_plot": "Sample Sonar Signal",
        "download": "📥 Download Results",
        "download_file": "sonar_results_{}.zip",
        "no_model": "Please train the model first!",
        "summary": "Summary",
        "accuracy": "Accuracy",
        "predict_new": "Predict New Data",
        "upload_new_data": "Upload a CSV file for prediction",
        "predict_button": "Predict",
        "prediction_results": "Prediction Results",
        "manual_input": "Enter 60 comma-separated values",
        "alarm_mine": "⚠️ Warning: Mine detected!",
        "no_mine": "No mines detected",
        "eda": "Exploratory Data Analysis",
    }
}

# دالة تحميل البيانات
@st.cache_data
def load_data_from_github(url=GITHUB_URL):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), header=None)
    except requests.RequestException as e:
        st.error(f"{TEXTS[st.session_state.get('lang_key', 'en')]['data_error']} ({str(e)})")
        st.stop()

# دالة استخراج الميزات الإحصائية
def extract_statistical_features(df):
    try:
        X = df.iloc[:, :60]
        X.columns = X.columns.astype(str)
        stats = pd.DataFrame({
            'mean': X.mean(axis=1),
            'std': X.std(axis=1),
            'max': X.max(axis=1),
            'min': X.min(axis=1)
        })
        return pd.concat([X, stats], axis=1)
    except Exception as e:
        st.error(f"Error extracting statistical features: {str(e)}")
        st.stop()

# دالة معالجة البيانات
@st.cache_data
def process_data(sonar_df, use_smote=True, use_umap=True, remove_outliers=True):
    try:
        sonar_df_enhanced = extract_statistical_features(sonar_df)
        X = sonar_df_enhanced.values
        y = sonar_df[60].values
        
        if remove_outliers:
            iso_forest = IsolationForest(random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            X = X[outlier_labels == 1]
            y = y[outlier_labels == 1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if use_umap:
            reducer = umap.UMAP(n_components=10, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train_scaled)
            X_test_reduced = reducer.transform(X_test_scaled)
        else:
            X_train_reduced, X_test_reduced = X_train_scaled, X_test_scaled
            reducer = None  # تحديد reducer كـ None عندما لا يتم استخدام UMAP
        
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_reduced, y_train_encoded)
        else:
            X_train_balanced, y_train_balanced = X_train_reduced, y_train_encoded
        
        sample_signal = X[0]
        return X_train_balanced, X_test_reduced, y_train_balanced, y_test_encoded, label_encoder, scaler, reducer, sample_signal
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()

# دالة تدريب النماذج
@st.cache_resource
def train_model_with_search(X_train, y_train, model_name, class_weights=None):
    try:
        if model_name == "XGBoost" or model_name == "XGBoost (إكس جي بوست)":
            base_model = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
        elif model_name == "Random Forest" or model_name == "Random Forest (الغابة العشوائية)":
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weights)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        elif model_name == "SVM" or model_name == "SVM (آلة المتجهات الداعمة)":
            base_model = SVC(probability=True, random_state=42, class_weight=class_weights)
            param_grid = {'C': [0.1, 1], 'kernel': ['rbf', 'linear']}
        elif model_name == "Logistic Regression" or model_name == "Logistic Regression (الانحدار اللوجستي)":
            base_model = LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights)
            param_grid = {'C': [0.1, 1]}
        
        search = RandomizedSearchCV(base_model, param_grid, n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_
    except Exception as e:
        st.error(f"Error training model {model_name}: {str(e)}")
        return None, None

def save_model(pipeline, model_name, timestamp, lang_key):
    try:
        model_filename = f"sonar_{model_name.split()[0].lower()}_{timestamp}.pkl"
        joblib.dump(pipeline, model_filename)
        return f"{TEXTS[lang_key]['saved'].format(model_name.split()[0].lower(), timestamp)}"
    except Exception as e:
        return f"Error: {str(e)}"

# دوال الرسومات
def plot_roc_curve(models, X_test_pca, y_test_enc, colors, lang_key):
    try:
        fig = go.Figure()
        for i, (model_name, model) in enumerate(models.items()):
            y_test_prob = model.predict_proba(X_test_pca)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_enc, y_test_prob)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{model_name} (AUC: {roc_auc:.2f})", line=dict(color=colors[i % len(colors)])))
        fig.update_layout(title=TEXTS[lang_key]["roc_curve"], xaxis_title="False Positive Rate" if lang_key == "en" else "معدل الإيجابيات الخاطئة", yaxis_title="True Positive Rate" if lang_key == "en" else "معدل الإيجابيات الصحيحة")
        return fig
    except Exception as e:
        st.error(f"Error plotting ROC curve: {str(e)}")
        return None

def plot_confusion_matrix(models, X_test_pca, y_test_enc, colors, lang_key):
    try:
        fig = go.Figure()
        for i, (model_name, model) in enumerate(models.items()):
            y_test_pred = model.predict(X_test_pca)
            cm = confusion_matrix(y_test_enc, y_test_pred)
            fig.add_trace(go.Heatmap(z=cm, x=['Predicted R', 'Predicted M'], y=['Actual R', 'Actual M'], colorscale='Blues', name=model_name, showscale=(i==0)))
        fig.update_layout(title=TEXTS[lang_key]["confusion_matrix"], width=600, height=600)
        return fig
    except Exception as e:
        st.error(f"Error plotting confusion matrix: {str(e)}")
        return None

def plot_shap_values(model, model_name, X_test_pca, lang_key):
    try:
        st.write(f"SHAP Analysis for {model_name} (5 samples)")
        plt.figure()
        
        if "XGBoost" in model_name or "Random Forest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_pca[:5])
            
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_test_pca[:5], plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values, X_test_pca[:5], plot_type="bar", show=False)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test_pca, 5))
            shap_values = explainer.shap_values(X_test_pca[:5])
            
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_test_pca[:5], plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values, X_test_pca[:5], plot_type="bar", show=False)
        
        fig = plt.gcf()
        plt.clf()
        return fig
    except Exception as e:
        st.error(f"Error generating SHAP plot for {model_name}: {str(e)}")
        return None

def plot_signal(sample_signal, lang_key):
    try:
        fig = px.line(x=range(len(sample_signal)), y=sample_signal, title=TEXTS[lang_key]["signal_plot"],
                      labels={'x': 'Time' if lang_key == "en" else 'الزمن', 'y': 'Amplitude' if lang_key == "en" else 'السعة'},
                      color_discrete_sequence=['#00cc96'])
        return fig
    except Exception as e:
        st.error(f"Error plotting signal: {str(e)}")
        return None

# دالة لإنشاء ملف ZIP
def create_zip_file(results, plots, code, timestamp, lang_key):
    try:
        buffer = io.BytesIO()
        zip_filename = TEXTS[lang_key]["download_file"].format(timestamp)
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for name, data in results.items():
                zip_file.writestr(f"{name}.csv", data.to_csv(index=False))
            for name, fig in plots.items():
                img_buffer = io.BytesIO()
                fig.write_image(img_buffer, format='png')
                zip_file.writestr(f"{name}.png", img_buffer.getvalue())
            zip_file.writestr("code.py", code)
        buffer.seek(0)
        return buffer, zip_filename
    except Exception as e:
        st.error(f"Error creating zip file: {str(e)}")
        return None, None

# دالة لمقارنة أداء النماذج
def plot_model_performance(models, X_train, y_train, X_test, y_test, lang_key):
    try:
        model_names = list(models.keys())
        train_accuracies = []
        test_accuracies = []

        for model_name, model in models.items():
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=model_names, y=train_accuracies, name='Training Accuracy', marker_color='#1f77b4'))
        fig.add_trace(go.Bar(x=model_names, y=test_accuracies, name='Test Accuracy', marker_color='#ff7f0e'))

        fig.update_layout(
            title="Model Performance Comparison" if lang_key == "en" else "مقارنة أداء النماذج",
            xaxis_title="Model" if lang_key == "en" else "النموذج",
            yaxis_title="Accuracy" if lang_key == "en" else "الدقة",
            barmode='group'
        )
        return fig
    except Exception as e:
        st.error(f"Error plotting model performance: {str(e)}")
        return None

# دالة لتحليل توزيع البيانات
def plot_feature_distribution(X_train, X_test, lang_key):
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=X_train.flatten(), name='Training Data', marker_color='#1f77b4'))
        fig.add_trace(go.Histogram(x=X_test.flatten(), name='Test Data', marker_color='#ff7f0e'))

        fig.update_layout(
            title="Feature Distribution Before and After Processing" if lang_key == "en" else "توزيع الميزات قبل وبعد المعالجة",
            xaxis_title="Feature Value" if lang_key == "en" else "قيمة الميزة",
            yaxis_title="Count" if lang_key == "en" else "العدد",
            barmode='overlay'
        )
        fig.update_traces(opacity=0.75)
        return fig
    except Exception as e:
        st.error(f"Error plotting feature distribution: {str(e)}")
        return None

# دالة لتحليل SHAP
def plot_shap_summary(model, X_test, model_name, lang_key):
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        fig = shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Summary for {model_name}" if lang_key == "en" else f"ملخص SHAP للنموذج {model_name}")
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"Error generating SHAP summary plot: {str(e)}")
        return None

# دالة لإنشاء ملف ZIP نهائي
def create_final_zip(models, X_train, y_train, X_test, y_test, lang_key):
    try:
        buffer = io.BytesIO()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_filename = f"sonar_final_results_{timestamp}.zip"

        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # حفظ الرسومات
            performance_fig = plot_model_performance(models, X_train, y_train, X_test, y_test, lang_key)
            if performance_fig:
                img_buffer = io.BytesIO()
                performance_fig.write_image(img_buffer, format='png')
                zip_file.writestr("model_performance.png", img_buffer.getvalue())

            feature_dist_fig = plot_feature_distribution(X_train, X_test, lang_key)
            if feature_dist_fig:
                img_buffer = io.BytesIO()
                feature_dist_fig.write_image(img_buffer, format='png')
                zip_file.writestr("feature_distribution.png", img_buffer.getvalue())

            for model_name, model in models.items():
                shap_fig = plot_shap_summary(model, X_test, model_name, lang_key)
                if shap_fig:
                    img_buffer = io.BytesIO()
                    shap_fig.savefig(img_buffer, format='png')
                    zip_file.writestr(f"shap_summary_{model_name}.png", img_buffer.getvalue())

            # حفظ النتائج
            results = {}
            for model_name, model in models.items():
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                results[model_name] = {"Training Accuracy": train_acc, "Test Accuracy": test_acc}

            results_df = pd.DataFrame(results).T
            zip_file.writestr("model_results.csv", results_df.to_csv(index=False))

        buffer.seek(0)
        return buffer, zip_filename
    except Exception as e:
        st.error(f"Error creating final zip file: {str(e)}")
        return None, None

def apply_css(lang_key, theme):
    direction = "rtl" if lang_key == "ar" else "ltr"
    alignment = "right" if lang_key == "ar" else "left"
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    body {{font-family: 'Tajawal', sans-serif; direction: {direction}; text-align: {alignment}; transition: background-color 0.5s;
           background-color: {'#eef2f7' if theme == TEXTS[lang_key]['light'] else '#2c3e50'}; color: {'#2c3e50' if theme == TEXTS[lang_key]['light'] else '#eef2f7'};}}
    .stButton>button {{background: linear-gradient(90deg, #4CAF50, #45a049); color: white; border-radius: 10px; padding: 12px 24px; border: none; transition: all 0.3s ease;}}
    .stButton>button:hover {{background: linear-gradient(90deg, #45a049, #4CAF50); transform: scale(1.05);}}
    .stTabs {{background: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 6px 12px rgba(0,0,0,0.1); margin-bottom: 20px;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    # الشريط الجانبي
    st.sidebar.header("Language / اللغة")
    language = st.sidebar.selectbox("Choose Language / اختر اللغة", ["Arabic / العربية", "English / الإنجليزية"])
    lang_key = "ar" if language.startswith("Arabic") else "en"
    st.session_state['lang_key'] = lang_key

    st.sidebar.header(TEXTS[lang_key]["sidebar_settings"])
    theme = st.sidebar.selectbox(TEXTS[lang_key]["theme"], [TEXTS[lang_key]["light"], TEXTS[lang_key]["dark"]])
    model_options = ["XGBoost", "Random Forest", "SVM", "Logistic Regression"] if lang_key == "en" else ["XGBoost (إكس جي بوست)", "Random Forest (الغابة العشوائية)", "SVM (آلة المتجهات الداعمة)", "Logistic Regression (الانحدار اللوجستي)"]
    selected_models = st.sidebar.multiselect(TEXTS[lang_key]["model_type"], model_options, default=[model_options[0]])
    use_smote = st.sidebar.checkbox(TEXTS[lang_key]["use_smote"], value=True)
    use_umap = st.sidebar.checkbox(TEXTS[lang_key]["use_umap"], value=True)
    remove_outliers = st.sidebar.checkbox(TEXTS[lang_key]["remove_outliers"], value=True)

    # تطبيق CSS
    apply_css(lang_key, theme)

    # تحميل ومعالجة البيانات
    with st.spinner(TEXTS[lang_key]["data_loading"]):
        sonar_df = load_data_from_github()
        X_train_bal, X_test_pca, y_train_bal, y_test_enc, label_encoder, scaler, reducer, sample_signal = process_data(
            sonar_df, use_smote, use_umap, remove_outliers
        )

    # عنوان الواجهة
    st.title(TEXTS[lang_key]["title"])

    # تبويبات الواجهة
    tabs = st.tabs([TEXTS[lang_key]["eda"], TEXTS[lang_key]["train_eval"], TEXTS[lang_key]["plots"], TEXTS[lang_key]["summary"], TEXTS[lang_key]["predict_new"]])

    # تبويب تحليل البيانات المسبق (EDA)
    with tabs[0]:
        st.subheader(TEXTS[lang_key]["eda"])
        fig_dist = px.histogram(sonar_df[60], title="Label Distribution" if lang_key == "en" else "توزيع التسميات",
                               labels={'value': 'Label' if lang_key == "en" else 'التسمية', 'count': 'Count' if lang_key == "en" else 'العدد'})
        st.plotly_chart(fig_dist)
        fig_box = px.box(sonar_df.iloc[:, :60], title="Feature Distribution" if lang_key == "en" else "توزيع الميزات")
        st.plotly_chart(fig_box)
        if use_umap and reducer is not None:
            sonar_df_enhanced = extract_statistical_features(sonar_df)
            X_umap = reducer.transform(scaler.transform(sonar_df_enhanced))
            fig_umap = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=sonar_df[60],
                                 title="UMAP Visualization" if lang_key == "en" else "تصور UMAP")
            st.plotly_chart(fig_umap)

    # تبويب التدريب والتقييم
    with tabs[1]:
        st.subheader(TEXTS[lang_key]["train_eval"])
        if st.button(TEXTS[lang_key]["start_training"]):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            models = {}
            model_results = {}
            progress_bar = st.progress(0)
            total_steps = len(selected_models)
            
            for i, model_name in enumerate(selected_models):
                with st.spinner(TEXTS[lang_key]["training_progress"].format(model_name)):
                    start_time = time.time()
                    model, best_params = train_model_with_search(X_train_bal, y_train_bal, model_name)
                    if model is None:
                        continue
                    models[model_name] = model
                    elapsed_time = time.time() - start_time
                    st.success(TEXTS[lang_key]["training_done"].format(model_name))
                    st.write(TEXTS[lang_key]["time_taken"].format(elapsed_time))
                    st.write(f"Best Parameters: {best_params}")
                    progress_bar.progress((i + 1) / total_steps)

                y_train_pred = model.predict(X_train_bal)
                y_test_pred = model.predict(X_test_pca)

                train_report = classification_report(y_train_bal, y_train_pred, target_names=label_encoder.classes_, output_dict=True)
                test_report = classification_report(y_test_enc, y_test_pred, target_names=label_encoder.classes_, output_dict=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{TEXTS[lang_key]['train_report'].format(model_name)}**")
                    st.dataframe(pd.DataFrame(train_report).T.style.format("{:.2f}").background_gradient(cmap='Blues'))
                with col2:
                    st.write(f"**{TEXTS[lang_key]['test_report'].format(model_name)}**")
                    st.dataframe(pd.DataFrame(test_report).T.style.format("{:.2f}").background_gradient(cmap='Greens'))

                model_results[model_name] = {
                    "Accuracy": test_report['accuracy'],
                }

                pipeline = {'model': model, 'label_encoder': label_encoder, 'scaler': scaler, 'reducer': reducer}
                st.success(save_model(pipeline, model_name, timestamp, lang_key))

            # تخزين النماذج ونتائج الاختبار في session_state للاستخدام في تبويبات أخرى
            st.session_state.models = models
            st.session_state.test_report = test_report

    # تبويب الرسومات التفاعلية
    with tabs[2]:
        st.subheader(TEXTS[lang_key]["plots"])
        if 'models' in st.session_state:
            models = st.session_state.models
            plots = {}
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            roc_fig = plot_roc_curve(models, X_test_pca, y_test_enc, colors, lang_key)
            if roc_fig:
                st.plotly_chart(roc_fig)

            cm_fig = plot_confusion_matrix(models, X_test_pca, y_test_enc, colors, lang_key)
            if cm_fig:
                st.plotly_chart(cm_fig)
            
            for model_name, model in models.items():
                shap_fig = plot_shap_values(model, model_name, X_test_pca, lang_key)
                if shap_fig:
                    st.pyplot(shap_fig)
            
            signal_fig = plot_signal(sample_signal, lang_key)
            if signal_fig:
                st.plotly_chart(signal_fig)

            results = {f'test_report_{name.split()[0].lower()}': pd.DataFrame(st.session_state.test_report).T for name in models.keys()}
            with open(__file__, 'r', encoding='utf-8') as f:
                code = f.read()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            zip_buffer, zip_filename = create_zip_file(results, plots, code, timestamp, lang_key)
            if zip_buffer:
                st.download_button(label=TEXTS[lang_key]["download"], data=zip_buffer, file_name=zip_filename, mime="application/zip")
        else:
            st.warning(TEXTS[lang_key]["no_model"])

    # تبويب الملخص
    with tabs[3]:
        st.subheader(TEXTS[lang_key]["summary"])
        if 'models' in st.session_state:
            model_results = {model_name: {"Accuracy": st.session_state.test_report['accuracy']} for model_name in st.session_state.models.keys()}
            comparison_df = pd.DataFrame(model_results).T
            st.dataframe(comparison_df.style.format("{:.4f}").background_gradient(cmap='Blues'))
        else:
            st.warning(TEXTS[lang_key]["no_model"])

    # تبويب التنبؤ ببيانات جديدة
    with tabs[4]:
        st.subheader(TEXTS[lang_key]["predict_new"])
        uploaded_file = st.file_uploader(TEXTS[lang_key]["upload_new_data"], type=["csv"])
        manual_input = st.text_area(TEXTS[lang_key]["manual_input"])
        
        if 'models' in st.session_state and (uploaded_file or manual_input):
            models = st.session_state.models
            if uploaded_file:
                new_data = pd.read_csv(uploaded_file, header=None)
                X_new = extract_statistical_features(new_data).values
            else:
                try:
                    input_values = np.array([float(x.strip()) for x in manual_input.split(',')]).reshape(1, -1)
                    if input_values.shape[1] != 60:
                        st.error("Please enter exactly 60 values." if lang_key == "en" else "يرجى إدخال 60 قيمة بالضبط.")
                        st.stop()
                    X_new = extract_statistical_features(pd.DataFrame(input_values)).values
                except ValueError:
                    st.error("Invalid input format." if lang_key == "en" else "صيغة إدخال غير صالحة.")
                    st.stop()

            X_new_scaled = scaler.transform(X_new)
            X_new_processed = reducer.transform(X_new_scaled) if reducer is not None else X_new_scaled

            if st.button(TEXTS[lang_key]["predict_button"]):
                st.write(f"**{TEXTS[lang_key]['prediction_results']}**")
                for model_name, model in models.items():
                    predictions = model.predict(X_new_processed)
                    pred_labels = label_encoder.inverse_transform(predictions)
                    st.write(f"{model_name}: {pred_labels.tolist()}")
                    if 'M' in pred_labels:
                        st.error(TEXTS[lang_key]["alarm_mine"])
                    else:
                        st.success(TEXTS[lang_key]["no_mine"])
        elif 'models' not in st.session_state:
            st.warning(TEXTS[lang_key]["no_model"])

if __name__ == "__main__":
    main()