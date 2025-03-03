import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE  # لمعالجة عدم توازن البيانات
from xgboost import XGBClassifier  # تعلم آلي بدلاً من CNN+LSTM
import joblib
import sys
import io

# تغيير ترميز الإخراج إلى UTF-8 لدعم العربية
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# تحميل البيانات
data_path = r"D:\نقل\ai\machine learning 3\projects\SONAR Rock vs Mine Prediction with Python\my\Sonar data.csv"
sonar_df = pd.read_csv(data_path, header=None)

# فصل الميزات والتسميات
X = sonar_df.iloc[:, :60].values
y = sonar_df[60].values

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ترميز التسميات
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# معالجة عدم توازن البيانات باستخدام SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_encoded)

# خيار تحسين الأداء باستخدام GridSearchCV (اختياري، قد يطول وقت التدريب)
use_grid_search = False  # قم بتغيير هذا إلى True إذا أردت تحسين المعلمات

if use_grid_search:
    # تعريف نطاق المعلمات
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # بناء النموذج
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # البحث عن أفضل معلمات
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)

    # استخدام أفضل النموذج
    best_model = grid_search.best_estimator_
    print("أفضل معلمات:", grid_search.best_params_)
else:
    # بناء نموذج تعلم آلي (XGBoost) مع معلمات افتراضية
    best_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_estimators=200,  # زيادة عدد الأشجار
        max_depth=6,  # تحسين العمق
        learning_rate=0.1
    )

# تدريب النموذج النهائي
best_model.fit(X_train_balanced, y_train_balanced)

# تقييم النموذج
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

print("\nتقرير التصنيف على بيانات التدريب:")
print(classification_report(y_train_encoded, y_train_pred, target_names=label_encoder.classes_))
print("\nتقرير التصنيف على بيانات الاختبار:")
print(classification_report(y_test_encoded, y_test_pred, target_names=label_encoder.classes_))

# حساب احتماليات التنبؤ لاختيار عتبة أمثل
y_test_prob = best_model.predict_proba(X_test)[:, 1]  # الحصول على احتمالية الفئة "M" (الألغام)
precision, recall, thresholds = precision_recall_curve(y_test_encoded, y_test_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # تجنب القسمة على صفر
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"العتبة الأمثل للتصنيف: {best_threshold}")

# تقييم النموذج مع العتبة الأمثل
y_test_pred_adjusted = (y_test_prob > best_threshold).astype(int)
print("\nتقرير التصنيف مع العتبة الأمثل على بيانات الاختبار:")
print(classification_report(y_test_encoded, y_test_pred_adjusted, target_names=label_encoder.classes_))

# حساب AUC-PR
auc_pr = auc(recall, precision)
print(f"AUC-PR على بيانات الاختبار: {auc_pr:.4f}")

# حفظ النموذج والمعالجات
pipeline = {
    'model': best_model,
    'label_encoder': label_encoder
}
joblib.dump(pipeline, 'sonar_pipeline_deep_learning_improved.pkl')
print("تم حفظ النموذج المحسن في 'sonar_pipeline_deep_learning_improved.pkl'")