import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

print("=== بدأت عملية المقارنة النهائية  ===")

# قراءة ملف الداتا سيت
df = pd.read_csv(r"C:\Users\HITECH\OneDrive\Desktop\Project\HR_GSR_Stress_Levels.csv")

# تحديد الميزات والهدف
X = df[["HR", "GSR (uS)"]]
y = df["Stress Level"]

# تعريف القائمة المطلوبة بالترتيب
models = {
    "Random Forest (الغابة العشوائية)": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=15), 
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "Naive Bayes (Gaussian NB)": GaussianNB()
}

print("=" * 60)
print("10-Fold Cross Validation Results")
print("=" * 60)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    print(f"{name}")
    print(f"Accuracy = {scores.mean()*100:.2f}%")
    print(f"Std = {scores.std()*100:.2f}%")
    print("-" * 60)

print("=== انتهى التشغيل بنجاح ===")
