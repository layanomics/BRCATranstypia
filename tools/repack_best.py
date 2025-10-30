from pathlib import Path
import joblib, os

# big file (works)
SRC = Path('models/model_panel5k_quantile_svm_isotonic.joblib')
# new small file we will create
DST = Path('models/model_panel5k_quantile_svm_best.joblib')
TMP = DST.with_suffix('.tmp')

print('Loading big model...')
model = joblib.load(SRC)

print('Saving compressed copy...')
joblib.dump(model, TMP, compress=('xz', 6))

os.replace(TMP, DST)
print('✅ Done. Size MB =', round(DST.stat().st_size/1e6, 2))
