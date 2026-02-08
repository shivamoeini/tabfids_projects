# data/preprocess.py
import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

class CICIDSPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def get_file_order(self):
        """ترتیب زمانی فایل‌ها"""
        return [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
    
    def load_single_file(self, filename, sample_frac=0.2):
        """لود یک فایل CSV"""
        filepath = os.path.join(self.data_dir, filename)
        print(f"  Loading {filename}...")
        
        if not os.path.exists(filepath):
            print(f"    ✗ File not found: {filename}")
            return None
        
        try:
            # لود با encoding مناسب
            df = pd.read_csv(filepath, encoding='latin1')
            
            # نمونه‌گیری برای کاهش حجم
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
                print(f"    ✓ Sampled {len(df):,} rows ({sample_frac*100:.0f}%)")
            
            # تمیز کردن نام ستون‌ها
            df.columns = df.columns.str.strip()
            
            # استانداردسازی نام ستون label
            for label_col in ['Label', 'label']:
                if label_col in df.columns:
                    df = df.rename(columns={label_col: 'Label'})
                    break
            
            # اگر Label پیدا نشد
            if 'Label' not in df.columns:
                print(f"    ⚠ 'Label' column not found. Using last column.")
                # استفاده از آخرین ستون به عنوان label
                last_col = df.columns[-1]
                if df[last_col].dtype == 'object':
                    df = df.rename(columns={last_col: 'Label'})
                else:
                    df['Label'] = 'BENIGN'  # default
            
            # حذف ستون‌های مشکل‌ساز
            initial_cols = len(df.columns)
            df = df.loc[:, ~df.columns.duplicated()]  # حذف ستون‌های تکراری
            
            # نمایش اطلاعات labelها
            if 'Label' in df.columns:
                label_counts = df['Label'].value_counts()
                top_labels = label_counts.head(3)
                print(f"    Top labels: {dict(top_labels)}")
            
            print(f"    ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"    ✗ Error loading {filename}: {str(e)}")
            return None
    
    def clean_dataframe(self, df):
        """تمیز کردن DataFrame"""
        if df is None or len(df) == 0:
            return None
        
        # فقط ستون‌های عددی را نگه دار (به جز Label)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # اگر Label ستون نیست، آن را اضافه کن
        if 'Label' in df.columns and 'Label' not in numeric_cols:
            df_numeric = df[numeric_cols + ['Label']]
        else:
            df_numeric = df[numeric_cols]
        
        # جایگزینی inf/-inf با NaN
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        
        # حذف ردیف‌های با NaN
        initial_len = len(df_numeric)
        df_clean = df_numeric.dropna()
        removed = initial_len - len(df_clean)
        
        if removed > 0:
            print(f"    Removed {removed} rows with NaN/Inf values")
        
        return df_clean
    
    def process_all_files(self, sample_frac=0.15):
        """پردازش تمام فایل‌ها"""
        print("Processing CIC-IDS-2017 dataset...")
        print("=" * 70)
        
        file_order = self.get_file_order()
        all_dataframes = []
        
        for i, filename in enumerate(file_order, 1):
            print(f"\n[{i}/{len(file_order)}] Processing {filename}")
            
            # لود فایل
            df = self.load_single_file(filename, sample_frac)
            
            if df is not None:
                # تمیز کردن داده
                df_clean = self.clean_dataframe(df)
                
                if df_clean is not None and len(df_clean) > 0:
                    all_dataframes.append(df_clean)
                
                # آزاد کردن حافظه
                del df
                gc.collect()
        
        if not all_dataframes:
            raise ValueError("No data was loaded successfully")
        
        # ترکیب تمام DataFrameها
        print(f"\n{'='*70}")
        print("Combining all data...")
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        print(f"Combined dataset: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
        print(f"Memory usage: {combined_df.memory_usage().sum() / 1024**2:.2f} MB")
        
        return combined_df
    
    def encode_labels(self, df):
        """کدگذاری labelها"""
        print(f"\n{'='*70}")
        print("Encoding labels...")
        
        # نمایش توزیع labelها
        print("Label distribution before encoding:")
        label_counts = df['Label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  '{label}': {count:,} ({percentage:.2f}%)")
        
        # کدگذاری
        df['Label'] = self.label_encoder.fit_transform(df['Label'])
        
        # نمایش نگاشت
        print(f"\nLabel encoding mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")
        
        return df
    
    def prepare_train_test(self, df):
        """آماده‌سازی train/test split"""
        print(f"\n{'='*70}")
        print("Preparing train/test split...")
        
        # جدا کردن features و labels
        X = df.drop('Label', axis=1).values
        y = df['Label'].values
        
        # ذخیره نام features
        self.feature_names = df.drop('Label', axis=1).columns.tolist()
        
        # نرمال‌سازی
        print("Applying StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)
        
        # تقسیم داده
        print("Splitting into train/test (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        print(f"\nFinal dataset sizes:")
        print(f"  X_train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"  X_test:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
        print(f"  Classes: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed(self, X_train, X_test, y_train, y_test, save_dir="data/preprocessed"):
        """ذخیره داده‌های پیش‌پردازش شده"""
        print(f"\n{'='*70}")
        print("Saving preprocessed data...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # ذخیره به فرمت numpy
        np.save(os.path.join(save_dir, "X_train.npy"), X_train)
        np.save(os.path.join(save_dir, "X_test.npy"), X_test)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train)
        np.save(os.path.join(save_dir, "y_test.npy"), y_test)
        
        # ذخیره transformers
        with open(os.path.join(save_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(save_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # ذخیره metadata
        metadata = {
            'feature_names': self.feature_names,
            'num_features': X_train.shape[1],
            'num_classes': len(self.label_encoder.classes_),
            'class_names': list(self.label_encoder.classes_),
            'class_distribution': dict(zip(*np.unique(y_train, return_counts=True)))
        }
        
        with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nSaved to {save_dir}:")
        print(f"  X_train.npy: {X_train.shape}")
        print(f"  X_test.npy:  {X_test.shape}")
        print(f"  y_train.npy: {y_train.shape}")
        print(f"  y_test.npy:  {y_test.shape}")
        print(f"  scaler.pkl, label_encoder.pkl, metadata.pkl")
        
        # خلاصه
        print(f"\nDataset summary:")
        print(f"  Total samples: {(len(X_train) + len(X_test)):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(self.label_encoder.classes_)}")
        
        print(f"\nClass distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            cls_name = self.label_encoder.inverse_transform([cls])[0]
            percentage = (count / len(y_train)) * 100
            print(f"  Class {cls} ({cls_name:30s}): {count:6,} ({percentage:5.1f}%)")

def main():
    """تابع اصلی"""
    print("=" * 80)
    print("CIC-IDS-2017 PREPROCESSING PIPELINE")
    print("=" * 80)
    
    try:
        # ایجاد preprocessor
        preprocessor = CICIDSPreprocessor(data_dir="data")
        
        # ۱. پردازش تمام فایل‌ها
        print("\n[1/4] Loading and processing files...")
        combined_df = preprocessor.process_all_files(sample_frac=0.15)
        
        # ۲. کدگذاری labels
        print("\n[2/4] Encoding labels...")
        combined_df = preprocessor.encode_labels(combined_df)
        
        # ۳. آماده‌سازی train/test
        print("\n[3/4] Preparing train/test split...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(combined_df)
        
        # ۴. ذخیره نتایج
        print("\n[4/4] Saving results...")
        preprocessor.save_preprocessed(X_train, X_test, y_train, y_test)
        
        print(f"\n{'='*80}")
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        print("\nNext steps:")
        print("1. Run 'python train.py' to train the model")
        print("2. Or run 'python test.py' to test existing model")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nPreprocessing failed. Please check the error messages.")