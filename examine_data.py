# examine_data.py
import pandas as pd
import os
import numpy as np

print("=" * 80)
print("EXAMINING CIC-IDS-2017 DATASET STRUCTURE")
print("=" * 80)

# لیست فایل‌های CSV
data_dir = "data"
csv_files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv", 
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

results = {}

for filename in csv_files:
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"⚠ File not found: {filename}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {filename}")
    print(f"Size: {os.path.getsize(filepath) / 1024**2:.2f} MB")
    print(f"{'='*60}")
    
    try:
        # سعی با encodingهای مختلف
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                # فقط ۱۰۰۰ ردیف برای نمونه بخوان
                df = pd.read_csv(filepath, nrows=1000, encoding=encoding)
                print(f"✓ Successfully read with {encoding} encoding")
                
                # اطلاعات پایه
                print(f"Shape: {df.shape}")
                print(f"Columns: {len(df.columns)}")
                
                # نمایش ۵ ستون اول و آخر
                print(f"\nFirst 5 columns: {df.columns[:5].tolist()}")
                print(f"Last 5 columns: {df.columns[-5:].tolist()}")
                
                # پیدا کردن ستون label
                label_col = None
                for col in df.columns:
                    if 'label' in col.lower() or 'Label' in col:
                        label_col = col
                        break
                
                if label_col:
                    print(f"\nLabel column: '{label_col}'")
                    print(f"Unique labels ({df[label_col].nunique()}):")
                    unique_counts = df[label_col].value_counts()
                    for label, count in unique_counts.head(10).items():
                        percentage = (count / len(df)) * 100
                        print(f"  '{label}': {count} ({percentage:.1f}%)")
                    
                    if len(unique_counts) > 10:
                        print(f"  ... and {len(unique_counts) - 10} more")
                else:
                    print("\n✗ No obvious label column found")
                    # بررسی آخرین ستون
                    last_col = df.columns[-1]
                    print(f"Last column '{last_col}' sample values:")
                    print(f"  Type: {df[last_col].dtype}")
                    print(f"  Sample: {df[last_col].unique()[:5]}")
                
                # اطلاعات data types
                print(f"\nData types distribution:")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    print(f"  {dtype}: {count} columns")
                
                # بررسی مقادیر missing
                missing_total = df.isnull().sum().sum()
                if missing_total > 0:
                    print(f"\n⚠ Missing values: {missing_total}")
                
                # ذخیره نتایج
                results[filename] = {
                    'shape': df.shape,
                    'encoding': encoding,
                    'label_col': label_col,
                    'num_labels': df[label_col].nunique() if label_col else 0,
                    'sample_labels': df[label_col].unique()[:5].tolist() if label_col else []
                }
                
                # نمایش ۳ ردیف نمونه
                print(f"\nSample data (first 3 rows):")
                print(df.head(3))
                
                break  # اگر موفق بودیم، ادامه نده
                
            except Exception as e:
                print(f"✗ Failed with {encoding}: {str(e)[:100]}...")
                continue
        
    except Exception as e:
        print(f"✗ Error processing file: {str(e)}")

# خلاصه نتایج
print(f"\n{'='*80}")
print("SUMMARY OF ALL FILES")
print(f"{'='*80}")

for filename, info in results.items():
    print(f"\n{filename}:")
    print(f"  Rows: {info['shape'][0]:,}")
    print(f"  Columns: {info['shape'][1]}")
    print(f"  Encoding: {info['encoding']}")
    print(f"  Label column: {info['label_col']}")
    if info['label_col']:
        print(f"  Number of unique labels: {info['num_labels']}")
        print(f"  Sample labels: {info['sample_labels']}")

print(f"\n{'='*80}")
print("RECOMMENDED PREPROCESSING STEPS:")
print(f"{'='*80}")
print("1. Use 'latin1' encoding for all files")
print("2. Expected label column: 'Label'")
print("3. Steps:")
print("   a. Load each CSV with latin1 encoding")
print("   b. Clean column names (strip whitespace)")
print("   c. Handle missing values")
print("   d. Convert labels to numeric (LabelEncoder)")
print("   e. Standardize numeric features")
print("   f. Split into train/test (80/20)")

# پیشنهاد کد برای ادامه
print(f"\n{'='*80}")
print("NEXT STEPS:")
print(f"{'='*80}")
print("1. Run: python data/preprocess.py")
print("2. Then run: python train.py")