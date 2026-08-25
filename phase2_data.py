import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split

sys.path.append(r"d:\assignment college\VLSI_PROJECT\WMPC_Stacking_TF2\run_code")
from extract_manual_features import find_regions, change_val, cubic_inter_mean, cubic_inter_std, fea_geom, extract_features

def main():
    print("Loading original data...")
    df = pd.read_pickle(r"d:\assignment college\VLSI_PROJECT\LSWMD.pkl\LSWMD.pkl")

    print("Filtering and formatting...")
    # Follow the preprocessing steps from the reference repo
    # The reference drops some cols and maps failureType
    if 'waferIndex' in df.columns:
        df = df.drop(['waferIndex', 'trianTestLabel', 'lotName'], axis=1, errors='ignore')
    
    df['failureNum'] = df.failureType
    mapping_type = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
    df = df.replace({'failureNum':mapping_type})
    
    # Filter labeled subset
    df_withlabel = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
    
    # Remove abnormal wafer maps with less than 100 dies
    df_withlabel = df_withlabel.drop(df_withlabel[df_withlabel['dieSize'] < 100].index.tolist()).reset_index(drop=True)
    
    print(f"Total labeled wafers: {len(df_withlabel)}")
    y = np.array(df_withlabel['failureNum']).astype(int)
    
    # Binarize and resize wafer maps for CNN
    print("Preparing CNN representation (64x64)...")
    X = df_withlabel.waferMap
    X_binary = [np.where(x <= 1, 0, 1) for x in X]
    X_resize = np.array([resize(x, (64, 64), preserve_range=True, anti_aliasing=False) for x in X_binary])
    X_resize = X_resize.reshape(-1, 64, 64, 1).astype(np.float16)
    
    # Extract Handcrafted Features
    print("Preparing Handcrafted Features (59-dim)... this may take a while.")
    # The extract_features function takes the dataframe and computes everything
    fea_all = extract_features(df_withlabel.copy())
    
    # Train-test split (using stratified split as the repo does in run_Stacking.py)
    # The paper uses 10,000 for test set
    print("Splitting train and test sets...")
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=10000, random_state=42, stratify=y)
    
    print(f"Train size: {len(idx_train)}, Test size: {len(idx_test)}")
    
    # Save the processed data
    os.makedirs(r"d:\assignment college\VLSI_PROJECT\data", exist_ok=True)
    with open(r"d:\assignment college\VLSI_PROJECT\data\X_CNN.pkl", 'wb') as f:
        pickle.dump((X_resize[idx_train], X_resize[idx_test]), f, protocol=4)
    with open(r"d:\assignment college\VLSI_PROJECT\data\X_MFE.pkl", 'wb') as f:
        pickle.dump((fea_all[idx_train], fea_all[idx_test]), f, protocol=4)
    with open(r"d:\assignment college\VLSI_PROJECT\data\y.pkl", 'wb') as f:
        pickle.dump((y[idx_train], y[idx_test]), f, protocol=4)
        
    print("Phase 2 data pipeline completed successfully!")

if __name__ == "__main__":
    main()
