"""
Data Preprocessing Pipeline for CICIDS2017 Dataset

This module provides reusable preprocessing functions for cleaning and preparing
CICIDS2017 network intrusion detection data for machine learning models.

Functions:
    - load_data: Load CICIDS2017 CSV files with proper encoding
    - clean_data: Handle missing values, inf values, and duplicates
    - encode_labels: Convert attack labels to binary (normal/anomaly)
    - scale_features: Normalize features using StandardScaler or MinMaxScaler
    - split_data: Create train/test splits with stratification
    - preprocess_pipeline: Complete end-to-end preprocessing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class CICIDS2017Preprocessor:
    """
    Preprocessing pipeline for CICIDS2017 dataset.

    Handles common data quality issues:
    - Missing values (NaN)
    - Infinite values (inf, -inf)
    - Duplicate rows
    - Feature scaling
    - Label encoding
    """

    def __init__(self, scaler_type: str = 'standard', random_state: int = 42):
        """
        Initialize preprocessor.

        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
            random_state: Random seed for reproducibility
        """
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.label_mapping = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CICIDS2017 CSV file with proper encoding handling.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        print(f"Loading data from: {file_path.name}")

        # Try UTF-8 encoding first, fall back to latin1
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df

    def clean_data(self,
                   df: pd.DataFrame,
                   remove_duplicates: bool = True,
                   handle_inf: str = 'replace',
                   handle_nan: str = 'drop',
                   inf_replacement: float = None) -> pd.DataFrame:
        """
        Clean dataset by handling duplicates, inf values, and NaN values.

        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_inf: How to handle inf values ('replace', 'drop', 'ignore')
            handle_nan: How to handle NaN values ('drop', 'mean', 'median', 'zero', 'ignore')
            inf_replacement: Value to replace inf with (None = use column max/min)

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)

        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)

        # 1. Remove duplicates
        if remove_duplicates:
            duplicates_before = df_clean.duplicated().sum()
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = duplicates_before
            print(f"Duplicates removed: {duplicates_removed:,}")

        # 2. Drop non-numeric metadata columns (Flow ID, IPs, Timestamp, etc.)
        non_feature_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
        cols_to_drop = [col for col in non_feature_cols if col in df_clean.columns or col.strip() in df_clean.columns]

        # Also check for columns with leading/trailing spaces
        for col in df_clean.columns:
            col_stripped = col.strip()
            if col_stripped in non_feature_cols and col not in cols_to_drop:
                cols_to_drop.append(col)

        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            print(f"Non-numeric metadata columns dropped: {len(cols_to_drop)}")

        # 3. Identify feature and label columns
        label_col = 'Label' if 'Label' in df_clean.columns else None
        if label_col:
            feature_cols = [col for col in df_clean.columns if col != label_col]
        else:
            feature_cols = df_clean.columns.tolist()

        # 3. Handle infinite values
        if handle_inf != 'ignore':
            inf_count = 0
            for col in feature_cols:
                if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    col_inf_count = np.isinf(df_clean[col]).sum()

                    if col_inf_count > 0:
                        inf_count += col_inf_count

                        if handle_inf == 'replace':
                            if inf_replacement is not None:
                                # Replace with specified value
                                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], inf_replacement)
                            else:
                                # Replace inf with column max, -inf with column min
                                finite_values = df_clean[col].replace([np.inf, -np.inf], np.nan)
                                col_max = finite_values.max()
                                col_min = finite_values.min()

                                df_clean[col] = df_clean[col].replace(np.inf, col_max)
                                df_clean[col] = df_clean[col].replace(-np.inf, col_min)

                        elif handle_inf == 'drop':
                            df_clean = df_clean[~np.isinf(df_clean[col])]

            print(f"Infinite values handled: {inf_count:,} (method: {handle_inf})")

        # 4. Handle missing values
        if handle_nan != 'ignore':
            nan_count = df_clean[feature_cols].isnull().sum().sum()

            if nan_count > 0:
                if handle_nan == 'drop':
                    df_clean = df_clean.dropna()
                elif handle_nan == 'mean':
                    for col in feature_cols:
                        if df_clean[col].isnull().any():
                            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif handle_nan == 'median':
                    for col in feature_cols:
                        if df_clean[col].isnull().any():
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif handle_nan == 'zero':
                    df_clean[feature_cols] = df_clean[feature_cols].fillna(0)

                print(f"Missing values handled: {nan_count:,} (method: {handle_nan})")

        # 5. Remove constant features (zero variance)
        constant_features = []
        for col in feature_cols:
            if df_clean[col].nunique() == 1:
                constant_features.append(col)

        if constant_features:
            df_clean = df_clean.drop(columns=constant_features)
            print(f"Constant features removed: {len(constant_features)}")

        final_rows = len(df_clean)
        rows_removed = initial_rows - final_rows

        print(f"Total rows removed: {rows_removed:,}")
        print(f"Final dataset: {final_rows:,} rows, {len(df_clean.columns)} columns")
        print("="*60 + "\n")

        return df_clean

    def encode_labels(self,
                      df: pd.DataFrame,
                      label_col: str = 'Label',
                      binary_encoding: bool = True) -> Tuple[pd.DataFrame, dict]:
        """
        Encode labels for classification.

        Args:
            df: Input DataFrame
            label_col: Name of label column
            binary_encoding: If True, convert to binary (BENIGN=0, all attacks=1)
                            If False, use label encoding for multi-class

        Returns:
            Tuple of (DataFrame with encoded labels, label mapping dict)
        """
        df_encoded = df.copy()

        if label_col not in df_encoded.columns:
            print(f"Warning: Label column '{label_col}' not found")
            return df_encoded, {}

        print("\n" + "="*60)
        print("LABEL ENCODING")
        print("="*60)

        # Show original label distribution
        print("\nOriginal Label Distribution:")
        label_counts = df_encoded[label_col].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count:,}")

        if binary_encoding:
            # Binary: BENIGN = 0 (normal), everything else = 1 (anomaly)
            df_encoded['Label_Binary'] = (df_encoded[label_col] != 'BENIGN').astype(int)

            normal_count = (df_encoded['Label_Binary'] == 0).sum()
            anomaly_count = (df_encoded['Label_Binary'] == 1).sum()

            print("\nBinary Encoding Applied:")
            print(f"  Normal (0): {normal_count:,}")
            print(f"  Anomaly (1): {anomaly_count:,}")

            label_mapping = {
                'BENIGN': 0,
                'ATTACKS': 1
            }

            # Store original labels for reference
            df_encoded['Label_Original'] = df_encoded[label_col]
            df_encoded[label_col] = df_encoded['Label_Binary']
            df_encoded = df_encoded.drop('Label_Binary', axis=1)

        else:
            # Multi-class encoding
            df_encoded['Label_Encoded'] = self.label_encoder.fit_transform(df_encoded[label_col])

            label_mapping = dict(zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_)
            ))

            print("\nMulti-class Encoding Applied:")
            for label, code in sorted(label_mapping.items(), key=lambda x: x[1]):
                count = (df_encoded['Label_Encoded'] == code).sum()
                print(f"  {label} → {code}: {count:,}")

            df_encoded['Label_Original'] = df_encoded[label_col]
            df_encoded[label_col] = df_encoded['Label_Encoded']
            df_encoded = df_encoded.drop('Label_Encoded', axis=1)

        print("="*60 + "\n")
        self.label_mapping = label_mapping

        return df_encoded, label_mapping

    def scale_features(self,
                       X_train: np.ndarray,
                       X_test: np.ndarray = None,
                       feature_names: List[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler or MinMaxScaler.

        Args:
            X_train: Training features
            X_test: Test features (optional)
            feature_names: List of feature names (for reference)

        Returns:
            Tuple of (scaled X_train, scaled X_test) or just scaled X_train
        """
        print("\n" + "="*60)
        print("FEATURE SCALING")
        print("="*60)

        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            print("Using StandardScaler (mean=0, std=1)")
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            print("Using MinMaxScaler (range=[0, 1])")
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        self.feature_names = feature_names

        # Fit on training data and transform
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"Training data scaled: {X_train_scaled.shape}")

        if X_test is not None:
            # Transform test data using fitted scaler
            X_test_scaled = self.scaler.transform(X_test)
            print(f"Test data scaled: {X_test_scaled.shape}")
            print("="*60 + "\n")
            return X_train_scaled, X_test_scaled

        print("="*60 + "\n")
        return X_train_scaled

    def split_data(self,
                   df: pd.DataFrame,
                   label_col: str = 'Label',
                   test_size: float = 0.2,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            label_col: Name of label column
            test_size: Proportion of data for test set
            stratify: Whether to stratify split by labels

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("TRAIN/TEST SPLIT")
        print("="*60)

        # Separate features and labels
        if label_col in df.columns:
            X = df.drop([label_col], axis=1)

            # Drop Label_Original if it exists
            if 'Label_Original' in X.columns:
                X = X.drop('Label_Original', axis=1)

            y = df[label_col]
        else:
            X = df
            y = None

        # Perform split
        if y is not None:
            stratify_col = y if stratify else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_col
            )

            print(f"Training set: {len(X_train):,} samples")
            print(f"Test set: {len(X_test):,} samples")
            print(f"Test size: {test_size*100:.1f}%")
            print(f"Stratified: {stratify}")

            if stratify:
                print("\nLabel distribution preserved:")
                print("Training set:")
                print(y_train.value_counts(normalize=True).sort_index())
                print("Test set:")
                print(y_test.value_counts(normalize=True).sort_index())

            print("="*60 + "\n")
            return X_train, X_test, y_train, y_test

        else:
            # No labels - simple split
            X_train, X_test = train_test_split(
                X,
                test_size=test_size,
                random_state=self.random_state
            )

            print(f"Training set: {len(X_train):,} samples")
            print(f"Test set: {len(X_test):,} samples")
            print("="*60 + "\n")

            return X_train, X_test, None, None

    def preprocess_pipeline(self,
                           file_path: str,
                           test_size: float = 0.2,
                           binary_labels: bool = True,
                           remove_duplicates: bool = True,
                           handle_inf: str = 'replace',
                           handle_nan: str = 'median',
                           scale: bool = True,
                           return_dataframe: bool = False) -> dict:
        """
        Complete end-to-end preprocessing pipeline.

        Args:
            file_path: Path to CICIDS2017 CSV file
            test_size: Proportion for test set
            binary_labels: Binary (anomaly detection) or multi-class classification
            remove_duplicates: Whether to remove duplicate rows
            handle_inf: How to handle infinite values
            handle_nan: How to handle missing values
            scale: Whether to scale features
            return_dataframe: Return DataFrames instead of numpy arrays

        Returns:
            Dictionary containing:
                - X_train, X_test, y_train, y_test
                - scaler, label_mapping
                - feature_names
                - preprocessing_stats
        """
        print("\n" + "="*70)
        print("CICIDS2017 PREPROCESSING PIPELINE")
        print("="*70)

        # 1. Load data
        df = self.load_data(file_path)
        initial_shape = df.shape

        # 2. Clean data
        df_clean = self.clean_data(
            df,
            remove_duplicates=remove_duplicates,
            handle_inf=handle_inf,
            handle_nan=handle_nan
        )

        # 3. Encode labels
        df_encoded, label_mapping = self.encode_labels(
            df_clean,
            binary_encoding=binary_labels
        )

        # 4. Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df_encoded,
            test_size=test_size,
            stratify=True
        )

        # Store feature names
        feature_names = X_train.columns.tolist()

        # 5. Scale features
        if scale:
            X_train_scaled, X_test_scaled = self.scale_features(
                X_train.values,
                X_test.values,
                feature_names=feature_names
            )

            if return_dataframe:
                X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
            else:
                X_train = X_train_scaled
                X_test = X_test_scaled
        else:
            if not return_dataframe:
                X_train = X_train.values
                X_test = X_test.values

        # Compile preprocessing statistics
        stats = {
            'initial_shape': initial_shape,
            'final_shape': (len(df_encoded), len(df_encoded.columns)),
            'rows_removed': initial_shape[0] - len(df_encoded),
            'feature_count': len(feature_names),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_size': test_size,
            'scaler_type': self.scaler_type if scale else None,
            'binary_labels': binary_labels
        }

        print("="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"Initial shape: {initial_shape}")
        print(f"Final shape: {stats['final_shape']}")
        print(f"Features: {stats['feature_count']}")
        print(f"Training samples: {stats['train_samples']:,}")
        print(f"Test samples: {stats['test_samples']:,}")
        print("="*70 + "\n")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'label_mapping': label_mapping,
            'feature_names': feature_names,
            'stats': stats
        }


# Convenience functions

def preprocess_cicids2017(file_path: str, **kwargs) -> dict:
    """
    Quick preprocessing of CICIDS2017 dataset with default settings.

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments passed to preprocess_pipeline

    Returns:
        Dictionary with preprocessed data and metadata
    """
    preprocessor = CICIDS2017Preprocessor()
    return preprocessor.preprocess_pipeline(file_path, **kwargs)


def load_and_clean(file_path: str,
                   remove_duplicates: bool = True,
                   handle_inf: str = 'replace',
                   handle_nan: str = 'median') -> pd.DataFrame:
    """
    Load and clean CICIDS2017 data without splitting or scaling.

    Args:
        file_path: Path to CSV file
        remove_duplicates: Whether to remove duplicates
        handle_inf: How to handle infinite values
        handle_nan: How to handle missing values

    Returns:
        Cleaned DataFrame
    """
    preprocessor = CICIDS2017Preprocessor()
    df = preprocessor.load_data(file_path)
    df_clean = preprocessor.clean_data(
        df,
        remove_duplicates=remove_duplicates,
        handle_inf=handle_inf,
        handle_nan=handle_nan
    )
    return df_clean


if __name__ == "__main__":
    # Example usage
    print("CICIDS2017 Preprocessing Module")
    print("="*70)
    print("\nExample usage:")
    print("""
from src.data.preprocessing import preprocess_cicids2017

# Preprocess dataset
data = preprocess_cicids2017(
    'data/raw/Monday-WorkingHours.pcap_ISCX.csv',
    test_size=0.2,
    binary_labels=True,
    scale=True
)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
    """)
    print("="*70)
