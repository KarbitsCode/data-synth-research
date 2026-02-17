import os
import sys
import logging
import subprocess
import argparse
from datetime import datetime

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basic_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from model import oversample_with_pytorch_gan, oversample_with_ctgan, oversample_with_cond_wgangp
from model.anomaly import add_anomaly_scores
from evaluation.synth_eval import evaluate_synthetic_data, extract_synthetic_tail
from loader.data_loader import UniversalDataLoader
from preprocessor.data_config import DATASET_CONFIG
from benchmark import AblationAnalyzer, AblationStudyManager, CrossDomainBenchmark

try:
    subprocess.run(['brew', 'install', 'libomp'], check=True, capture_output=True)
    logger.info("OpenMP runtime installed successfully")
except subprocess.CalledProcessError:
    logger.warning("Failed to install OpenMP runtime")
except FileNotFoundError:
    logger.warning("Homebrew not found")

# === COMMAND LINE ARGUMENTS ===
# Enable selective experiment execution (e.g., run only model experiments)
# Default behavior: runs all experiments sequentially
parser = argparse.ArgumentParser(description='Run ablation study experiments')
parser.add_argument(
    '--experiment',
    type=str,
    choices=['oversampling', 'model', 'anomaly', 'calibration', 'pairwise', 'all'],
    default='all',
    help='Which experiment to run (default: all)'
)
parser.add_argument(
    '--skip-significance',
    action='store_true',
    help='Skip statistical significance testing'
)
args = parser.parse_args()

# === DATASET SELECTION ===
# Change this to switch datasets
DATASET_NAME = '05_online_payment.csv' # or '01_creditcard.csv', '03_fraud_oracle.csv', '04_bank_account.csv', '05_online_payment.csv'
DATA_ROOT = os.path.join(project_root, "datasets")
logger.info(f"Loading dataset: {DATASET_CONFIG[DATASET_NAME]['name']}")
logger.info(f"Running experiment: {args.experiment}")

# === ANOMALY FEATURE SETTINGS ===
USE_ANOMALY_FEATURES = True
ANOMALY_METHOD = "IsolationForest"  # None, IsolationForest, LOF  (Autoencoder disabled for runtime)
ANOMALY_CONTAMINATION = 0.01

# === FULL ABLATION (SINGLE DATASET) ===
RUN_FULL_ABLATION_SINGLE_DATASET = True
FULL_ABLATION_RANDOM_SEEDS = [42]
FULL_ABLATION_RUN_SIGNIFICANCE = True

# === GAN OVERSAMPLING OPTIMIZATION ===
GAN_TRAIN_MAX_MINORITY_RATIO = 0.5
GAN_HIDDEN_DIM = 64
GAN_NOISE_DIM = 50
GAN_N_CRITIC = 2
GAN_EPOCHS = 100
GAN_BATCH_SIZE = 128
GAN_CACHE_PATH = os.path.join(project_root, "results", "synthetic_cache")
GAN_EARLY_STOPPING = True
GAN_EARLY_STOPPING_PATIENCE = 10
GAN_EARLY_STOPPING_DELTA = 1e-3

# === LOAD AND PREPROCESS DATA ===
loader = UniversalDataLoader(
    DATASET_NAME, 
    project_root=project_root, 
    data_root=DATA_ROOT,
    verbose=True,
    large_data=False # For large datasets e.g. 04_bank_account.csv, use chunking
)

if RUN_FULL_ABLATION_SINGLE_DATASET:
    logger.info(f"Running ablation study: {args.experiment}")
    dataset_label = DATASET_CONFIG[DATASET_NAME]["name"]
    datasets = {dataset_label: DATASET_NAME}
    dataset_tag = DATASET_NAME.replace(".csv", "")
    ablation_mgr = AblationStudyManager(output_dir="results/ablation")
    
    # Generate experiments based on command-line argument
    experiments_to_run = {}
    
    if args.experiment in ['oversampling', 'all']:
        exp_oversampling = ablation_mgr.generate_single_factor_ablation("oversampling")
        ablation_mgr.save_experiments(
            exp_oversampling,
            "ablation_oversampling_single_dataset.json",
            dataset_tag=dataset_tag,
            experiment_tag="ablation_oversampling_single_dataset",
        )
        experiments_to_run['oversampling'] = exp_oversampling
        logger.info(f"Loaded {len(exp_oversampling)} oversampling experiments")
    
    if args.experiment in ['model', 'all']:
        exp_model = ablation_mgr.generate_single_factor_ablation("model")
        ablation_mgr.save_experiments(
            exp_model,
            "ablation_model_single_dataset.json",
            dataset_tag=dataset_tag,
            experiment_tag="ablation_model_single_dataset",
        )
        experiments_to_run['model'] = exp_model
        logger.info(f"Loaded {len(exp_model)} model experiments")
    
    if args.experiment in ['anomaly', 'all']:
        exp_anomaly = ablation_mgr.generate_single_factor_ablation("anomaly_signal")
        ablation_mgr.save_experiments(
            exp_anomaly,
            "ablation_anomaly_single_dataset.json",
            dataset_tag=dataset_tag,
            experiment_tag="ablation_anomaly_single_dataset",
        )
        experiments_to_run['anomaly'] = exp_anomaly
        logger.info(f"Loaded {len(exp_anomaly)} anomaly experiments")
    
    if args.experiment in ['calibration', 'all']:
        exp_calibration = ablation_mgr.generate_single_factor_ablation("calibration")
        ablation_mgr.save_experiments(
            exp_calibration,
            "ablation_calibration_single_dataset.json",
            dataset_tag=dataset_tag,
            experiment_tag="ablation_calibration_single_dataset",
        )
        experiments_to_run['calibration'] = exp_calibration
        logger.info(f"Loaded {len(exp_calibration)} calibration experiments")
    
    if args.experiment in ['pairwise', 'all']:
        exp_pairwise = ablation_mgr.generate_pairwise_ablation("oversampling", "model")
        ablation_mgr.save_experiments(
            exp_pairwise,
            "ablation_pairwise_single_dataset.json",
            dataset_tag=dataset_tag,
            experiment_tag="ablation_pairwise_single_dataset",
        )
        experiments_to_run['pairwise'] = exp_pairwise
        logger.info(f"Loaded {len(exp_pairwise)} pairwise experiments")

    # Initialize benchmark
    benchmark = CrossDomainBenchmark(
        datasets=datasets,
        output_dir="results/cross_domain",
        random_seeds=FULL_ABLATION_RANDOM_SEEDS,
        data_root=DATA_ROOT,
        precision_target=0.9,
        fpr_target=0.05,
        threshold_strategy="precision",
        bootstrap_samples=2000,
    )
    
    # Run selected experiments
    results_dfs = {}
    for exp_name, experiments in experiments_to_run.items():
        logger.info(f"{'='*60}")
        logger.info(f"Running {exp_name} experiments ({len(experiments)} total)")
        logger.info(f"{'='*60}\n")
        
        df = benchmark.run_ablation_study(
            experiments,
            project_root,
            experiment_tag=f"ablation_{exp_name}_single_dataset",
            dataset_tag=dataset_tag,
        )
        results_dfs[exp_name] = df
        logger.info(f"Completed {exp_name} experiments")

    # Statistical significance testing
    if FULL_ABLATION_RUN_SIGNIFICANCE and not args.skip_significance:
        logger.info("" + "="*60)
        logger.info("Running statistical significance tests")
        logger.info("="*60 + "\n")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_keys = list(datasets.keys())

        # Oversampling vs None
        if 'oversampling' in results_dfs:
            logger.info("Testing oversampling significance...")
            oversampling_analyzer = AblationAnalyzer(results_dfs['oversampling'])
            oversampling_baseline = "ablation_oversampling_None"
            oversampling_results = []
            oversampling_map = {exp.exp_id: exp.components for exp in experiments_to_run['oversampling']}
            for exp in experiments_to_run['oversampling']:
                if exp.exp_id == oversampling_baseline:
                    continue
                for dataset in dataset_keys:
                    res = oversampling_analyzer.statistical_significance_test(
                        exp_id_1=oversampling_baseline,
                        exp_id_2=exp.exp_id,
                        dataset=dataset,
                        metric="pr_auc",
                    )
                    res["threshold_strategy"] = benchmark.threshold_strategy
                    res["precision_target"] = benchmark.precision_target
                    res["fpr_target"] = benchmark.fpr_target
                    res["exp_1_components"] = oversampling_map.get(res["exp_1"], {})
                    res["exp_2_components"] = oversampling_map.get(res["exp_2"], {})
                    oversampling_results.append(res)
            if oversampling_results:
                pd.DataFrame(oversampling_results).to_csv(
                    os.path.join("results", "ablation", f"significance_oversampling_single_{timestamp}.csv"),
                    index=False,
                )
                logger.info("Oversampling significance saved")

        # Model vs baselines
        if 'model' in results_dfs:
            logger.info("Testing model significance...")
            model_analyzer = AblationAnalyzer(results_dfs['model'])
            model_baselines = ["ablation_model_LogisticRegression", "ablation_model_DecisionTree"]
            model_results = []
            model_map = {exp.exp_id: exp.components for exp in experiments_to_run['model']}
            for exp in experiments_to_run['model']:
                if exp.exp_id in model_baselines:
                    continue
                for baseline in model_baselines:
                    for dataset in dataset_keys:
                        res = model_analyzer.statistical_significance_test(
                            exp_id_1=baseline,
                            exp_id_2=exp.exp_id,
                            dataset=dataset,
                            metric="pr_auc",
                        )
                        res["threshold_strategy"] = benchmark.threshold_strategy
                        res["precision_target"] = benchmark.precision_target
                        res["fpr_target"] = benchmark.fpr_target
                        res["exp_1_components"] = model_map.get(res["exp_1"], {})
                        res["exp_2_components"] = model_map.get(res["exp_2"], {})
                        model_results.append(res)
            if model_results:
                pd.DataFrame(model_results).to_csv(
                    os.path.join("results", "ablation", f"significance_models_single_{timestamp}.csv"),
                    index=False,
                )
                logger.info("Model significance saved")

        # Anomaly vs None
        if 'anomaly' in results_dfs:
            logger.info("Testing anomaly significance...")
            anomaly_analyzer = AblationAnalyzer(results_dfs['anomaly'])
            anomaly_baseline = "ablation_anomaly_signal_None"
            anomaly_results = []
            anomaly_map = {exp.exp_id: exp.components for exp in experiments_to_run['anomaly']}
            for exp in experiments_to_run['anomaly']:
                if exp.exp_id == anomaly_baseline:
                    continue
                for dataset in dataset_keys:
                    res = anomaly_analyzer.statistical_significance_test(
                        exp_id_1=anomaly_baseline,
                        exp_id_2=exp.exp_id,
                        dataset=dataset,
                        metric="pr_auc",
                    )
                    res["threshold_strategy"] = benchmark.threshold_strategy
                    res["precision_target"] = benchmark.precision_target
                    res["fpr_target"] = benchmark.fpr_target
                    res["exp_1_components"] = anomaly_map.get(res["exp_1"], {})
                    res["exp_2_components"] = anomaly_map.get(res["exp_2"], {})
                    anomaly_results.append(res)
            if anomaly_results:
                pd.DataFrame(anomaly_results).to_csv(
                    os.path.join("results", "ablation", f"significance_anomaly_single_{timestamp}.csv"),
                    index=False,
                )
                logger.info("Anomaly significance saved")

        # Calibration vs None
        if 'calibration' in results_dfs:
            logger.info("Testing calibration significance...")
            calib_analyzer = AblationAnalyzer(results_dfs['calibration'])
            calib_baseline = "ablation_calibration_None"
            calib_results = []
            calib_map = {exp.exp_id: exp.components for exp in experiments_to_run['calibration']}
            for exp in experiments_to_run['calibration']:
                if exp.exp_id == calib_baseline:
                    continue
                for dataset in dataset_keys:
                    res = calib_analyzer.statistical_significance_test(
                        exp_id_1=calib_baseline,
                        exp_id_2=exp.exp_id,
                        dataset=dataset,
                        metric="pr_auc",
                    )
                    res["threshold_strategy"] = benchmark.threshold_strategy
                    res["precision_target"] = benchmark.precision_target
                    res["fpr_target"] = benchmark.fpr_target
                    res["exp_1_components"] = calib_map.get(res["exp_1"], {})
                    res["exp_2_components"] = calib_map.get(res["exp_2"], {})
                    calib_results.append(res)
            if calib_results:
                pd.DataFrame(calib_results).to_csv(
                    os.path.join("results", "ablation", f"significance_calibration_single_{timestamp}.csv"),
                    index=False,
                )
                logger.info("Calibration significance saved")
        
        logger.info("All significance tests completed")

    raise SystemExit(0)

X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = loader.train_val_test_split()
logger.info(f"Data Training: {X_train.shape}, {y_train.shape}")
logger.info(f"Data Validation: {X_val.shape}, {y_val.shape}")
logger.info(f"Data Testing: {X_test.shape}, {y_test.shape}")
random_state = 42

if USE_ANOMALY_FEATURES and ANOMALY_METHOD != "None":
    logger.info("Adding anomaly score feature: %s", ANOMALY_METHOD)
    X_train, X_val, X_test = add_anomaly_scores(
        X_train,
        X_val,
        X_test,
        method=ANOMALY_METHOD,
        random_state=random_state,
        contamination=ANOMALY_CONTAMINATION,
    )

# === RESAMPLING METHODS ===
logger.info("Applying resampling techniques")

# 1. Regular SMOTE
logger.info("Applying SMOTE")
sm = SMOTE(random_state=random_state)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train) # pyright: ignore[reportAssignmentType]
logger.info(f"SMOTE - Class distribution: {y_train_smote.value_counts().to_dict()}")

# 2. Borderline SMOTE
logger.info("Applying Borderline SMOTE")
blsmote = BorderlineSMOTE(random_state=random_state)
X_train_blsmote, y_train_blsmote, *_ = blsmote.fit_resample(X_train, y_train)
logger.info(f"Borderline SMOTE - Class distribution: {y_train_blsmote.value_counts().to_dict()}")

# 3. SMOTEENN
logger.info("Applying SMOTEENN")
smote_enn = SMOTEENN(random_state=random_state)
X_train_smoteenn, y_train_smoteenn, *_ = smote_enn.fit_resample(X_train, y_train)
logger.info(f"SMOTEENN - Class distribution: {y_train_smoteenn.value_counts().to_dict()}")

gan_dataset_tag = DATASET_NAME.replace(".csv", "")

# 4. PyTorch GAN
logger.info("Training PyTorch GAN")
X_train_gan, y_train_gan, gen_losses, disc_losses = oversample_with_pytorch_gan(
    X_train,
    y_train,
    target_class=1,
    oversample_ratio=1.0,
    epochs=GAN_EPOCHS,
    batch_size=GAN_BATCH_SIZE,
    noise_dim=GAN_NOISE_DIM,
    hidden_dim=GAN_HIDDEN_DIM,
    train_max_minority_ratio=GAN_TRAIN_MAX_MINORITY_RATIO,
    early_stopping=GAN_EARLY_STOPPING,
    early_stopping_patience=GAN_EARLY_STOPPING_PATIENCE,
    early_stopping_delta=GAN_EARLY_STOPPING_DELTA,
    cache_path=GAN_CACHE_PATH,
    cache_tag=gan_dataset_tag,
)
logger.info(f"PyTorch GAN - Class distribution: {pd.Series(y_train_gan).value_counts().to_dict()}")
gan_syn_X, gan_syn_y = extract_synthetic_tail(X_train, X_train_gan, y_train, y_train_gan)

# 5. CTGAN
logger.info("Training CTGAN")
X_train_ctgan, y_train_ctgan, gen_losses_ctgan, disc_losses_ctgan = oversample_with_ctgan(
    X_train,
    y_train,
    target_class=1,
    oversample_ratio=1.0,
    epochs=GAN_EPOCHS,
    batch_size=GAN_BATCH_SIZE,
    noise_dim=GAN_NOISE_DIM,
    hidden_dim=GAN_HIDDEN_DIM,
    n_critic=GAN_N_CRITIC,
    train_max_minority_ratio=GAN_TRAIN_MAX_MINORITY_RATIO,
    early_stopping=GAN_EARLY_STOPPING,
    early_stopping_patience=GAN_EARLY_STOPPING_PATIENCE,
    early_stopping_delta=GAN_EARLY_STOPPING_DELTA,
    cache_path=GAN_CACHE_PATH,
    cache_tag=gan_dataset_tag,
)
logger.info(f"CTGAN - Class distribution: {pd.Series(y_train_ctgan).value_counts().to_dict()}")
ctgan_syn_X, ctgan_syn_y = extract_synthetic_tail(X_train, X_train_ctgan, y_train, y_train_ctgan)

# 6. Conditional WGAN-GP
logger.info("Training Conditional WGAN-GP")
X_train_cwgangp, y_train_cwgangp, gen_losses_cwgangp, disc_losses_cwgangp = oversample_with_cond_wgangp(
    X_train,
    y_train,
    target_class=1,
    target_ratio=1.0,
    epochs=GAN_EPOCHS,
    batch_size=GAN_BATCH_SIZE,
    noise_dim=GAN_NOISE_DIM,
    hidden_dim=GAN_HIDDEN_DIM,
    n_critic=GAN_N_CRITIC,
    train_max_minority_ratio=GAN_TRAIN_MAX_MINORITY_RATIO,
    early_stopping=GAN_EARLY_STOPPING,
    early_stopping_patience=GAN_EARLY_STOPPING_PATIENCE,
    early_stopping_delta=GAN_EARLY_STOPPING_DELTA,
    cache_path=GAN_CACHE_PATH,
    cache_tag=gan_dataset_tag,
)
logger.info(f"Conditional WGAN-GP - Class distribution: {pd.Series(y_train_cwgangp).value_counts().to_dict()}")
cwgan_syn_X, cwgan_syn_y = extract_synthetic_tail(X_train, X_train_cwgangp, y_train, y_train_cwgangp)

# === SYNTHETIC DATA EVALUATION ===
logger.info("Evaluating synthetic data quality")
synth_eval_rows = []

def _append_synth_eval(method_name, syn_X, syn_y):
    if syn_X is None or X_test is None or y_test is None:
        return
    
    metrics = evaluate_synthetic_data(
        X_real=X_train,
        X_syn=syn_X,
        X_test=X_test,
        y_test=y_test,
        y_syn=syn_y,
        y_real=y_train,
        seed=random_state,
    )
    metrics["method"] = method_name
    synth_eval_rows.append(metrics)

_append_synth_eval("PyTorch_GAN", gan_syn_X, gan_syn_y)
_append_synth_eval("CTGAN", ctgan_syn_X, ctgan_syn_y)
_append_synth_eval("Conditional_WGAN_GP", cwgan_syn_X, cwgan_syn_y)

if synth_eval_rows:
    synth_eval_df = pd.DataFrame(synth_eval_rows)
    synth_eval_path = os.path.join(
        project_root, "results", f"synth_eval_{DATASET_NAME.replace('.csv', '')}_{GAN_EPOCHS}.csv"
    )
    synth_eval_df.to_csv(synth_eval_path, index=False)
    logger.info("Synthetic evaluation saved to %s", synth_eval_path)
