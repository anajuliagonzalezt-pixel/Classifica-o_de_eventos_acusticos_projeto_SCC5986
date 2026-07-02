import optuna
import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

'''
Testaremos aqui os hiperparâmetros do modelo XGBoost sem a presença de métodos de amostragem
'''

# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================

output_dir = 'scripts/Estudos_Salvos/Features_Acusticas/Exp1'
data_dir= 'scripts/dados'
GLOBAL_SEED = 42
num_trials = 40

print("Carregando os dados...")
X_train = pd.read_csv(f'{data_dir}/X_train_acustico.csv')
df_y_train = pd.read_csv(f'{data_dir}/y_train_acustico.csv')

# Extraindo apenas os valores (array 1D) para o modelo não apresentar avisos (warnings)
y_train = df_y_train['target_encoded'].values

print(f"Dados carregados! X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print("-" * 50)

# ==========================================
# DEFINIÇÃO DA FUNÇÃO OBJETIVO (OPTUNA)
# ==========================================


def objective(trial):
    start = time.time()
    
    # 1. Parâmetros de escolha (Mantido None conforme o objetivo descrito)
    bal_method = None
    sampler = None
    
    if bal_method == 'SMOTE':
        sampler = SMOTE(random_state=GLOBAL_SEED)
    elif bal_method == 'UnderSampler':
        sampler = RandomUnderSampler(random_state=GLOBAL_SEED)
    elif bal_method == 'SMOTEENN':
        sampler = SMOTEENN(random_state=GLOBAL_SEED)
    elif bal_method == 'BorderlineSMOTE':
        sampler = BorderlineSMOTE(random_state=GLOBAL_SEED)
    elif bal_method == 'ADASYN':
        sampler = ADASYN(random_state=GLOBAL_SEED)


    modelo = XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 50, 300, step=50),
            max_depth=trial.suggest_int('xgb_max_depth', 3, 10),
            learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('xgb_subsample', 0, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0, 1.0),
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            random_state=GLOBAL_SEED, eval_metric='mlogloss', n_jobs=1
        )
    

    # 4. Construção do Pipeline
    steps = [('scaler', StandardScaler())]
    
    if sampler is not None:
        steps.append(('sampler', sampler))
        
    steps.append(('classifier', modelo))
    
    pipeline = ImbPipeline(steps)

    # 5. Cross-Validation
    try:
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        
        score = cross_val_score(
            pipeline, 
            X_train, 
            y_train, 
            cv=cv_strategy, 
            scoring='balanced_accuracy', 
            n_jobs=-1
        )
        score_mean = score.mean()
    except Exception as e:
        print(f"Trial {trial.number} falhou devido a erro: {e}")
        raise optuna.TrialPruned()

    elapsed = time.time() - start
    
    print(f"Trial {trial.number} (MLP + Sem Amostragem) terminou em {elapsed:.2f} segundos. Score: {score_mean:.4f}")

    return score_mean

# ==========================================
# CRIAÇÃO E EXECUÇÃO DO ESTUDO
# ==========================================
db_path = f"sqlite:///{output_dir}/estudo_XGBoost.db"


study = optuna.create_study( 
    study_name="experimento1", 
    storage=db_path, 
    load_if_exists=True,      
    direction="maximize",
    sampler=TPESampler(seed=GLOBAL_SEED)      
)
study.set_user_attr("seed_utilizada", GLOBAL_SEED)

print("Iniciando a otimização de hiperparâmetros...")

# Roda o experimento por 10 trials
study.optimize(objective, n_trials=num_trials, n_jobs=-1) 

# 2. Puxe as informações do melhor trial
melhor_trial = study.best_trial

# 3. Exiba o resumo dos resultados 
print("\n=== RESULTADO DA OTIMIZAÇÃO ===")
print(f"Trial Número: {melhor_trial.number}")
print(f"Melhor Métrica (Ex: Acurácia): {melhor_trial.value:.4f}")
print("\nMelhores Hiperparâmetros encontrados:")

for parametro, valor in melhor_trial.params.items():
    print(f"  -> {parametro}: {valor}")