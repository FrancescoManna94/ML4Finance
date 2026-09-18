import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os
import optuna.visualization
from tqdm import tqdm
import pandas_ta as ta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in uso: {device}")
if torch.cuda.is_available():
    print(f"GPU in uso: {torch.cuda.get_device_name(device)}")



df_original = pd.read_csv('D:/Unical/Esami/Fatti/Machine e Deep Learning/Progetto/S&P 500 Dati.csv',dtype={'Data': str},)
df=df_original.copy()
df['Data'] = pd.to_datetime(df['Data'],format='%d%m%Y',dayfirst=True)
df.rename(columns={'Data': 'Date'}, inplace=True)
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
print(df.shape)

def add(df):
    # RSI
    df['rsi5']  = ta.rsi(df['Ultimo'], length=5)
    df['rsi10'] = ta.rsi(df['Ultimo'], length=10)
    df['rsi20'] = ta.rsi(df['Ultimo'], length=20)

    # MACD → restituisce DataFrame, non tupla
    macd = ta.macd(df['Ultimo'], fast=12, slow=26, signal=9)
    df['macd']       = macd.iloc[:, 0]
    df['macdsignal'] = macd.iloc[:, 1]
    df['macdhist']   = macd.iloc[:, 2]

    # STOCH → restituisce DataFrame
    stoch = ta.stoch(df['Massimo'], df['Minimo'], df['Ultimo'], k=5, d=3)
    df['slowk'] = stoch.iloc[:, 0]
    df['slowd']  = stoch.iloc[:, 1]

    # STOCHF (Stochastic Fast) → restituisce DataFrame
    stochf = ta.stochrsi(df['Ultimo'], length=5, rsi_length=3)
    df['fastk'] = stochf.iloc[:, 0]
    df['fastd']  = stochf.iloc[:, 1]

    # Williams %R
    df['WR5']  = ta.willr(df['Massimo'], df['Minimo'], df['Ultimo'], length=5)
    df['WR10'] = ta.willr(df['Massimo'], df['Minimo'], df['Ultimo'], length=10)
    df['WR20'] = ta.willr(df['Massimo'], df['Minimo'], df['Ultimo'], length=20)

    # ROC
    df['ROC5']  = ta.roc(df['Ultimo'], length=5)
    df['ROC10'] = ta.roc(df['Ultimo'], length=10)
    df['ROC20'] = ta.roc(df['Ultimo'], length=20)

    # CCI
    df['CCI5']  = ta.cci(df['Massimo'], df['Minimo'], df['Ultimo'], length=5)
    df['CCI10'] = ta.cci(df['Massimo'], df['Minimo'], df['Ultimo'], length=10)
    df['CCI20'] = ta.cci(df['Massimo'], df['Minimo'], df['Ultimo'], length=20)

    # ATR
    df['ATR5']  = ta.atr(df['Massimo'], df['Minimo'], df['Ultimo'], length=5)
    df['ATR10'] = ta.atr(df['Massimo'], df['Minimo'], df['Ultimo'], length=10)
    df['ATR20'] = ta.atr(df['Massimo'], df['Minimo'], df['Ultimo'], length=20)

    # NATR (pandas_ta non ha NATR nativo → lo calcoliamo manualmente)
    df['NATR5']  = (df['ATR5']  / df['Ultimo']) * 100
    df['NATR10'] = (df['ATR10'] / df['Ultimo']) * 100
    df['NATR20'] = (df['ATR20'] / df['Ultimo']) * 100

    # True Range
    df['TRANGE'] = ta.true_range(df['Massimo'], df['Minimo'], df['Ultimo'])

    return df
df=add(df)
print(df.shape)
# Selezione delle colonne numeriche da usare per la BiLSTM
features_cols = ['Ultimo','Apertura', 'Massimo', 'Minimo',
                 'rsi5', 'rsi10', 'rsi20', 'macd', 'macdsignal', 'macdhist',
                 'slowk', 'slowd', 'fastk', 'fastd', 'WR5', 'WR10', 'WR20',
                 'ROC5', 'ROC10', 'ROC20', 'CCI5', 'CCI10', 'CCI20',
                 'ATR5', 'ATR10', 'ATR20', 'NATR5', 'NATR10', 'NATR20', 'TRANGE']
print(f"Colonne selezionate per la BiLSTM: {features_cols}")
# Gestione dei NaN
all_relevant_cols_for_nan_check = features_cols
original_rows_count = df.shape[0]
df.dropna(subset=all_relevant_cols_for_nan_check, inplace=True)
rows_dropped_count = original_rows_count - df.shape[0]

if rows_dropped_count > 0:
    print(f"\nATTENZIONE: Rimossa {rows_dropped_count} riga/e a causa di valori NaN in feature.")
print(f"Numero di righe del DataFrame DOPO la pulizia dei NaN: {df.shape[0]}")
print("-" * 50)

def scaler_dinamico(data,rolling):
    rolling_mean =data.rolling(window=rolling).mean()
    rolling_std = data.rolling(window=rolling).std()
    dinamico = (data - rolling_mean) / rolling_std
    parametri_scaling = pd.DataFrame(index=data.index)

    for col in data.columns:
        parametri_scaling[f'{col}_mean'] = rolling_mean[col]
        parametri_scaling[f'{col}_std'] = rolling_std[col]

    return dinamico, parametri_scaling

rolling=30
dati_dinamici, parametri_scaling = scaler_dinamico(df[features_cols], rolling)
dati_dinamici.dropna(inplace=True)
parametri_scaling.dropna(inplace=True)
print(dati_dinamici.shape)
print(parametri_scaling.shape)
print(dati_dinamici.index)

total_observations = len(dati_dinamici)# Dovrebbe essere 3712
print(total_observations)
train_size = 2599 # 70% 2599
validation_size = 743 # 20% 743
test_size = 370 # 10%  370

# Verifica che le dimensioni siano coerenti
if (train_size + validation_size + test_size) != total_observations:
    print(f"AVVISO: Le dimensioni specificate ({train_size} + {validation_size} + {test_size} = {train_size + validation_size + test_size}) non corrispondono al totale delle osservazioni ({total_observations}). Questo potrebbe portare a dati mancanti o sovrapposti.")

# Calcola gli indici di divisione
train_end_index = train_size
validation_end_index = train_size + validation_size
# test_end_index è semplicemente la fine del DataFrame

# Divisione del DataFrame principale
df_train = dati_dinamici.iloc[:train_end_index].copy()
df_validation = dati_dinamici.iloc[train_end_index:validation_end_index].copy()
df_test = dati_dinamici.iloc[validation_end_index:].copy()
print(f"\nDimensioni dei set dopo lo split temporale:")
print(f"df: {df.shape}")
print(f"df_train shape: {df_train.shape}")
print(f"df_validation shape: {df_validation.shape}")
print(f"df_test shape: {df_test.shape}")

def create_sequences_for_bilstm(data, features_cols, target_col, time_steps):
    X, y = [], []

    features = data[features_cols].values
    target = data[target_col].values

    for i in range(len(data) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(target[i + time_steps])  # valore futuro da prevedere

    return np.array(X), np.array(y)
TIME_STEPS = 30
target_col = 'Ultimo'

X_sequences_train, y_sequences_train = create_sequences_for_bilstm(df_train, features_cols, target_col, TIME_STEPS)
X_sequences_validation, y_sequences_validation = create_sequences_for_bilstm(df_validation, features_cols, target_col, TIME_STEPS)
X_sequences_test, y_sequences_test = create_sequences_for_bilstm(df_test, features_cols, target_col, TIME_STEPS)
print(f"X_train: {X_sequences_train.shape}, y_train: {y_sequences_train.shape}")
print(f"X_val: {X_sequences_validation.shape}, y_val: {y_sequences_validation.shape}")
print(f"X_test: {X_sequences_test.shape}, y_test: {y_sequences_test.shape}")

X_train_t = torch.tensor(X_sequences_train, dtype=torch.float32)
y_train_t = torch.tensor(y_sequences_train, dtype=torch.float32)

X_val_t = torch.tensor(X_sequences_validation, dtype=torch.float32)
y_val_t = torch.tensor(y_sequences_validation, dtype=torch.float32)

X_test_t = torch.tensor(X_sequences_test, dtype=torch.float32)
y_test_t = torch.tensor(y_sequences_test, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definizione modello BiLSTM PyTorch
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, bilstm_units_l1, bilstm_units_l2, bilstm_units_l3, feature_dim, dropout_rate):
        super(BiLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, bilstm_units_l1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(bilstm_units_l1 * 2, bilstm_units_l2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lstm3 = nn.LSTM(bilstm_units_l2 * 2, bilstm_units_l3, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.feature_layer = nn.Linear(bilstm_units_l3 * 2, feature_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: [batch, time_steps, features]
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        # Prendiamo l'ultimo timestep
        out = out[:, -1, :]  # [batch, bilstm_units_l3*2]
        features = self.relu(self.feature_layer(out))
        output = self.output_layer(features)
        return output.squeeze(), features  # output: [batch], features: [batch, feature_dim]

    # Metodo per estrarre features da un input
    def extract_features(self, x):
        with torch.no_grad():
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            out, _ = self.lstm3(out)
            out = self.dropout3(out)
            out = out[:, -1, :]
            features = self.relu(self.feature_layer(out))
        return features

def objective(trial):
    # Hyperparametri BiLSTM
    bilstm_units_l1 = trial.suggest_int('bilstm_units_l1', 32, 512, step=8)
    bilstm_units_l2 = trial.suggest_int('bilstm_units_l2', 16, 256, step=8)
    bilstm_units_l3 = trial.suggest_int('bilstm_units_l3', 8, 128, step=8)
    feature_dim = trial.suggest_int('feature_dim', 8, 128, step=8)
    bilstm_learning_rate = trial.suggest_float('bilstm_learning_rate', 1e-4, 5e-3, log=True)
    bilstm_epochs = trial.suggest_int('bilstm_epochs', 50, 200)
    bilstm_batch_size = trial.suggest_categorical('bilstm_batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # Hyperparametri LightGBM
    lgbm_n_estimators = trial.suggest_int('lgbm_n_estimators', 100, 1000)
    lgbm_learning_rate = trial.suggest_float('lgbm_learning_rate', 0.001, 0.1, log=True)
    lgbm_num_leaves = trial.suggest_int('lgbm_num_leaves', 16, 64)
    lgbm_max_depth = trial.suggest_int('lgbm_max_depth', 5, 15)
    lgbm_reg_alpha = trial.suggest_float('lgbm_reg_alpha', 1e-4, 1.0, log=True)
    lgbm_reg_lambda = trial.suggest_float('lgbm_reg_lambda', 1e-4, 1.0, log=True)
    lgbm_colsample_bytree = trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0)

    # Prepara dati tensori PyTorch
    X_train_tensor = torch.tensor(X_sequences_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_sequences_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_sequences_validation, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_sequences_validation, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=bilstm_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bilstm_batch_size, shuffle=False)

    # Crea modello
    model = BiLSTMModel(
        input_size=len(features_cols),
        bilstm_units_l1=bilstm_units_l1,
        bilstm_units_l2=bilstm_units_l2,
        bilstm_units_l3=bilstm_units_l3,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=bilstm_learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5

    for epoch in tqdm(range(bilstm_epochs), desc='Epochs'):
        model.train()
        batch_loop = tqdm(train_loader, desc='Training batches', leave=False)
        for X_batch, y_batch in batch_loop:
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            batch_loop.set_postfix(loss=loss.item())

        # Validazione
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_outputs, _ = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                val_losses.append(val_loss.item())
        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/bilstm_trial_{trial.number}.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Carica miglior modello per estrarre features
    model.load_state_dict(torch.load(f"models/bilstm_trial_{trial.number}.pt"))
    model.eval()

    # Estrazione features train
    train_features_list = []
    with torch.no_grad():
        for X_batch, _ in train_loader:
            feats = model.extract_features(X_batch)
            train_features_list.append(feats.cpu().numpy())
    train_features = np.vstack(train_features_list)

    # Estrazione features validation
    val_features_list = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            feats = model.extract_features(X_batch)
            val_features_list.append(feats.cpu().numpy())
    val_features = np.vstack(val_features_list)

    # Allena LightGBM sulle feature estratte
    lgbm_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=lgbm_n_estimators,
        learning_rate=lgbm_learning_rate,
        num_leaves=lgbm_num_leaves,
        max_depth=lgbm_max_depth,
        reg_alpha=lgbm_reg_alpha,
        reg_lambda=lgbm_reg_lambda,
        colsample_bytree=lgbm_colsample_bytree,
        random_state=42,
        n_jobs=-1
    )

    lgbm_model.fit(train_features, y_sequences_train)

    # Valutazione LightGBM sul set di validazione
    val_predictions = lgbm_model.predict(val_features)
    rmse = np.sqrt(mean_squared_error(y_sequences_validation, val_predictions))

    # Salva info nel trial
    trial.set_user_attr("val_predictions", val_predictions.tolist())
    trial.set_user_attr("trial_number", trial.number)

    # Salva modello LightGBM
    joblib.dump(lgbm_model, f"models/lgbm_trial_{trial.number}.txt")

    return rmse


# Creiamo uno studio Optuna
study = optuna.create_study(direction='minimize', study_name='Hybrid_BiLSTM_LightGBM_Optimization')

# Esegue l'ottimizzazione, abbiamo impostato n_trials=2 per evitare che l'esecuzione sia troppo lunga
study.optimize(objective, n_trials=2) #Sono stati effettuati 100 trial
print("\n" + "="*50)
print("Ottimizzazione Completata!")
print(f"Numero di trial completati: {len(study.trials)}")
print(f"Miglior trial trovato (RMSE): {study.best_value}")
print("Migliori Iperparametri:")
for key, value in study.best_params.items():
    print(f"   {key}: {value}")
print("="*50)
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 8))
plt.plot(y_sequences_validation, label='Valori Reali (Validation)', color='blue', linewidth=2)

# Plotta le previsioni di tutti i trial completi
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE and "val_predictions" in trial.user_attrs:
        trial_predictions = np.array(trial.user_attrs["val_predictions"])
        trial_number = trial.user_attrs["trial_number"]
        rmse_value = trial.value  # RMSE del trial
        plt.plot(trial_predictions, label=f'Pred. Trial {trial_number} (RMSE: {rmse_value:.4f})', alpha=0.5, linestyle='--')

plt.title('Confronto Previsioni Trial vs. Valori Reali (Validation Set)')
plt.xlabel('Indice Temporale')
plt.ylabel(target_col)
plt.legend()
plt.grid(True)
plt.show()

# Plot del miglior trial
best_trial = study.best_trial
if best_trial and "val_predictions" in best_trial.user_attrs:
    best_predictions = np.array(best_trial.user_attrs["val_predictions"])
    plt.figure(figsize=(12, 6))
    plt.plot(date_validation, y_sequences_validation, label='Valori Reali (Validation)', color='blue', linewidth=2)
    plt.plot(date_validation, best_predictions, label=f'Miglior Predizione (RMSE: {best_trial.value:.4f})', color='red', linestyle='--')
    plt.title('Migliore Previsione (Trial Optuna) vs. Valori Reali (Validation Set)')
    plt.xlabel('Data')
    plt.ylabel(target_col)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(f"Miglior Trial: {best_trial.number}")
print(f"Miglior RMSE: {best_trial.value}")
print(f"Migliori Parametri:")
for key, value in best_trial.params.items():
    print(f"   {key}: {value}")

# Caricamento del modello salvato (deve essere definita anche la classe del modello)
model = torch.load('D:/Unical/Fatti/Machine e Deep Learning/Progetto/models/bilstm_trial_7.pt')
model.eval()  # imposta il modello in modalità valutazione

# Stampa del modello
print(model)

# Carica il modello LightGBM
lgb_model = joblib.load('D:/Unical/Fatti/Machine e Deep Learning/Progetto/LGBM.txt')
# Estrai i parametri
params = lgb_model.get_params()
# Stampa i parametri con tabulate
print("\nParametri LightGBM con tabulate:\n")
print(tabulate(params.items(), headers=["Parametro", "Valore"], tablefmt="fancy_grid"))
#Recupero le date dal df originale
date = df['Date'].values
date_test = date[-len(predictions_test):]
date_validation = date[-(len(predictions_val) + len(predictions_test)):-len(predictions_test)]

# Estrazione feature dal layer specifico
feature_extractor = Model(inputs=bilstm_model.input,
                          outputs=bilstm_model.get_layer('extracted_features_for_training').output)

# Estrai feature dal dataset di input (training o test)
features_val = feature_extractor.predict(X_sequences_validation)
features_test= feature_extractor.predict(X_sequences_test)
# Previsioni LightGBM
predictions_val = lgb_model.predict(features_val)
predictions_test = lgb_model.predict(features_test)
#validation
mse_val = mean_squared_error(y_sequences_validation, predictions_val)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_sequences_validation, predictions_val)


mse_test = mean_squared_error(y_sequences_test, predictions_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_sequences_test, predictions_test)
target_col = 'Ultimo'  # metti il nome della tua variabile target

# Indici validation e test (esempi, sostituisci con i tuoi indici reali)
val_indices = dati_dinamici.index[-(len(predictions_val) + len(predictions_test)):-len(predictions_test)]
test_indices = dati_dinamici.index[-len(predictions_test):]

# Recupera mean e std dai parametri di scaling
mean_val = parametri_scaling.loc[val_indices, f'{target_col}_mean'].values
std_val = parametri_scaling.loc[val_indices, f'{target_col}_std'].values

mean_test = parametri_scaling.loc[test_indices, f'{target_col}_mean'].values
std_test = parametri_scaling.loc[test_indices, f'{target_col}_std'].values

# Riscalare predizioni (element-wise)
pred_val_orig = predictions_val * std_val + mean_val
pred_test_orig = predictions_test * std_test + mean_test

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

target_col = 'Ultimo'  # metti qui il nome della tua variabile target

# Indici validation e test (esempi, sostituisci con i tuoi indici corretti)
val_indices = dati_dinamici.index[-(len(predictions_val) + len(predictions_test)):-len(predictions_test)]
test_indices = dati_dinamici.index[-len(predictions_test):]


y_val_orig = y_sequences_validation * std_val + mean_val
y_test_orig = y_sequences_test * std_test + mean_test

# Calcolo RMSE per le predizioni riscalate
rmse_val = np.sqrt(mean_squared_error(y_val_orig, pred_val_orig))
rmse_test = np.sqrt(mean_squared_error(y_test_orig, pred_test_orig))

# Plot Validation
plt.figure(figsize=(12, 6))
plt.plot(date_validation, y_val_orig, label='Valori Reali (Validation)', color='blue', linewidth=2)
plt.plot(date_validation, pred_val_orig, label='Predizioni LightGBM ', color='red', linestyle='--')
plt.title('Predizioni LightGBM vs Valori Reali (Validation)')
plt.xlabel('Data')
plt.ylabel('Valore')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Test
plt.figure(figsize=(12, 6))
plt.plot(date_test, y_test_orig, label='Valori Reali (Test)', color='blue', linewidth=2)
plt.plot(date_test, pred_test_orig, label='Predizioni LightGBM ', color='red', linestyle='--')
plt.title('Predizioni LightGBM vs Valori Reali (Test)')
plt.xlabel('Data')
plt.ylabel('Valore')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# Validation metrics
mape_val = mean_absolute_percentage_error(y_val_orig, pred_val_orig)
print(f"MAPE Validation: {mape_val:.2f}%")
# Test metrics
mape_test = mean_absolute_percentage_error(y_test_orig, pred_test_orig)
print(f"MAPE Test: {mape_test:.2f}%")

# carica i dati
ferrari=pd.read_csv('D:/Unical/Fatti/Machine e Deep Learning/Progetto/RACE Cronologia Dati.csv',dtype={'Data': str},)
amazon=pd.read_csv('D:/Unical/Fatti/Machine e Deep Learning/Progetto/AMZN Cronologia Dati.csv',dtype={'Data': str},)
oro=pd.read_csv('D:/Unical/Fatti/Machine e Deep Learning/Progetto/Future Oro Dati Storici.csv',dtype={'Data': str},)
intesa=pd.read_csv('D:/Unical/Fatti/Machine e Deep Learning/Progetto/ISP Cronologia Dati.csv',dtype={'Data': str},)
mib=pd.read_csv('D:/Unical/Fatti/Machine e Deep Learning/Progetto/FTSE MIB Dati Storici.csv',dtype={'Data': str},)
ferrari.drop(columns=['Vol.', 'Var. %'], inplace=True)
amazon.drop(columns=['Vol.', 'Var. %'], inplace=True)
oro.drop(columns=['Vol.', 'Var. %'], inplace=True)
intesa.drop(columns=['Vol.', 'Var. %'], inplace=True)
mib.drop(columns=['Vol.', 'Var. %'], inplace=True)
print(ferrari)
print(mib)
#convert 'Data' column to datetime format
ferrari['Data'] = pd.to_datetime(ferrari['Data'], format='%d.%m.%Y')
amazon['Data'] = pd.to_datetime(amazon['Data'], format='%d.%m.%Y')
oro['Data'] = pd.to_datetime(oro['Data'], format='%d.%m.%Y')
intesa['Data'] = pd.to_datetime(intesa['Data'], format='%d.%m.%Y')
mib['Data'] = pd.to_datetime(mib['Data'], format='%d.%m.%Y')
# Sort the dataframes by 'Data' and reset the index
ferrari.sort_values(by='Data', ascending=True, inplace=True)
ferrari.reset_index(drop=True, inplace=True)
amazon.sort_values(by='Data', ascending=True, inplace=True)
amazon.reset_index(drop=True, inplace=True)
oro.sort_values(by='Data', ascending=True, inplace=True)
oro.reset_index(drop=True, inplace=True)
intesa.sort_values(by='Data', ascending=True, inplace=True)
intesa.reset_index(drop=True, inplace=True)
mib.sort_values(by='Data', ascending=True, inplace=True)
mib.reset_index(drop=True, inplace=True)

# ADATTA LE VIRGOLE E I PUNTI
cols_to_convert = ['Ultimo', 'Apertura', 'Massimo', 'Minimo']
for col in cols_to_convert:
    ferrari[col] = ferrari[col].str.replace(',', '.').astype(float)
    amazon[col] = amazon[col].str.replace(',', '.').astype(float)
    oro[col] = oro[col].str.replace('.', '', regex=False)  # rimuove i punti
    oro[col] = oro[col].str.replace(',', '.', regex=False) # sostituisce virgola con punto
    mib[col] = mib[col].str.replace('.', '', regex=False)
    mib[col] = mib[col].str.replace(',', '.', regex=False)
    intesa[col] = intesa[col].str.replace(',', '.').astype(float)
    oro[col] = oro[col].str.replace(',', '.').astype(float)
    mib[col] = mib[col].str.replace(',', '.').astype(float)

#DEBUG
print(ferrari)
print('-' *50)
print(amazon)
print('-' *50)
print(oro)
print('-' *50)
print(intesa)
print('-' *50)
print(mib)
print('-' *50)
# Add technical indicators
ferrari=add(ferrari)
print(ferrari.shape)
amazon=add(amazon)
print(amazon.shape)
oro=add(oro)
print(oro.shape)
intesa=add(intesa)
print(intesa.shape)
mib=add(mib)
print(mib.shape)

"""# 3.1.2 🧹 Gestione dei NaN"""

# GESTIONE DEI NaN
all_relevant_cols_for_nan_check = features_cols
original_rows_count_ferrari = ferrari.shape[0]
original_rows_count_amazon = amazon.shape[0]
original_rows_count_oro = oro.shape[0]
original_rows_count_intesa = intesa.shape[0]
original_rows_count_mib = mib.shape[0]

ferrari.dropna(subset=all_relevant_cols_for_nan_check, inplace=True)
amazon.dropna(subset=all_relevant_cols_for_nan_check, inplace=True)
oro.dropna(subset=all_relevant_cols_for_nan_check, inplace=True)
intesa.dropna(subset=all_relevant_cols_for_nan_check, inplace=True)
mib.dropna(subset=all_relevant_cols_for_nan_check, inplace=True)
rows_dropped_count_ferrari = original_rows_count_ferrari - ferrari.shape[0]
rows_dropped_count_amazon = original_rows_count_amazon - amazon.shape[0]
rows_dropped_count_oro = original_rows_count_oro - oro.shape[0]
rows_dropped_count_intesa = original_rows_count_intesa - intesa.shape[0]
row_dropped_count_mib = original_rows_count_mib - mib.shape[0]


if any([rows_dropped_count_ferrari, rows_dropped_count_amazon, rows_dropped_count_oro, rows_dropped_count_intesa]):
    if rows_dropped_count_ferrari > 0:
        print(f"ATTENZIONE: Rimossa {rows_dropped_count_ferrari} riga/e da Ferrari a causa di valori NaN.")
    if rows_dropped_count_amazon > 0:
        print(f"ATTENZIONE: Rimossa {rows_dropped_count_amazon} riga/e da Amazon a causa di valori NaN.")
    if rows_dropped_count_oro > 0:
        print(f"ATTENZIONE: Rimossa {rows_dropped_count_oro} riga/e da Oro a causa di valori NaN.")
    if rows_dropped_count_intesa > 0:
        print(f"ATTENZIONE: Rimossa {rows_dropped_count_intesa} riga/e da Intesa a causa di valori NaN.")
else:
    print("Nessuna riga rimossa a causa di valori NaN.")

print(f"Numero di righe del DataFrame DOPO la pulizia dei NaN: {ferrari.shape[0]}")
print(f"Numero di righe del DataFrame DOPO la pulizia dei NaN: {amazon.shape[0]}")
print(f"Numero di righe del DataFrame DOPO la pulizia dei NaN: {oro.shape[0]}")
print(f"Numero di righe del DataFrame DOPO la pulizia dei NaN: {intesa.shape[0]}")
print(f"Numero di righe del DataFrame DOPO la pulizia dei NaN: {mib.shape[0]}")
print("-" * 50)

"""# 3.1.3 Applicazione Scaling Dinamico"""

# ESEGUI LO SCALING DINAMICO
rolling = 30
datasets = {
    'ferrari': ferrari,
    'amazon': amazon,
    'oro': oro,
    'intesa': intesa,
    'mib' : mib
}

dati_dinamici_dict = {}
parametri_scaling_dict = {}

for nome, df in datasets.items():
    dati_dinamici, parametri_scaling = scaler_dinamico(df[features_cols], rolling)
    dati_dinamici.dropna(inplace=True)
    parametri_scaling.dropna(inplace=True)

    dati_dinamici_dict[nome] = dati_dinamici
    parametri_scaling_dict[nome] = parametri_scaling

    print(f"{nome} - dati dinamici shape: {dati_dinamici.shape}")
    print(f"{nome} - parametri scaling shape: {parametri_scaling.shape}")

print(dati_dinamici_dict['ferrari'])
print(dati_dinamici_dict['amazon'])
print(dati_dinamici_dict['oro'])
print(dati_dinamici_dict['intesa'])
print(dati_dinamici_dict['mib'])

"""# 3.1.4 Split in Train e Test
## In questo blocco, i 5 nuovi dataset vengono divisi in **Train e Test (80/20)**, seguendo sempre l'ordine temporale. Non è previsto un set di validazione, poiché il modello è già addestrato: l’obiettivo è solo **valutare le sue prestazioni su dati completamente nuovi**.

"""

#Split i dati in train e test
datasets = {
    'ferrari': dati_dinamici_dict['ferrari'],
    'amazon': dati_dinamici_dict['amazon'],
    'oro': dati_dinamici_dict['oro'],
    'intesa': dati_dinamici_dict['intesa'],
    'mib' : dati_dinamici_dict['mib']
}

train_ratio = 0.8
test_ratio = 0.2

train_data = {}
test_data = {}

for nome, df in datasets.items():
    n = len(df)
    train_end_index = int(n * train_ratio)

    train_data[nome] = df.iloc[:train_end_index].copy()
    test_data[nome] = df.iloc[train_end_index:].copy()

    print(f"{nome}: train shape = {train_data[nome].shape}, test shape = {test_data[nome].shape}")

"""# 3.1.5 Estrazione Parametri di Scaling
## In questo blocco estraiamo i parametri di scaling (mean e std) per ciascun test set, utilizzando gli stessi indici. Questi verranno usati successivamente per riscalare le previsioni alla scala originale.
"""

# Estrai i parametri di scaling per il test set
parametri_scaling_test = {}

for nome in ['ferrari', 'amazon', 'oro', 'intesa', 'mib']:
    params = parametri_scaling_dict[nome]
    test_idx = test_data[nome].index  # gli stessi indici usati per test_data
    parametri_scaling_test[nome] = params.loc[test_idx].copy()

    print(f"{nome}: parametri_scaling_test shape = {parametri_scaling_test[nome].shape}")

"""# 3.1.6 Creazione Sequenze per BiLSTM"""

def create_sequences_for_bilstm(data, time_steps):
    x = []
    # Assicurati che data sia un array NumPy per un slicing corretto
    if isinstance(data, pd.DataFrame):
        data = data.values
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps)])
    return np.array(x)
# crea le sequenze per BiLSTM
X_sequences_train = {}
y_sequences_train = {}
X_sequences_test = {}
y_sequences_test = {}

for nome in ['ferrari', 'amazon', 'oro', 'intesa', 'mib']:
    # Train
    X_train = train_data[nome][features_cols].values
    y_train = train_data[nome]['Ultimo'].values  # target nel train

    X_sequences_train[nome] = create_sequences_for_bilstm(X_train,TIME_STEPS)
    y_sequences_train[nome] = y_train[TIME_STEPS:].copy()

    # Test
    X_test = test_data[nome][features_cols].values
    y_test = test_data[nome]['Ultimo'].values  # target nel test

    X_sequences_test[nome] = create_sequences_for_bilstm(X_test, TIME_STEPS)
    y_sequences_test[nome] = y_test[TIME_STEPS:].copy()

    print(f"{nome} - train X shape: {X_sequences_train[nome].shape}, train y shape: {y_sequences_train[nome].shape}")
    print(f"{nome} - test X shape: {X_sequences_test[nome].shape}, test y shape: {y_sequences_test[nome].shape}")

"""# 3.2 Feature Extraction con BiLSTM e Previsioni LightGBM

In questa fase carichiamo il modello BiLSTM pre-addestrato e creiamo un modellodi estrazione delle feature intermedie, che corrispondono all’output del layer `extracted_features_for_training`. Applichiamo questo estrattore sia ai dati di training che di test per tutti e cinque i dataset nuovi, ottenendo così rappresentazioni compatte e significative da usare come input per il modello LightGBM. Successivamente, carichiamo il modello LightGBM precedentemente addestrato, e lo utilizziamo per fare previsioni sui dati di test trasformati tramite BiLSTM. Per ciascun dataset valutiamo le performance del modello con metriche di errore comuni (RMSE, MSE, MAE), così da confrontare l’efficacia della predizione su diversi mercati e asset.
"""

# Carica il modello BiLSTM pre-addestrato
feature_extractor_model_full = load_model('/content/drive/MyDrive/BiLSTM.keras')
feature_extractor_model = Model(inputs=feature_extractor_model_full.input,
                               outputs=feature_extractor_model_full.get_layer('extracted_features_for_training').output)

extracted_features_train_lgbm = {}

for nome in ['ferrari', 'amazon', 'oro', 'intesa','mib']:
    extracted_features_train_lgbm[nome] = feature_extractor_model.predict(X_sequences_train[nome])
    print(f"Features estratte per {nome} - shape: {extracted_features_train_lgbm[nome].shape}")

extracted_features_test_lgbm = {}

for nome in ['ferrari', 'amazon', 'oro', 'intesa','mib']:
    extracted_features_test_lgbm[nome] = feature_extractor_model.predict(X_sequences_test[nome])
    print(f"Features estratte per {nome} (test) - shape: {extracted_features_test_lgbm[nome].shape}")

# Prepara i dati per LightGBM
X_train_lgb = {}
y_train_lgb = {}

X_test_lgb = {}
y_test_lgb = {}

for nome in ['ferrari', 'amazon', 'oro', 'intesa','mib']:
    X_train_lgb[nome] = extracted_features_train_lgbm[nome]
    y_train_lgb[nome] = y_sequences_train[nome]
    X_test_lgb[nome] = extracted_features_test_lgbm[nome]
    y_test_lgb[nome] = y_sequences_test[nome]

# Carica il modello LightGBM pre-addestrato
lgb_model = joblib.load('/content/drive/MyDrive/LGBM.txt')
# Valuta il modello LightGBM sui dati di test
y_pred_lgb_all = {}
y_true_all = {}

for nome in ['ferrari', 'amazon', 'oro', 'intesa','mib']:
    y_pred_lgb_all[nome] = lgb_model.predict(X_test_lgb[nome])
    y_true_all[nome] = y_test_lgb[nome]

    mse = mean_squared_error(y_true_all[nome], y_pred_lgb_all[nome])
    mae = mean_absolute_error(y_true_all[nome], y_pred_lgb_all[nome])
    rmse = np.sqrt(mse)

    print(f"\nValutazione modello LightGBM per {nome}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("-" * 50)

"""# 3.2.3 RMSE, MSE E MAE"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

# Dizionari per memorizzare i valori
rmse_dict = {}
mse_dict = {}
mae_dict = {}

# Calcolo degli errori per ciascun asset
for nome in ['ferrari', 'amazon', 'oro', 'intesa', 'mib']:
    y_pred_lgb_all[nome] = lgb_model.predict(X_test_lgb[nome])
    y_true_all[nome] = y_test_lgb[nome]

    mse = mean_squared_error(y_true_all[nome], y_pred_lgb_all[nome])
    mae = mean_absolute_error(y_true_all[nome], y_pred_lgb_all[nome])
    rmse = np.sqrt(mse)

    mse_dict[nome] = mse
    rmse_dict[nome] = rmse
    mae_dict[nome] = mae

# Ordine degli asset
assets = ['ferrari', 'amazon', 'oro', 'intesa', 'mib']

# Valori per il plot
rmse_vals = [rmse_dict[n] for n in assets]
mse_vals = [mse_dict[n] for n in assets]
mae_vals = [mae_dict[n] for n in assets]

print("Metriche per ciascun asset:\n")
print(f"{'Asset':<10} {'MSE':>10} {'RMSE':>10} {'MAE':>10}")
print("-" * 40)
for nome in assets:
    print(f"{nome:<10} {mse_dict[nome]:10.4f} {rmse_dict[nome]:10.4f} {mae_dict[nome]:10.4f}")

"""#3.2.1 Plot Previsioni non riscalate"""

df_originali = {'ferrari': ferrari,'amazon': amazon,'oro': oro,'intesa': intesa,'mib': mib}
for nome in ['ferrari', 'amazon', 'oro', 'intesa', 'mib']:
    y_true = y_true_all[nome]
    y_pred = y_pred_lgb_all[nome]
    n = len(y_true)

    # Ottieni gli indici delle ultime n righe nel DataFrame dinamico
    indici_test = dati_dinamici_dict[nome].index[-n:]

    # Ottieni le date corrispondenti dal DataFrame originale
    date_test = df_originali[nome].loc[indici_test, 'Data'].values

    # Plot con date sull'asse X
    plt.figure(figsize=(14,6))
    plt.plot(date_test, y_true, label='Valori Reali non scalati', color='blue')
    plt.plot(date_test, y_pred, label='Previsioni LightGBM non scalate', color='red', linestyle='--')
    plt.title(f'Confronto Valori non scalati Reali vs Previsioni - {nome.capitalize()}')
    plt.xlabel('Data')
    plt.ylabel('Prezzo Chiusura')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""#3.2.2 Plot Previsioni Riscalate"""

# Funzione per invertire lo scaling e ottenere i valori originali
def inverse_scaling(y_scaled, dati_dinamici, parametri_scaling):
    n = len(y_scaled)
    indici_test = dati_dinamici.index[-n:]
    rolling_mean = parametri_scaling.loc[indici_test, 'Ultimo_mean'].values
    rolling_std = parametri_scaling.loc[indici_test, 'Ultimo_std'].values
    return y_scaled * rolling_std + rolling_mean
y_true_orig_dict = {}
y_pred_orig_dict = {}

df_originali = {'ferrari': ferrari,'amazon': amazon,'oro': oro,'intesa': intesa,'mib': mib}

for nome in ['ferrari', 'amazon', 'oro', 'intesa', 'mib']:
    # Numero di previsioni
    n = len(y_true_all[nome])

    # Indici delle ultime n righe del DataFrame dati_dinamici
    indici_test = dati_dinamici_dict[nome].index[-n:]

    # Recupera le date corrispondenti dai DataFrame originali
    date_test = df_originali[nome].loc[indici_test, 'Data'].values

    # Inversione dello scaling
    y_true_orig_dict[nome] = inverse_scaling(y_true_all[nome], dati_dinamici_dict[nome], parametri_scaling_dict[nome])
    y_pred_orig_dict[nome] = inverse_scaling(y_pred_lgb_all[nome], dati_dinamici_dict[nome], parametri_scaling_dict[nome])

    # Plot con date reali
    plt.figure(figsize=(14,6))
    plt.plot(date_test, y_true_orig_dict[nome], label='Valori Reali', color='blue')
    plt.plot(date_test, y_pred_orig_dict[nome], label='Previsioni LightGBM', color='red', linestyle='--')
    plt.title(f'Confronto Valori Reali vs Previsioni - {nome.capitalize()}')
    plt.xlabel('Data')
    plt.ylabel('Prezzo Chiusura')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()