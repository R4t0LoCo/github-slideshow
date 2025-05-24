import numpy as np
import random
from itertools import product
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import joblib

# Definir seed para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Função para criar dados de séries temporais sintéticos
def create_time_series_data(n_samples=1000, n_features=5, sequence_length=10):
    # Gerar dados base
    X_base, y_base = make_regression(n_samples=n_samples + sequence_length, 
                                     n_features=n_features, noise=0.1, random_state=42)
    X_sequences = []
    y_sequences = []
    for i in range(len(X_base) - sequence_length):
        X_sequences.append(X_base[i:i+sequence_length])
        y_sequences.append(y_base[i+sequence_length])
    return np.array(X_sequences), np.array(y_sequences)

# Criar dataset de séries temporais
sequence_length = 10
X, y = create_time_series_data(n_samples=1000, n_features=5, sequence_length=sequence_length)
print(f"Formato dos dados: X={X.shape}, y={y.shape}")

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

from itertools import combinations

n_features = X.shape[2] # Should be 5 from the data generation
feature_indices = list(range(n_features))
all_feature_subsets = []
for i in range(1, n_features + 1):
    for subset in combinations(feature_indices, i):
        all_feature_subsets.append(list(subset))
# print(f"Generated {len(all_feature_subsets)} feature subsets.")

# This is the global list to store results from all subsets
all_subsets_overall_results = [] 

# param_combinations and actions should be defined once, as they are for LSTM HPs
# and don't change per subset.
param_combinations = generate_full_param_space(model_space) # model_space is LSTM only
actions = list(range(len(param_combinations)))
STATE_SIZE = 12 # This was set in a previous subtask

print(f"Total LSTM hyperparameter combinations to explore per subset: {len(param_combinations)}")

for current_subset_indices in all_feature_subsets:
    print(f"\n{'='*70}")
    print(f"PROCESSING FEATURE SUBSET: {current_subset_indices}")
    print(f"{'='*70}")

    # Filter Data for the current subset
    X_train_subset = X_train[:, :, current_subset_indices]
    X_val_subset = X_val[:, :, current_subset_indices]
    X_test_subset = X_test[:, :, current_subset_indices]
    X_train_full_subset = X_train_full[:, :, current_subset_indices]

    print(f"  Subset Data Shapes: X_train_subset={X_train_subset.shape}, X_val_subset={X_val_subset.shape}")

    # RL Agent Re-initialization for this subset
    agent = SimpleRLAgent(actions, epsilon=0.4, state_size=STATE_SIZE) # Use defined actions

    # Variables to track best for THIS SUBSET's RL run
    current_subset_best_mse_val = float('inf')
    current_subset_best_params_val = None 
    current_subset_best_model_val_instance = None
    subset_results_history_for_report = [] # For this subset's RL progress

    # --- RL TRAINING LOOP (adapted for current subset) ---
    max_episodes_subset = 5 # Can be adjusted, e.g., make it smaller for subsets
    subset_start_time = time()
    print(f"  Starting RL training for subset {current_subset_indices} with {max_episodes_subset} episodes...")
    current_rl_state_subset = np.zeros(STATE_SIZE) # Initial state for this subset's agent

    for episode in range(max_episodes_subset):
        action_idx = agent.get_action(current_rl_state_subset)
        chosen_params_tuple = param_combinations[action_idx] # param_combinations is LSTM HPs
        model_name_eval = chosen_params_tuple[0] # Should always be 'LSTM'
        
        # print(f"    [{episode+1}/{max_episodes_subset}] Subset {current_subset_indices} - Action:{action_idx} -> {model_name_eval}")
        
        mse_eval, evaluated_model_instance = evaluate_pipeline(
            chosen_params_tuple, 
            X_train_subset, y_train, # Use subset data
            X_val_subset, y_val      # Use subset data
        )
        
        if mse_eval != float('inf') and evaluated_model_instance is not None:
            reward_eval = -mse_eval
            subset_results_history_for_report.append((mse_eval, action_idx, chosen_params_tuple))
            next_rl_state_subset = params_to_state(chosen_params_tuple)
            agent.update(current_rl_state_subset, action_idx, reward_eval, next_rl_state_subset)
            current_rl_state_subset = next_rl_state_subset
            
            if mse_eval < current_subset_best_mse_val:
                current_subset_best_mse_val = mse_eval
                current_subset_best_params_val = chosen_params_tuple
                current_subset_best_model_val_instance = evaluated_model_instance
                # print(f"      ✓ New best MSE for subset {current_subset_indices}: {mse_eval:.4f}")
        else:
            # print(f"      ✗ Failed eval for subset {current_subset_indices} with Action {action_idx}")
            agent.update(current_rl_state_subset, action_idx, -1000.0, None) # Heavy penalty

        if (episode + 1) % 5 == 0: # Or some other frequency
            agent.decay_epsilon(0.90)
            # print(f"      Subset {current_subset_indices} - Episode {episode+1} done. Epsilon: {agent.epsilon:.4f}")

    print(f"  RL training for subset {current_subset_indices} completed. Best Val MSE: {current_subset_best_mse_val:.4f}")

    # --- Subset-Specific Test Evaluation ---
    subset_mse_on_test = float('inf')
    if current_subset_best_params_val:
        print(f"    Re-training best model for subset {current_subset_indices} on full training data for test evaluation...")
        _ , model_class_for_test, _, hp_dict_for_test = current_subset_best_params_val
        
        final_test_model = model_class_for_test(**hp_dict_for_test)
        final_test_model.fit(X_train_full_subset, y_train_full) # Fit on combined train+val for this subset
        predictions_test = final_test_model.predict(X_test_subset)
        subset_mse_on_test = mean_squared_error(y_test, predictions_test)
        
        print(f"      MSE on Test Set for subset {current_subset_indices}: {subset_mse_on_test:.4f}")

        all_subsets_overall_results.append({
            'subset_indices': current_subset_indices,
            'best_mse_validation': current_subset_best_mse_val,
            'best_hyperparameters': current_subset_best_params_val[3], # hp_dict
            'best_model_state_encoding': params_to_state(current_subset_best_params_val),
            'mse_on_test_set': subset_mse_on_test,
            'trained_model_instance': final_test_model # Keep instance for saving
        })
    else:
        print(f"    No successful model found for subset {current_subset_indices} during RL phase to evaluate on test set.")
        all_subsets_overall_results.append({
            'subset_indices': current_subset_indices,
            'best_mse_validation': float('inf'),
            'best_hyperparameters': None,
            'best_model_state_encoding': None,
            'mse_on_test_set': float('inf'),
            'trained_model_instance': None
        })
    
    subset_str = "_".join(map(str, current_subset_indices))
    plot_learning_curve(agent.history, 
                        title=f"RL Curve for Subset {current_subset_indices}",
                        filename=f"training_progress_subset_{subset_str}.png")

    # Save the trained model for the current subset
    current_result_dict = all_subsets_overall_results[-1] # Get the dict we just added
    if current_result_dict['trained_model_instance'] is not None:
        subset_model_filename = f"best_model_subset_{subset_str}.pkl"
        try:
            joblib.dump(current_result_dict['trained_model_instance'], subset_model_filename)
            print(f"✓ Modelo do subset salvo como '{subset_model_filename}'")
        except Exception as e:
            print(f"Erro ao salvar o modelo do subset {current_subset_indices}: {e}")

# --- END OF OUTER SUBSET LOOP ---

# Final Reporting
print("\n" + "="*70)
print("OVERALL RESULTS ACROSS ALL FEATURE SUBSETS")
print("="*70)

# Sort all_subsets_overall_results by 'mse_on_test_set' or 'best_mse_validation'
# Let's sort by 'mse_on_test_set' as it's the ultimate performance metric.
# Handle potential float('inf') values in sorting by placing them at the end.
sorted_results = sorted(
    all_subsets_overall_results, 
    key=lambda x: (x['mse_on_test_set'] == float('inf'), x['mse_on_test_set'])
)

for result in sorted_results:
    print(f"\nFeature Subset: {result['subset_indices']}")
    val_mse_str = f"{result['best_mse_validation']:.4f}" if result['best_mse_validation'] != float('inf') else "N/A (No model succeeded)"
    test_mse_str = f"{result['mse_on_test_set']:.4f}" if result['mse_on_test_set'] != float('inf') else "N/A (No model succeeded or test failed)"
    
    print(f"  Best Val MSE: {val_mse_str}")
    print(f"  Test MSE: {test_mse_str}")
    print(f"  Best Hyperparameters: {result['best_hyperparameters']}")
    # print(f"  Encoded State: {result['best_model_state_encoding']}") # Can be verbose

# Comment out old reporting and saving
# report_df = create_model_report(top_5_results_for_report, param_combinations)
# print(report_df.to_string(index=False))
# import joblib
# if final_trained_model_for_test is not None:
#     try:
#         joblib.dump(final_trained_model_for_test, 'best_model_final.pkl')
#     except Exception as e:
#         print(f"Erro ao salvar o modelo final: {e}")
# else:
#     print("Nenhum modelo final treinado para salvar.")

print(f"\n{'='*70}")
print("FEATURE SUBSET ANALYSIS COMPLETED!")
print("="*70)
# total_script_time = time() - overall_start_time # If overall_start_time was defined at script beginning
# print(f"Total script execution time: {total_script_time:.2f} seconds")

def plot_overall_subset_performance(all_results, output_filename="subset_performance_summary.png"):
    if not all_results:
        print("Nenhum resultado de subset para plotar.")
        return

    # Filter out results where 'mse_on_test_set' might be None or non-numeric if necessary,
    # or ensure it's always a float (even float('inf')). For this plot, we assume valid numbers or inf.
    
    # Using original order as subsets are generated systematically.
    # If sorting is desired, uncomment:
    # all_results_sorted = sorted(all_results, key=lambda x: (x['mse_on_test_set'] == float('inf'), x['mse_on_test_set']))
    
    subset_labels = [str(res['subset_indices']) for res in all_results]
    # Replace inf with a large number for plotting, or handle separately if preferred.
    # For simplicity, we'll plot inf as a very high bar if it occurs.
    test_mses = [res['mse_on_test_set'] if res['mse_on_test_set'] != float('inf') else 1e9 for res in all_results] # Cap inf for plot

    plt.figure(figsize=(max(15, len(subset_labels) * 0.6), 9)) # Dynamic width, increased height
    bars = plt.bar(subset_labels, test_mses, color='mediumseagreen')
    plt.xlabel("Feature Subset Indices")
    plt.ylabel("Mean Squared Error (on Test Set)")
    plt.title("Test MSE by Feature Subset (Lower is Better)")
    plt.xticks(rotation=75, ha='right', fontsize=8) # Rotate labels for readability
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for i, bar in enumerate(bars):
        yval = all_results[i]['mse_on_test_set'] # Use original MSE for text
        if yval != float('inf'):
            plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f'{yval:.3f}', 
                     va='bottom', ha='center', fontsize=7, rotation=0)
        else:
            plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), 'N/A', 
                     va='bottom', ha='center', fontsize=7, rotation=0)

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de resumo do desempenho dos subsets salvo como '{output_filename}'")
    except Exception as e:
        print(f"Erro ao salvar o gráfico de resumo: {e}")
    plt.close()

# Call this function after the main loop over subsets:
plot_overall_subset_performance(all_subsets_overall_results)


class LSTMRegressor:
    def __init__(self, lstm_units=50, dropout_rate=0.2, learning_rate=0.001, 
                 epochs=50, batch_size=32, patience=10,
                 num_lstm_layers=1, use_bidirectional=False, 
                 num_dense_layers=1, dense_units=32, use_batch_norm=False):
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.num_lstm_layers = num_lstm_layers
        self.use_bidirectional = use_bidirectional
        self.num_dense_layers = num_dense_layers
        self.dense_units = dense_units
        self.use_batch_norm = use_batch_norm
        self.model = None
        self.scaler = None

    def build_model(self, input_shape):
        model = Sequential()
        is_first_layer = True

        # LSTM Layers
        for i in range(self.num_lstm_layers):
            units = self.lstm_units // (2**i)
            units = max(units, 16)  # Ensure minimum units
            return_sequences = (i < self.num_lstm_layers - 1)

            lstm_layer_args = {'units': units, 'return_sequences': return_sequences}
            if is_first_layer:
                lstm_layer_args['input_shape'] = input_shape
            
            current_lstm_layer = LSTM(**lstm_layer_args)

            if self.use_bidirectional:
                # If Bidirectional is used on the first layer, it should get the input_shape
                if is_first_layer:
                    # Remove input_shape from lstm_layer_args if Bidirectional handles it
                    del lstm_layer_args['input_shape'] 
                    current_lstm_layer = LSTM(**lstm_layer_args) # Recreate LSTM without input_shape
                    layer_to_add = keras.layers.Bidirectional(current_lstm_layer, input_shape=input_shape)
                else:
                    layer_to_add = keras.layers.Bidirectional(current_lstm_layer)
            else:
                layer_to_add = current_lstm_layer
            
            model.add(layer_to_add)
            is_first_layer = False

            if self.use_batch_norm:
                model.add(keras.layers.BatchNormalization())
            model.add(Dropout(self.dropout_rate))

        # Dense Layers
        for i in range(self.num_dense_layers):
            model.add(Dense(self.dense_units, activation='relu'))
            if self.use_batch_norm:
                model.add(keras.layers.BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Final Output Layer
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def fit(self, X, y):
        # Determine input_shape from the data passed to fit
        self.input_shape = (X.shape[1], X.shape[2]) # (sequence_length, num_selected_features)
        
        self.scaler = StandardScaler() 
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled_reshaped = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(X.shape)
        
        # build_model is now called with the dynamically determined input_shape
        self.model = self.build_model(self.input_shape) 
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, 
                                       restore_best_weights=True, verbose=0)
        self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        return self

    def predict(self, X):
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não foi treinado ainda!")
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(X.shape)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()

model_space = {
    'LSTM': {
        'model_class': LSTMRegressor,
        'params': {
            'lstm_units': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'epochs': [30, 50],
            'batch_size': [16, 32],
            'num_lstm_layers': [1, 2],
            'use_bidirectional': [True, False],
            'num_dense_layers': [1, 2],
            'dense_units': [32, 64],
            'use_batch_norm': [True, False]
        }
    }
}

def generate_full_param_space(model_space):
    full_space = []
    for model_name, config in model_space.items():
        model_class = config['model_class']
        hp_keys = list(config['params'].keys())
        hp_values = list(config['params'].values())
        for hp_comb in product(*hp_values):
            param_dict = dict(zip(hp_keys, hp_comb))
            # Only LSTM models are expected now, preprocessor is always None
            full_space.append((model_name, model_class, None, param_dict))
    return full_space

def evaluate_pipeline(params, X_train, y_train, X_val, y_val):
    model_name, model_class, preprocessor, hp_dict = params # preprocessor will be None
    try:
        # Logic is now only for LSTM
        model_instance = model_class(**hp_dict)
        model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_val)
        model_to_return = model_instance
        mse = mean_squared_error(y_val, predictions)
        return mse, model_to_return
    except Exception as e:
        return float('inf'), None

def params_to_state(params):
    model_name, model_class_or_constructor, preprocessor, hp_dict = params
    model_code = list(model_space.keys()).index(model_name) # Should always be 0 for LSTM
    preprocessor_code = 0 # Preprocessor is always None for LSTM
    hp_values = []
    # Only LSTM logic is needed
    param_order = [
        'lstm_units', 'dropout_rate', 'learning_rate', 'epochs', 'batch_size',
        'num_lstm_layers', 'use_bidirectional', 'num_dense_layers', 'dense_units', 'use_batch_norm'
    ]
    norm_factors = {
        'lstm_units': 128.0, 'dropout_rate': 1.0, 'learning_rate': 0.01, 'epochs': 50.0, 'batch_size': 32.0,
        'num_lstm_layers': 2.0, 'use_bidirectional': 1.0, 'num_dense_layers': 2.0, 'dense_units': 64.0, 'use_batch_norm': 1.0
    }
    for k in param_order:
        v = hp_dict.get(k)
        if isinstance(v, (int, float)):
            hp_values.append(v / norm_factors.get(k, 1.0))
        elif isinstance(v, bool): # For use_bidirectional, use_batch_norm
            hp_values.append(1.0 if v else 0.0)
        elif isinstance(v, tuple): # Should not happen with current LSTM params
            hp_values.append(sum(v) / norm_factors.get(k, 1.0 if sum(v) == 0 else sum(v) * 2))
        elif isinstance(v, str): # Should not happen with current LSTM params
            hp_values.append(float(abs(hash(v))) % 100 / 100.0)
        elif v is None:
            hp_values.append(0.0)
        else:
            hp_values.append(0.0) # Default for unexpected types
    max_hp_features = 10 # 5 original + 5 new
    while len(hp_values) < max_hp_features:
        hp_values.append(0.0)
    current_state_array = np.array(
        [model_code / len(model_space), preprocessor_code / 1.0] + hp_values[:max_hp_features]
    )
    expected_state_size = 2 + max_hp_features # 2 + 10 = 12
    if len(current_state_array) < expected_state_size:
        padding = np.zeros(expected_state_size - len(current_state_array))
        current_state_array = np.concatenate((current_state_array, padding))
    return current_state_array.round(4)

class SimpleRLAgent:
    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=0.9, state_size=7):
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.state_size = state_size
        self.history = []

    def get_q_value(self, state_tuple, action_idx):
        return self.q_table.get(state_tuple, {}).get(action_idx, 0.0)

    def get_action(self, state):
        state_key = tuple(state)
        if random.random() < self.epsilon or state_key not in self.q_table:
            return random.choice(self.actions)
        q_values_for_state = self.q_table.get(state_key, {act: 0.0 for act in self.actions})
        if not q_values_for_state:
             return random.choice(self.actions)
        return max(q_values_for_state, key=q_values_for_state.get)

    def update(self, state, action_idx, reward, new_state=None):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {act: 0.0 for act in self.actions}
        old_value = self.q_table[state_key].get(action_idx, 0.0)
        if new_state is not None:
            new_state_key = tuple(new_state)
            if new_state_key not in self.q_table:
                self.q_table[new_state_key] = {act: 0.0 for act in self.actions}
            q_values_new_state = self.q_table.get(new_state_key, {act: 0.0 for act in self.actions})
            if not q_values_new_state:
                next_max = 0.0
            else:
                next_max = max(q_values_new_state.values())
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        else:
            new_value = old_value + self.alpha * (reward - old_value)
        self.q_table[state_key][action_idx] = new_value
        self.history.append({'action_idx': action_idx, 'reward': reward, 'state': state_key, 'q_value_updated': new_value})

    def decay_epsilon(self, decay_rate=0.95):
        self.epsilon = max(0.01, self.epsilon * decay_rate)

def plot_learning_curve(agent_history, title="Progresso do Treinamento RL", filename="training_progress.png"):
    plt.figure(figsize=(12, 6))
    rewards = [item['reward'] for item in agent_history]
    if not rewards:
        print("Nenhuma recompensa registrada para plotar.")
        plt.text(0.5, 0.5, "Sem dados de recompensa.", ha='center', va='center')
        plt.title(title)
        plt.xlabel('Episódios')
        plt.ylabel('Recompensa (-MSE)')
        plt.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return
    plt.subplot(1, 2, 1)
    plt.plot(rewards, 'b-', alpha=0.7, label='Recompensa por Episódio')
    plt.title("Recompensas por Episódio")
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa (-MSE)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    window = 10
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        plt.plot(moving_avg, 'r-', linewidth=2, label=f'Média Móvel (janela={window})')
    else:
        plt.plot(rewards, 'r-', linewidth=2, label='Recompensas (sem média móvel)')
    plt.title(f"Média Móvel das Recompensas")
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa Média (-MSE)')
    plt.legend()
    plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Curva de aprendizado salva como '{filename}'")

def create_model_report(results, param_combinations):
    report_data = []
    for rank, (mse, model_idx, evaluated_params) in enumerate(results):
        model_name, _, preprocessor_obj, hp_dict = evaluated_params
        prep_name = "None"
        if preprocessor_obj:
            prep_name = type(preprocessor_obj).__name__
        report_data.append({
            'Rank': rank + 1,
            'Model': model_name,
            'Preprocessor': prep_name,
            'Hyperparameters': str(hp_dict),
            'MSE': mse
        })
    return pd.DataFrame(report_data)

# The main training and evaluation logic will be moved inside the loop above.
# For now, I'm removing the old main execution block, as it will be integrated
# into the per-subset loop structure.
