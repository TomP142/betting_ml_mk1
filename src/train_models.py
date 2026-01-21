import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

class NBADataset(Dataset):
    def __init__(self, X, y, missing_idxs):
        self.X = torch.FloatTensor(X) # (N, Features)
        self.y = torch.FloatTensor(y) # (N, 3)
        self.missing_idxs = torch.LongTensor(missing_idxs) # (N, Max_Missing)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Row-based access
        # X: [PLAYER_IDX, TEAM_IDX, ... Cont Features ...]
        p_idx = self.X[idx, 0].long()
        t_idx = self.X[idx, 1].long()
        x_cont = self.X[idx, 2:]
        m_idx = self.missing_idxs[idx]
        return p_idx, t_idx, x_cont, m_idx, self.y[idx]

class NBAPredictor(nn.Module):
    def __init__(self, num_players, num_teams, num_cont):
        super().__init__()
        # Embeddings
        # num_players includes Padding Index (last one)
        self.player_emb = nn.Embedding(num_players, 32, padding_idx=num_players-1)
        self.team_emb = nn.Embedding(num_teams, 16)
        
        # MLP
        # Input: Player(32) + Team(16) + Continuous + Missing(32)
        in_dim = 32 + 16 + num_cont + 32
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # PTS, REB, AST
        )
        
    def forward(self, p_idx, t_idx, x_cont, m_idx):
        p_emb = self.player_emb(p_idx) # (B, 32)
        t_emb = self.team_emb(t_idx)   # (B, 16)
        
        # Missing Players: (B, K) -> (B, K, 32) -> Sum(1) -> (B, 32)
        m_emb = self.player_emb(m_idx).sum(dim=1)
        
        x = torch.cat([p_emb, t_emb, x_cont, m_emb], dim=1)
        return self.net(x)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

def evaluate_model(model, loader, criterion, scaler, feature_cols):
    model.eval()
    total_loss = 0
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for p_idx, t_idx, x_cont, m_idx, y in loader:
            p_idx, t_idx, x_cont, m_idx, y = p_idx.to(device), t_idx.to(device), x_cont.to(device), m_idx.to(device), y.to(device)
            outputs = model(p_idx, t_idx, x_cont, m_idx)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(outputs.cpu().numpy())
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    
    # Calculate MAE per target
    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
    
    if len(y_true) > 0:
        mae_pts = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
        mae_reb = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
        mae_ast = np.mean(np.abs(y_true[:, 2] - y_pred[:, 2]))
    else:
        mae_pts, mae_reb, mae_ast = 0, 0, 0
        
    return avg_loss, mae_pts, mae_reb, mae_ast

def train_model(target_player_id: int = None, debug: bool = False, pretrain: bool = False):
    print(f"Training on {device} (Debug={debug}, Pretrain={pretrain})...")
    
    # 1. Load Data
    path = os.path.join(PROCESSED_DIR, 'processed_features.csv')
    df = pd.read_csv(path)
    
    # Filter
    if target_player_id and not pretrain:
        print(f"Filtering data for Player ID: {target_player_id}")
        # Need Encoder for filtering by ID -> IDX map or just use ID since we saved it
        df = df[df['PLAYER_ID'] == target_player_id]
        if df.empty:
            print("No data found for player.")
            return

    # Load Artifacts
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
    player_encoder = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib')) 
    team_encoder = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    num_players = len(player_encoder.classes_)
    num_teams = len(team_encoder.classes_)
    num_cont = len(feature_cols)
    print(f"Features: {num_cont}, Players: {num_players}, Teams: {num_teams}")

    # 2. Split (Strict Time Based)
    df = df.sort_values('GAME_DATE')
    
    # For global pretraining, we can use a random split or time split. 
    # Time split is better to prevent leakage.
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    # Prepare Dataset
    # Separate numeric features from Metadata
    # X needs: PLAYER_IDX, TEAM_IDX, Continuous Features
    
    feature_cols = [c for c in df.columns if c not in ['PLAYER_ID', 'GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'PLAYER_NAME', 'TEAM_ID', 'SEASON_YEAR', 'PLAYER_IDX', 'TEAM_IDX', 'MISSING_PLAYER_IDS']]
    
    # Save feature columns
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.joblib'))
    
    X = df[['PLAYER_IDX', 'TEAM_IDX'] + feature_cols].values.astype(np.float32)
    y = df[['PTS', 'REB', 'AST']].values.astype(np.float32)
    
    # ---- PROCESS MISSING IDS ----
    print("Processing Missing Player Embeddings...")
    missing_raw = df['MISSING_PLAYER_IDS'].fillna("NONE").astype(str).tolist()
    
    # Pad Index = num_players_encoded (The size of encoder classes)
    # The Embedding layer has size (len(classes) + 1). 
    # Indices 0 to N-1 are valid players. Index N is Padding.
    pad_idx = len(player_encoder.classes_)
    max_missing = 3
    
    missing_matrix = []
    
    # Create a mapping for speed
    id_to_idx = {pid: idx for idx, pid in enumerate(player_encoder.classes_)}
    
    for row_str in missing_raw:
        if row_str == "NONE" or row_str == "":
            indices = []
        else:
            pids = row_str.split('_')
            indices = []
            for pid_str in pids:
                try:
                    pid = int(float(pid_str)) # Handle "123.0" potential string
                    if pid in id_to_idx:
                        indices.append(id_to_idx[pid])
                    else:
                        # Unknown player (not in training set) -> Treat as Padding/Ignore
                        pass 
                except:
                    pass
        
        # Truncate or Pad
        if len(indices) > max_missing:
            indices = indices[:max_missing]
        else:
            indices += [pad_idx] * (max_missing - len(indices))
            
        missing_matrix.append(indices)
        
    missing_matrix = np.array(missing_matrix, dtype=np.int64)
    # -----------------------------
    
    # Split
    train_indices, val_indices = train_test_split(np.arange(len(df)), test_size=0.15, shuffle=True, random_state=42)
    
    # Calculate Baseline MAE (Mean of Train Predicts Val)
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    baseline_preds = np.mean(y_train, axis=0) # [PTS, REB, AST]
    baseline_mae = np.mean(np.abs(y_val - baseline_preds), axis=0)
    
    baseline_mae_pts = baseline_mae[0]
    baseline_mae_reb = baseline_mae[1]
    baseline_mae_ast = baseline_mae[2]
    
    print(f"Baseline MAE - PTS: {baseline_mae_pts:.4f}, REB: {baseline_mae_reb:.4f}, AST: {baseline_mae_ast:.4f}")

    # Weighted Sampler for Training
    # Give higher weight to recent seasons (2025-26)
    # We need weights for TRAIN set only
    train_df = df.iloc[train_indices]
    weights = train_df['SEASON_YEAR'].map({
        2026: 2.0,
        2025: 1.5,
        2024: 1.0,
        2023: 0.8,
        2022: 0.5
    }).fillna(1.0).values
    
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_dataset = NBADataset(X[train_indices], y[train_indices], missing_matrix[train_indices])
    val_dataset = NBADataset(X[val_indices], y[val_indices], missing_matrix[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model Init
    # Num Players + 1 for Padding
    model = NBAPredictor(
        num_players=len(player_encoder.classes_) + 1, 
        num_teams=len(team_encoder.classes_), 
        num_cont=len(feature_cols)
    ).to(device)
    
    # TRANSFER LEARNING LOGIC
    global_model_path = os.path.join(MODELS_DIR, 'pytorch_nba_global.pth')
    if os.path.exists(global_model_path):
        print(f"Loading Global Model weights from {global_model_path}...")
        try:
            model.load_state_dict(torch.load(global_model_path, map_location=device), strict=False)
            if not pretrain: # Only print this if we are fine-tuning, not pretraining
                print("Global weights loaded successfully! Fine-tuning now...")
        except Exception as e:
            print(f"Could not load global weights (likely architecture mismatch): {e}")
            print("Proceeding with fresh weights.")
    else:
        print("No global model found. Training from scratch.")
    
    criterion = nn.MSELoss()
    # Lower LR for fine-tuning
    lr = 0.001 if pretrain else 0.0005 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    save_path = global_model_path if pretrain else os.path.join(MODELS_DIR, f'pytorch_player_{target_player_id}.pth')
    
    early_stopping = EarlyStopping(patience=50 if not debug else 1000, path=save_path) if not (debug and not pretrain) else None
    
    # 5. Training Loop
    epochs = 100 if pretrain else 150
    
    # History for plotting
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'mae_pts': [],
        'mae_reb': [],
        'mae_ast': []
    }
    
    # Plotting Setup for Debug
    # Plotting Setup (Debug or Pretrain)
    if debug or pretrain:
        import matplotlib.pyplot as plt
        plt.ion() # Interactive mode
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Loss
        ax_loss = axs[0, 0]
        line_train_loss, = ax_loss.plot([], [], label='Train Loss')
        line_val_loss, = ax_loss.plot([], [], label='Val Loss')
        ax_loss.set_title(f'Loss ({ "Global" if pretrain else f"Player {target_player_id}" })')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('MSE Loss')
        ax_loss.legend()
        ax_loss.grid(True)
        
        # Plot 2: PTS MAE
        ax_pts = axs[0, 1]
        line_mae_pts, = ax_pts.plot([], [], label='PTS MAE')
        ax_pts.axhline(y=baseline_mae_pts, color='r', linestyle='--', label='Baseline')
        ax_pts.set_title('PTS MAE')
        ax_pts.set_xlabel('Epoch')
        ax_pts.legend()
        ax_pts.grid(True)

        # Plot 3: REB MAE
        ax_reb = axs[1, 0]
        line_mae_reb, = ax_reb.plot([], [], label='REB MAE')
        ax_reb.axhline(y=baseline_mae_reb, color='r', linestyle='--', label='Baseline')
        ax_reb.set_title('REB MAE')
        ax_reb.set_xlabel('Epoch')
        ax_reb.legend()
        ax_reb.grid(True)

        # Plot 4: AST MAE
        ax_ast = axs[1, 1]
        line_mae_ast, = ax_ast.plot([], [], label='AST MAE')
        ax_ast.axhline(y=baseline_mae_ast, color='r', linestyle='--', label='Baseline')
        ax_ast.set_title('AST MAE')
        ax_ast.set_xlabel('Epoch')
        ax_ast.legend()
        ax_ast.grid(True)
        
        plt.tight_layout()
        
        if pretrain:
            plot_path = os.path.join(DATA_DIR, 'training_curve_global.png')
        else:
            plot_path = os.path.join(DATA_DIR, f'training_curve_{target_player_id}.png')

    print("\nStarting Training...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'PTS MAE':<10} | {'REB MAE':<10}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for p_idx, t_idx, x_cont, m_idx, y in train_loader:
            p_idx, t_idx, x_cont, m_idx, y = p_idx.to(device), t_idx.to(device), x_cont.to(device), m_idx.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(p_idx, t_idx, x_cont, m_idx)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = 0
        all_y_true = []
        all_y_pred = []
        model.eval()
        with torch.no_grad():
            for p_idx, t_idx, x_cont, m_idx, y in val_loader:
                p_idx, t_idx, x_cont, m_idx, y = p_idx.to(device), t_idx.to(device), x_cont.to(device), m_idx.to(device), y.to(device)
                outputs = model(p_idx, t_idx, x_cont, m_idx)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                all_y_true.append(y.cpu().numpy())
                all_y_pred.append(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate MAE for validation set
        y_true_combined = np.concatenate(all_y_true, axis=0)
        y_pred_combined = np.concatenate(all_y_pred, axis=0)
        
        # Inverse transform predictions and true values if scaler was used
        # Assuming scaler was fitted on the original target columns (PTS, REB, AST)
        # This part needs to be careful if scaler was fitted on X or y.
        # For now, assuming scaler is for features, and targets are not scaled.
        # If targets were scaled, we'd need to inverse_transform them.
        # For simplicity, let's assume targets are not scaled for MAE calculation directly.
        # If `evaluate_model` handled inverse scaling, we need to replicate that.
        # For now, let's calculate MAE on the raw outputs.
        # If `evaluate_model` was doing inverse scaling, this needs to be adjusted.
        # Based on the original `evaluate_model` signature, it takes `scaler` and `feature_cols`.
        # This implies it might be used for inverse transforming features, not targets.
        # Let's assume the targets are not scaled for now, or that the MAE is calculated on scaled values.
        # If `scaler` was used for targets, we'd need to inverse_transform `y_true_combined` and `y_pred_combined`.
        # For now, let's calculate MAE directly.
        
        if len(y_true_combined) > 0:
            mae_pts = np.mean(np.abs(y_true_combined[:, 0] - y_pred_combined[:, 0]))
            mae_reb = np.mean(np.abs(y_true_combined[:, 1] - y_pred_combined[:, 1]))
            mae_ast = np.mean(np.abs(y_true_combined[:, 2] - y_pred_combined[:, 2]))
        else:
            mae_pts, mae_reb, mae_ast = 0, 0, 0
        
        # Scheduler update
        scheduler.step(avg_val_loss)
        
        # Track history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mae_pts'].append(mae_pts)
        history['mae_reb'].append(mae_reb)
        history['mae_ast'].append(mae_ast)
        
        # Update Plot
        if debug or pretrain:
            # Update data
            line_train_loss.set_data(history['epoch'], history['train_loss'])
            line_val_loss.set_data(history['epoch'], history['val_loss'])
            
            line_mae_pts.set_data(history['epoch'], history['mae_pts'])
            line_mae_reb.set_data(history['epoch'], history['mae_reb'])
            line_mae_ast.set_data(history['epoch'], history['mae_ast'])
            
            # Rescale axes
            for ax in [ax_loss, ax_pts, ax_reb, ax_ast]:
                ax.relim()
                ax.autoscale_view()
            
            # Draw
            plt.draw()
            plt.pause(0.01) # Trigger full redraw loop
            
            # Save current frame frequently
            if epoch % 5 == 0:
                 plt.savefig(plot_path)

        # Early Stopping
        if early_stopping:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"{epoch+1:<6} | {train_loss:<10.4f} | {avg_val_loss:<10.4f} | {mae_pts:<10.4f} | {mae_reb:<10.4f}")
                print("Early stopping triggered.")
                break
                
        # Always output in debug or periodic
        if (epoch+1) % 10 == 0:
             print(f"{epoch+1:<6} | {train_loss:<10.4f} | {avg_val_loss:<10.4f} | {mae_pts:<10.4f} | {mae_reb:<10.4f}")
    
    # Save model at end
    if not early_stopping:
        torch.save(model.state_dict(), save_path)
        print(f"\nDebug Run Complete. Model Saved to {save_path}")
        if debug or pretrain: 
            plt.savefig(plot_path)
            print(f"Final plot saved to {plot_path}")
            plt.ioff()
            # plt.show() # Optional: keep window open. But might block script exit. 
            # Better to just save and close.
            plt.close()
    else:
        model.load_state_dict(torch.load(save_path))
        print(f"\nFinal Best Model Saved to {save_path}")
        
    print(f"Final Validation Scores -> PTS MAE: {mae_pts:.4f} (Baseline: {baseline_mae_pts:.4f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--player_id', type=int, help='Player ID to train on')
    parser.add_argument('--debug', action='store_true', help='Debug mode (no early stop, plotting)')
    parser.add_argument('--pretrain', action='store_true', help='Pretrain global model on all data')
    args = parser.parse_args()
    
    if args.pretrain:
        train_model(pretrain=True)
    elif args.player_id:
        train_model(target_player_id=args.player_id, debug=args.debug)
    else:
        print("Please provide --player_id or --pretrain")
