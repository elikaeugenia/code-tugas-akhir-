import os
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datareader import ShopeeComment
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model_ringan import TextCNN
from model_sedang import TextCNN
from model_berat import TextCNN
from model_x import TextCNN as TextCNN_X 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import json
from datetime import datetime

# Implementasi Muon optimizer manual
class MuonOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(MuonOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                p.data.add_(buf, alpha=-group['lr'])
        
        return loss
    
# Setup Moun optimizer
Moun = MuonOptimizer

# Confusion Matrix Visualization Function
def plot_confusion_matrix(cm, model_name="TextCNN", class_names=['Negative', 'Positive'], 
                         precision=None, recall=None, f1=None, accuracy=None, 
                         save_path=None, show_plot=False):
    """
    Plot confusion matrix with style similar to the reference image
    """
    plt.figure(figsize=(8, 6))
    
    # Create the heatmap with blue colormap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': ''}, annot_kws={'size': 14, 'weight': 'bold'})
    
    # Set labels and title
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add metrics text below the plot if provided
    if any([precision, recall, f1, accuracy]):
        metrics_text = f"{model_name} Evaluation Metrics:\n"
        if accuracy is not None:
            metrics_text += f"Accuracy: {accuracy:.4f}\n"
        if precision is not None:
            metrics_text += f"Precision: {precision:.4f}\n"
        if recall is not None:
            metrics_text += f"Recall: {recall:.4f}\n"
        if f1 is not None:
            metrics_text += f"F1-Score: {f1:.4f}"
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   verticalalignment='bottom', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return plt.gcf()

# Model Factory Function 
def get_model(model_name, vocab_size, num_classes, dropout):
    if model_name == "ringan":
        return TextCNN(vocab_size=vocab_size, embed_dim=100, num_classes=num_classes, do=dropout)
    elif model_name == "sedang":
        return TextCNN(vocab_size=vocab_size, embed_dim=300, num_classes=num_classes, do=dropout)
    elif model_name == "berat":
        return TextCNN(vocab_size=vocab_size, embed_dim=512, num_classes=num_classes, do=dropout)
    elif model_name == "x":
        return TextCNN_X(vocab_size=vocab_size, embed_dim=256, num_classes=num_classes, do=dropout)
    else:
        raise ValueError(f"Model size '{model_name}' is not supported.")

# Optimizer Factory Function 
def get_optimizer(optimizer_name, model_params, lr):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif optimizer_name =="muon":
        return Moun(model_params, lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

# Training Function
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Akumulasi total loss dan akurasi
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), correct / total

# Enhanced Evaluation Function
def evaluate(model, data_loader, criterion, device, plot_cm=False, model_name="TextCNN", save_path=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix if requested
    fig = None
    if plot_cm:
        fig = plot_confusion_matrix(cm, model_name=model_name, 
                                   precision=precision, recall=recall, 
                                   f1=f1, accuracy=accuracy,
                                   save_path=save_path, show_plot=False)

    return total_loss / len(data_loader), accuracy, precision, recall, f1, cm, fig

# Single Fold Training Function
def train_single_fold(fold, args, device, output_dir, vocab_size):
    """Train a single fold and return the best metrics"""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold}")
    print(f"{'='*50}")
    
    # Create datasets for this fold
    train_dataset = ShopeeComment(
        file_path="dataset.xlsx",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="shopee_datareader_simple_folds.json",
        random_state=2025,
        split="train",
        fold=fold,
        augmentasi_file="augmentasi.json"
    )
    
    val_dataset = ShopeeComment(
        file_path="dataset.xlsx",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="shopee_datareader_simple_folds.json",
        normalization_file="normalization_dict.json", 
        random_state=2025,
        split="val",
        fold=fold,
        augmentasi_file="augmentasi.json",
        typo_prob=0,         
        swap_prob=0,         
        delete_prob=0,      
        synonym_prob=0,      
        phrase_prob=0
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = get_model(args.model_name, vocab_size, args.num_classes, args.dropout).to(device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(), args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    # Display torch summary only for the first fold to avoid clutter
    if fold == 0:
        try:
            from torchinfo import summary
            print(f"\n{'='*60}")
            print(f"Model Summary - TextCNN ({args.model_name})")
            print(f"{'='*60}")
            summary(model, input_size=(args.batch_size, args.max_len), 
                   dtypes=[torch.long], 
                   col_names=["input_size", "output_size", "num_params", "params_percent"])
            print(f"{'='*60}")
        except ImportError:
            print("\nWarning: torchinfo not available. Install with 'pip install torchinfo' for model summary.")
            print(f"Model: TextCNN-{args.model_name}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = get_optimizer(args.optimizer_name, model.parameters(), args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    # Training variables
    best_acc = 0
    best_epoch = 0
    best_metrics = {}
    fold_history = []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        
        # Plot confusion matrix only on the last epoch
        plot_cm = (epoch == args.epochs - 1)
        cm_save_path = os.path.join(output_dir, f'fold_{fold}_epoch_{epoch+1}_confusion_matrix.png') if plot_cm else None
        
        val_loss, val_acc, val_precision, val_recall, val_f1, val_cm, cm_fig = evaluate(
            model, val_loader, criterion, device, 
            plot_cm=plot_cm, 
            model_name=f"TextCNN-{args.model_name} Fold-{fold}",
            save_path=cm_save_path
        )

        print(f"Fold {fold} - Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"Val   Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Store epoch metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'confusion_matrix': val_cm.tolist()
        }
        fold_history.append(epoch_metrics)

        # Log to wandb with fold prefix
        log_dict = {
            f"fold_{fold}/epoch": epoch + 1,
            f"fold_{fold}/train_loss": train_loss,
            f"fold_{fold}/train_acc": train_acc,
            f"fold_{fold}/val_loss": val_loss,
            f"fold_{fold}/val_acc": val_acc,
            f"fold_{fold}/val_precision": val_precision,
            f"fold_{fold}/val_recall": val_recall,
            f"fold_{fold}/val_f1": val_f1,
            f"fold_{fold}/lr": scheduler._last_lr[0] if hasattr(scheduler, '_last_lr') else args.learning_rate
        }

        if cm_fig is not None:
            log_dict[f"fold_{fold}/confusion_matrix"] = wandb.Image(cm_fig)
            plt.close(cm_fig)

        wandb.log(log_dict)
        scheduler.step(val_loss)

        # Check if this is the best epoch for this fold
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_metrics = {
                'fold': fold,
                'epoch': best_epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'confusion_matrix': val_cm.tolist()
            }
            
            # Save best model for this fold
            model_path = os.path.join(output_dir, f'best_model_fold_{fold}.pth')
            torch.save(model.state_dict(), model_path)
            
            # Save best confusion matrix for this fold
            best_cm_save_path = os.path.join(output_dir, f'best_model_fold_{fold}_confusion_matrix.png')
            best_cm_fig = plot_confusion_matrix(val_cm, 
                                               model_name=f"TextCNN-{args.model_name} Fold-{fold} (Best)", 
                                               precision=val_precision, recall=val_recall, 
                                               f1=val_f1, accuracy=val_acc,
                                               save_path=best_cm_save_path, show_plot=False)
            plt.close(best_cm_fig)
            
            # Log best confusion matrix to wandb
            wandb.log({f"fold_{fold}/best_confusion_matrix": wandb.Image(best_cm_save_path)})

    print(f"Fold {fold} completed! Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    
    return best_metrics, fold_history

# Main Training Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_name", type=str, default="adam", choices=["adam", "muon"])
    parser.add_argument("--augment_prob", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="ringan", choices=["ringan", "sedang", "berat", "x"])
    parser.add_argument("--name", type=str, default="experiment")
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    # Initialize wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="cnn_shopeecomment_multifold",
        name=f"multifold_exp_{timestamp}",
        mode="online",
        config=vars(args)
    )

    # Set random seeds
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'models', f'multifold_results_{timestamp}')
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created at: {output_dir}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return

    # Get vocab size 
    temp_dataset = ShopeeComment(
        file_path="dataset.xlsx",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="shopee_datareader_simple_folds.json",
        random_state=2025,
        split="train",
        fold=0,
        augmentasi_file="augmentasi.json"
    )
    vocab_size = temp_dataset.tokenizer.vocab_size
    del temp_dataset  # Clean up

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Vocab size: {vocab_size}")

    # Train all folds
    all_fold_results = []
    all_fold_histories = []
    overall_best_acc = 0
    overall_best_fold = 0
    overall_best_epoch = 0
    overall_best_metrics = {}

    print(f"\n{'='*60}")
    print(f"Starting Multi-Fold Training ({args.num_folds} folds)")
    print(f"{'='*60}")

    # Loop untuk setiap fold 0-4
    for fold in range(args.num_folds):
        fold_best_metrics, fold_history = train_single_fold(fold, args, device, output_dir, vocab_size)
        all_fold_results.append(fold_best_metrics)
        all_fold_histories.append({f'fold_{fold}': fold_history})
        
        # Track fold mana yang menghasilkan accuracy terbaik
        if fold_best_metrics['val_acc'] > overall_best_acc:
            overall_best_acc = fold_best_metrics['val_acc']
            overall_best_fold = fold
            overall_best_epoch = fold_best_metrics['epoch']
            overall_best_metrics = fold_best_metrics.copy()

    # Hitung rata-rata dan standar deviasi dari semua fold
    avg_metrics = {}
    metrics_to_avg = ['val_acc', 'val_precision', 'val_recall', 'val_f1']
    for metric in metrics_to_avg:
        avg_metrics[f'avg_{metric}'] = np.mean([fold[metric] for fold in all_fold_results])
        avg_metrics[f'std_{metric}'] = np.std([fold[metric] for fold in all_fold_results])

    # Print final results
    print(f"\n{'='*60}")
    print("MULTI-FOLD TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Overall Best Performance:")
    print(f"  Best Fold: {overall_best_fold}")
    print(f"  Best Epoch: {overall_best_epoch}")
    print(f"  Best Accuracy: {overall_best_acc:.4f}")
    print(f"  Best Precision: {overall_best_metrics['val_precision']:.4f}")
    print(f"  Best Recall: {overall_best_metrics['val_recall']:.4f}")
    print(f"  Best F1-Score: {overall_best_metrics['val_f1']:.4f}")
    
    print(f"\nAverage Performance Across All Folds:")
    for metric in metrics_to_avg:
        print(f"  {metric}: {avg_metrics[f'avg_{metric}']:.4f} ± {avg_metrics[f'std_{metric}']:.4f}")

    print(f"\nPer-Fold Results:")
    for i, fold_result in enumerate(all_fold_results):
        print(f"  Fold {i}: Acc={fold_result['val_acc']:.4f}, F1={fold_result['val_f1']:.4f}, Best Epoch={fold_result['epoch']}")

    # Save detailed results
    results_summary = {
        'experiment_config': vars(args),
        'overall_best': overall_best_metrics,
        'average_metrics': avg_metrics,
        'all_fold_results': all_fold_results,
        'fold_histories': all_fold_histories,
        'timestamp': timestamp
    }

    results_file = os.path.join(output_dir, 'multifold_results_summary.json')
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

    # Log final summary to wandb
    wandb.log({
        "overall_best_fold": overall_best_fold,
        "overall_best_epoch": overall_best_epoch,
        "overall_best_accuracy": overall_best_acc,
        **avg_metrics
    })

    # Save best model artifact
    best_model_path = os.path.join(output_dir, f'best_model_fold_{overall_best_fold}.pth')
    artifact = wandb.Artifact("best_multifold_model", type="model")
    artifact.add_file(best_model_path)
    artifact.add_file(results_file)
    wandb.log_artifact(artifact)

    print(f"\nTraining completed successfully!")
    print(f"All results saved in: {output_dir}")
    
    wandb.finish()

if __name__ == '__main__':
    main()
