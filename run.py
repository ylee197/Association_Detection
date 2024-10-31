import numpy as np
from tqdm import tqdm
import argparse
import logging
import time
import pandas as pd

import torch
from torch_geometric.data import NeighborSampler

from data_loading import get_dataset
from data_creating import creating_dataset1, creating_dataset2, creating_dataset3
from data_utils import set_uniform_train_val_test_split
from seeds import val_seeds
from filling_strategies import filling
from evaluation import test
from train import train
from reconstruction import spatial_reconstruction
from GCN import GNN
from result import result

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics

parser = argparse.ArgumentParser('FP_GCN')
parser.add_argument('--data_path',type=str,help = 'Data path',default = '.')
parser.add_argument('--patience', type = int, help='Patience for early stopping', default = 100)
parser.add_argument('--lr', type=float, help='Learning Rate', default=0.05)
parser.add_argument('--epochs', type=int, help='Max number of epochs', default = 10000)
parser.add_argument('--n_runs', type=int, help='Max number of runs', default=10)
parser.add_argument('--hidden_dim', type=int, help='Hidden dimension of model', default=64)
parser.add_argument('--num_layers', type=int, help='Number of GNN layers', default=2)
parser.add_argument('--num_iterations', type=int, help='Number of diffsuion iterations for feature reconstruction', default = 10)
parser.add_argument('--lp_alpha', type=float, help='Alpha parameter of label propagation', default=0.9)
parser.add_argument('--dropout', type=float, help='Feature dropout', default=0.5)
parser.add_argument('--batch_size', type=int, help='Batch size for models trained with neighborhood sampling', default=1024)
parser.add_argument('--log', type=str, help='Log Level', default="INFO", choices=['DEBUG', 'INFO', 'WARNING'])
parser.add_argument('--gpu_idx', type=int, help='Indexes of gpu to run program on', default=0)

def run(args):
    device=torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    ## Creating dataset
    n_user, n_domain = creating_dataset1(data_path=args.data_path)
  
    ## Data loading
    dataset = get_dataset(n_user, n_domain, data_path=args.data_path, label = 'user')
    
    print(dataset)
    n_nodes, n_features = dataset.data.x.shape
    test_accs, best_val_accs, relative_reconstruction_errors, train_times, f1s, aucs = [], [], [], [], [], []
    
    df_pred = pd.DataFrame()
    
    f = open(f'{args.data_path}/data/confusion_matrix.txt', 'w')
    for seed in tqdm(val_seeds[:args.n_runs]):
        num_classes = dataset.num_classes
        data = set_uniform_train_val_test_split(
            seed=seed,
            data=dataset.data,
        ).to(device)

        train_start = time.time()
        feature_mask = dataset.data.missing

        x = data.x.clone()
        x[~feature_mask] = float('nan')
      
        logger.info("Starting feature filling")
        start = time.time()
        filled_features = filling(data.edge_index, x, feature_mask, 5)
        logger.info(f"Feature filling completed. It took: {time.time() - start:.2f}s")
        relative_reconstruction_errors.append(spatial_reconstruction(data.x, filled_features, feature_mask))

        model = GNN(num_features=data.num_features, num_classes=num_classes, num_layers=args.num_layers, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
        criterion = torch.nn.NLLLoss()

        test_acc = 0
        val_accs = []

        for epoch in range(0, args.epochs):
            start = time.time()

            x = torch.where(feature_mask, data.x, filled_features)

            train(model, x, data, optimizer, criterion, train_loader=None, device=device)

            (train_acc, val_acc, tmp_test_acc), out, _=test(model, x=x, data=data, device=device)

            if epoch ==0 or val_acc > max(val_accs):
                test_acc = tmp_test_acc
                y_soft = out.softmax(dim=-1)


            val_accs.append(val_acc)
            if epoch > args.patience and max(val_accs[-args.patience:]) <= max(val_accs[:-args.patience]):
                print(f'Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}. It took {time.time() - start:.2f}s')
                break

            logger.info(f'Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}. It took {time.time() - start:.2f}s')
         
        out = model(data.x, data.edge_index)
        (_, val_acc, test_acc), _, pred = test(model, x=x, data=data, logits=y_soft, evaluator=False)
    
        fpr, tpr, thresholds = metrics.roc_curve(data.y[data.test_mask].cpu().tolist(),pred.cpu().tolist(), pos_label=2)
        auc = metrics.auc(fpr, tpr)
        print('AUC : '+str(auc))
        f.write('\noriginal user_id :{}\n'.format(set(data.y[data.test_mask].cpu().tolist())))
        f.write('\npredicted user_id :{}\n'.format(set(data.y[data.test_mask].cpu().tolist())))
        f.write('\n user_id test_acc:{}\n'.format(test_acc))
        f.write('Confusion Matrix for user_id\n{}\n\n'.format(confusion_matrix(data.y[data.test_mask].cpu().tolist(),pred.cpu().tolist())))
        f.write('F1 for user_id\n{}\n\n'.format(f1_score(data.y[data.test_mask].cpu().tolist(),pred.cpu().tolist(),average='micro')))
        f.write('auc for user_id\n{}\n\n'.format(auc))
        best_val_accs.append(val_acc)
        test_accs.append(test_acc)
        f1s.append(f1_score(data.y[data.test_mask].cpu().tolist(),pred.cpu().tolist(),average='micro'))
        aucs.append(auc)
        train_times.append(time.time() - train_start)

        df = pd.DataFrame()
        df_feature = pd.DataFrame(filled_features.cpu().numpy())
        df_feature.to_csv(f'{args.data_path}/data/feature_output.csv', index = False)
        df['node_index'] = data.test_idx.cpu().tolist()
        df = df.sort_values(by = ['node_index'])
        df['predict'] = pred.cpu().tolist()
        df_pred = df_pred.append(df)
    df_pred.to_csv(f'{args.data_path}/data/predict_output.csv', index = False)

    relative_reconstruction_error_mean, relative_reconstruction_error_std = np.mean(relative_reconstruction_errors), np.std(relative_reconstruction_errors)

    results = {"relative_reconstruction_error_mean": relative_reconstruction_error_mean, "relative_reconstruction_error_std": relative_reconstruction_error_std}
    print(f'Reconstruction error: {relative_reconstruction_error_mean} +- {relative_reconstruction_error_std}')

   
    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    val_acc_mean, val_acc_std = np.mean(best_val_accs), np.std(best_val_accs)
    train_time_mean, train_time_std = np.mean(train_times), np.std(train_times)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s)
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)
    print(f'{test_acc_mean} +- {test_acc_std}')

    results = {**results, **{"test_acc_mean": test_acc_mean, "test_acc_std": test_acc_std, "val_acc_mean": val_acc_mean, "val_acc_std": val_acc_std, "train_time_mean": train_time_mean, "train_time_std": train_time_std, "f1_mean": f1_mean, "f1_std": f1_std, "auc_mean": auc_mean, "auc_std": auc_std}}
    print("test_acc_mean : " + str(test_acc_mean) + " test_acc_std : "+ str(test_acc_std) +" val_acc_mean : " + str(val_acc_mean) + " val_acc_std : "+ str(val_acc_std)+ " train_time_mean : "+ str(train_time_mean)+ " train_time_std : "+ str(train_time_std) + " f1_mean : "+ str(f1_mean) + " f1_std : "+ str(f1_std) + " auc_mean : "+ str(auc_mean) +" auc_std : "+ str(auc_std))
    
    f.write('Results:{}\n'.format(results))
    
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=getattr(logging, args.log.upper(), None))
    logging.basicConfig(filename='log_file.log', filemode='w', level=logging.INFO, force=True)
    run(args)
    
  
