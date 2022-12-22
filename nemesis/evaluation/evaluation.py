import torch
import torch_geometric
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from ..node_features.feature_generation import get_features
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph

if torch.cuda.is_available():
    print('CUDA')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print('CPU')



def evaluate_model(model, loader):

    preds = []
    truths = []
    scores = []

    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            preds.append(pred)
            truths.append(data.y)
            scores.append(out)
        preds = torch.cat(preds).cpu()
        truths = torch.cat(truths).cpu()
        scores = torch.cat(scores).cpu()

    return preds, truths, scores


class energy_gap_test():
    
    def __init__(self, det, model, all_events_test, all_records_test, all_labels_test, num_divisions:int, k:int):
        self.det = det
        self.model = model
        self.all_events_test = all_events_test
        self.all_records_test = all_records_test
        self.all_labels_test = all_labels_test
        self.k = k
        self.num_divisions = num_divisions
        
    def energy_division_loaders(self, log_emin=None, log_emax=None):
        #Create the energy bins
        if log_emin and log_emax:
            #Emin = np.power(10, log_emin)
            #Emax = np.power(10, log_emax)
            energy_divisions = np.logspace(log_emin, log_emax, self.num_divisions)
            #energy_divisions = np.linspace(Emin, Emax, num_divisions)
        else:
            all_energies = [sub['energy'] for sub in self.all_records_test]
            energy_divisions = np.linspace(np.min(all_energies), np.max(all_energies), self.num_divisions)
        
        len_divisions = 0
        #Create dict with the index of the events for each energy division
        interval_dict = {}
        
        for interval_nr in range(len(energy_divisions)-1):
            interval_dict[f"interval{interval_nr}"] = []
            count = 0
            for nr,rc in enumerate(self.all_records_test):
                if rc.mc_info[0]['energy'] >= energy_divisions[interval_nr] and rc.mc_info[0]['energy'] < energy_divisions[interval_nr+1]:
                    count+=1
                    #print(energy_divisions[interval_nr+1])
                    interval_dict[f"interval{interval_nr}"].append(nr)
            len_divisions+=count

        pbar = tqdm(total=len(self.all_labels_test))
        test_energy_loaders = []

        #Generate the test energy loaders for each energy division
        for int_nr, interval in enumerate(interval_dict):
            data_array = []
            for ev in interval_dict[interval]:
                features = get_features(self.det, self.all_events_test[ev])
                valid = np.all(np.isfinite(features), axis=1)
                features = features[valid]
                x = torch.Tensor(features)
                
                edge_index = knn_graph(x[:, [-1, -2, -3]], k=self.k, loop=False)
                #data = transf(torch_geometric.data.Data(x, edge_index, y=torch.tensor([label], dtype=torch.int64)).to(device))
                data = torch_geometric.data.Data(x, edge_index, y=torch.tensor(self.all_labels_test[ev], dtype=torch.int64)).to(device)
                data_array.append(data)
                pbar.update()
            loader = DataLoader(data_array)
            test_energy_loaders.append(loader)
            self.test_energy_loaders = test_energy_loaders
            
        return energy_divisions


    def energy_evaluation(self, label_map = {0:"Contained cascade", 1:'Throughgoing Track', 2:"Starts in detector", 3:"Rest of events"}):    
        
        nplt = int(np.ceil(np.sqrt(len(self.test_energy_loaders))))
        fig = plt.figure(figsize=(nplt * 4, nplt * 4))
        pbar = tqdm(total=len(self.test_energy_loaders))
        test_accuracies = []

        ## Checking for predictions
        with torch.no_grad():
            for i, loader in enumerate(self.test_energy_loaders):
                test_preds = []
                test_truths = []
                test_scores = []
                all_preds_test_len = 0
                
                for data in loader:  # Iterate in batches over the training/test dataset.
                    out = self.model(data)
                    #print(out.shape)
                    test_pred = out.argmax(dim=1)
                    test_preds.append(test_pred)
                    test_truths.append(data.y)
                    test_scores.append(out)
                    all_preds_test_len+=len(data.y)
                    
                test_preds = torch.cat(test_preds).cpu()
                test_truths = torch.cat(test_truths).cpu()
                test_scores = torch.cat(test_scores).cpu()
                misscls_idx = np.atleast_1d(np.argwhere(test_preds != test_truths).ravel())
                test_accuracy = 100 * (all_preds_test_len - len(misscls_idx)) / all_preds_test_len
                all_preds = []
                
                for j in range(len(label_map)):
                    true_sel = test_truths == j
                    predictions = np.histogram(test_preds[true_sel], bins=np.arange(0, len(label_map)+1, 1))[0]
                    if predictions.sum() == 0:
                        predictions = predictions
                    else:
                        predictions = predictions / predictions.sum()
                    all_preds.append(predictions)
                    
                all_preds = np.vstack(all_preds)
                test_accuracies.append(test_accuracy)
                ax = fig.add_subplot(nplt, nplt, i + 1)
                ax1 = sns.heatmap(all_preds, cmap=plt.cm.Blues, annot=True, ax=ax)# xticklabels=list(label_map.values()), yticklabels=list(label_map.values()), ax=ax)
                ax1.set_xlabel("Predicted Label")
                ax1.set_ylabel("True Label")
                #ax1.set_title()
                #fig1 = ax.get_figure()
                pbar.update()
                ax, ax1 = 0,0
                
                
        plt.show()
        print('The accuracies of the different energy divisions are', test_accuracies)
        
        return fig, test_accuracies, misscls_idx



def model_evaluation(model, test_loader, all_events, all_records, all_labels, label_map, test_indices, miss_labels=False):
    
    preds = []
    truths = []
    scores = []
    all_preds_len = 0
    
    ## Checking for predictions
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            out = model(data)#.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            preds.append(pred)
            truths.append(data.y)
            scores.append(out)
            all_preds_len+=len(data.y)

        preds = torch.cat(preds).cpu()
        truths = torch.cat(truths).cpu()
        scores = torch.cat(scores).cpu()
        
    ## Missed indices
    misscls_idx = np.atleast_1d(test_indices[np.argwhere(preds != truths).ravel()])
    corrcls_idx = np.atleast_1d(test_indices[np.argwhere(preds == truths).ravel()])
    misscls = [all_events[i] for i in misscls_idx]
    misscls_records = [all_records[i] for i in misscls_idx]
    misscls_pred = preds[np.argwhere(preds != truths).ravel()].numpy()
    miss_idx = {'idx': [], 'truth': [], 'pred': []}
    
    #Storage indexes of the missclassified events
    for idx, pred in zip(misscls_idx, misscls_pred):
        miss_idx['idx'].append(idx)
        miss_idx['truth'].append(label_map[all_labels[idx]])
        miss_idx['pred'].append(label_map[pred])
        
    miss_len = len(miss_idx['idx'])
    test_accuracy = 100 * (all_preds_len - miss_len) / all_preds_len
    all_preds = []
    
    #Check predictions
    for i in range(len(label_map)):
        true_sel = truths == i
        predictions = np.histogram(preds[true_sel], bins=np.arange(0, len(label_map)+1, 1))[0]
        if predictions.sum() == 0:
            predictions = predictions
        else:
            predictions = predictions / predictions.sum()
        all_preds.append(predictions)
        
    all_preds = np.vstack(all_preds)
    all_preds_len = len(all_preds)
    
    ax = sns.heatmap(all_preds, cmap=plt.cm.Blues, annot=True, xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig = ax.get_figure()
    #fig.savefig(f'./confmatrix/{model.__class__.__name__}_{len(all_events)}1.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, miss_idx, test_accuracy