import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


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


def energy_division_loaders(all_events_test, all_records_test, all_labels_test, num_divisions:int, k:int, log_emin=None, log_emax=None):
    #Create the energy bins
    if log_emin and log_emax:
        #Emin = np.power(10, log_emin)
        #Emax = np.power(10, log_emax)
        energy_divisions = np.logspace(log_emin, log_emax, num_divisions)
        #energy_divisions = np.linspace(Emin, Emax, num_divisions)
        
    else:
        all_energies = [sub['energy'] for sub in all_records_test]
        energy_divisions = np.linspace(np.min(all_energies), np.max(all_energies), num_divisions)
    
    len_divisions = 0

    #Create dict with the index of the events for each energy division
    interval_dict = {}
    for interval_nr in range(len(energy_divisions)-1):
        interval_dict[f"interval{interval_nr}"] = []
        count = 0
        for nr,rc in enumerate(all_records_test):
            if rc.mc_info[0]['energy'] >= energy_divisions[interval_nr] and rc.mc_info[0]['energy'] < energy_divisions[interval_nr+1]:
                count+=1
                #print(energy_divisions[interval_nr+1])
                interval_dict[f"interval{interval_nr}"].append(nr)
        len_divisions+=count

    pbar = tqdm(total=len(all_labels_test))
    test_energy_loaders = []

    #Generate the test energy loaders for each energy division
    for int_nr, interval in enumerate(interval_dict):
        #print(interval_dict[interval])
        data_array = []
        for ev in interval_dict[interval]:

            features = get_features(det, all_events_test[ev])
            valid = np.all(np.isfinite(features), axis=1)
            features = features[valid]
            x = torch.Tensor(features)
            
            edge_index = knn_graph(x[:, [-1, -2, -3]], k=15, loop=False)
            #data = transf(torch_geometric.data.Data(x, edge_index, y=torch.tensor([label], dtype=torch.int64)).to(device))
            data = torch_geometric.data.Data(x, edge_index, y=torch.tensor(all_labels_test[ev], dtype=torch.int64)).to(device)
            data_array.append(data)
            pbar.update()
        loader = DataLoader(data_array)
        test_energy_loaders.append(loader)
        
    return test_energy_loaders, energy_divisions

def energy_evaluation(model, test_energy_loaders, label_map = {0:"Contained cascade", 1:'Throughgoing Track', 2:"Starts in detector", 3:"Rest of events"}):    
    nplt = int(np.ceil(np.sqrt(len(test_energy_loaders))))
    fig = plt.figure(figsize=(nplt * 4, nplt * 4))

    pbar = tqdm(total=len(test_energy_loaders))
    test_accuracies = []

    ## Checking for predictions
    with torch.no_grad():
        for i, loader in enumerate(test_energy_loaders):
            test_preds = []
            test_truths = []
            test_scores = []
            all_preds_test_len = 0
            for data in loader:  # Iterate in batches over the training/test dataset.
                out = model(data)
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