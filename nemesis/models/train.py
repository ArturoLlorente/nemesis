import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nemesis.plotting.plotting import plot_confusion
from tqdm.auto import tqdm

if torch.cuda.is_available():
    print('CUDA')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print('CPU')
    
    

class model_training():   
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def get_loss(self, data):
        out = self.model(data)#, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = self.criterion(out, data.y)  # Compute the loss.
        pred = out.argmax(dim=1)
        return loss, pred
    
    def train(self):
        self.model.train()
        total_loss = 0
        correct = 0
        for data in self.train_loader:  # Iterate in batches over the training dataset.
            data.to(self.device)
            loss, pred = self.get_loss(data)
            total_loss += loss.item() * len(torch.unique(data.batch))
            correct += int((pred == (data.y)).sum())
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

        return total_loss / len(self.train_loader.dataset), correct / len(self.train_loader.dataset)
    
    def validation(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        for data in self.val_loader:
            data.to(self.device)
            loss, pred = self.get_loss(data)
            total_loss += loss.item() * len(torch.unique(data.batch))
            correct += int((pred == (data.y)).sum())
        return total_loss / len(self.val_loader.dataset), correct / len(self.val_loader.dataset)
    
    

def train_model(model, train_dataset, val_dataset, label_map, k=8, lr=0.001, batch_size=200, epochs=200, patience=200, print_step=1, use_writer=False):

    pbar = tqdm(total=epochs)

    #model = Dynamic_class(out_channels=num_classes, k=k)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    #swa_model = torch.optim.swa_utils.AveragedModel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)
    if use_writer:
        writer = SummaryWriter(f"/app/tensorboard/July24/model_5 k_value {k} lr: {lr}")# LR {lr}")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model_train = model_training(model, train_loader, val_loader, optimizer, criterion, device)
    all_trains_acc, all_vals_acc, all_trains_loss, all_vals_loss = [], [], [], []

    for epoch in range(epochs):
        
        train_loss, train_acc = model_train.train()
        val_loss, val_acc = model_train.validation()

        if epoch % print_step == 0:
            print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, patience: {patience_count}")
        if use_writer:
            writer.add_scalar("Training Accuracy", train_acc, epoch)
            writer.add_scalar("Val Accuracy", val_acc, epoch)
            writer.add_scalar("Training Loss", train_loss, epoch)
            writer.add_scalar("Val Loss", val_loss, epoch)
            preds, truths, scores = evaluate_model(model, val_loader)
            fig = plot_confusion(preds, truths, label_map)
            writer.add_figure("confusion_truth", fig, epoch)
        all_trains_acc.append(train_acc)
        all_vals_acc.append(val_acc)
        all_trains_loss.append(train_loss)
        all_vals_loss.append(val_loss)
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_count = 0
            #torch.save(model.state_dict(), f"/app/models/July24/model_5 k_value {k} lr: {lr}")
        patience_count += 1
        if patience_count == patience:
            break
        #if train_acc > 0.98:
        #    break
        pbar.update()

    if use_writer:
        writer.add_hparams({"knn value":k},{"Train Accuracy":train_acc,"Train Loss":train_loss,"Val Accuracy":val_acc,"Val Loss":val_loss},)
        writer.close()  
        
    print('The best epoch was: ', best_epoch)
    print("Number of training epochs: ", epoch)

    return model, all_trains_acc, all_vals_acc