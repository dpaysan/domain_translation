import copy
import time

import numpy as np
import pandas as pd
import torch

from src.helper.data import DataHandler
from src.models.ae import GeneSetAE, VanillaAE
from src.models.custom_networks import GeneSetAE_v2
from src.utils.torch.data import init_seq_dataset
from src.utils.torch.general import get_device

gene_set_dataset = init_seq_dataset("../../data/cd4/cd4_rna_seq_pbmc_10k/gene_kegg_pathway_data_and_labels.csv")
gene_set_adjacencies = pd.read_csv("../../data/cd4/cd4_rna_seq_pbmc_10k/gene_kegg_pathway_adj_filtered_nCD4.csv",
                                   index_col=0)
geneset_adj_matrix = torch.from_numpy(np.array(gene_set_adjacencies))

geneset_ae = GeneSetAE(input_dim=5785, hidden_dims=[512, 512, 512, 512, 512, 512], latent_dim=256,
                       geneset_adjacencies=geneset_adj_matrix)

# geneset_ae = VanillaAE(input_dim=5785, latent_dim=256, hidden_dims=[2048, 1024, 512, 256, 256])

#geneset_ae = GeneSetAE_v2(adjacency_matrix=geneset_adj_matrix, hidden_dims=[64, 32, 16, 8], latent_dim=256)

data_key = 'seq_data'
label_key = 'label'

dh = DataHandler(
    dataset=gene_set_dataset,
    batch_size=32,
    num_workers=0,
    random_state=1234,
    transformation_dict=None,
    drop_last_batch=False,
)
dh.stratified_train_val_test_split(splits=[0.6, 0.2, 0.2])
dh.get_data_loader_dict()
data_loader_dict = dh.data_loader_dict

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(geneset_ae.parameters(), lr=0.0001)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.infty
    device = get_device()

    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data[data_key].to(device)
                labels = data[label_key].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)['recons']
                    loss = criterion(outputs, inputs)

                    # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.6f}'.format(phase, epoch_loss))

            # deep copy the model if it has the best val accurary
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


trained_geneset_ae = train_model(geneset_ae, dataloaders=data_loader_dict,
                                 criterion=criterion, optimizer=optimizer,
                                 num_epochs=3000)
print(trained_geneset_ae)


def test_model(model, dataloaders, criterion):
    device = get_device()
    model.to(device)
    model.eval()
    runninq_loss = 0
    for data in dataloaders['test']:
        inputs = data[data_key].to(device)
        outputs = model(inputs)['recons']
        runninq_loss += criterion(outputs, inputs).item() * inputs.size(0)
    test_loss = runninq_loss / len(dataloaders["test"].dataset)
    print("Test loss: ", test_loss)
    return test_loss


test_model(trained_geneset_ae, data_loader_dict, criterion)
