import cls_data_generator
import parameters
import seldnet_model
import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(data_generator, optimizer, model, criterion,  device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for data, target in data_generator.generate():
        # load one batch of data
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        nb_train_batches += 1

    train_loss /= nb_train_batches
    return train_loss

params = parameters.get_params('9')

params['batch_size'] = 32
use_cuda = torch.cuda.is_available()

data_gen_train = cls_data_generator.DataGenerator(
    params=params, split=[1]
)
data_gen_val = cls_data_generator.DataGenerator(
    params=params, split=[2], shuffle=False, per_file=True
)

device = torch.device("cuda" if use_cuda else "cpu")

data_in, data_out = data_gen_train.get_data_sizes()
model = seldnet_model.CRNN(data_in, data_out, params).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

# for data, target in data_gen_val.generate():
#     data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
#     # import pdb; pdb.set_trace()
#     output = model(data)
#     # print(f'output.shape = {output.shape}')
    

for i in range(10):
    loss = train_epoch(data_gen_train, optimizer, model, criterion,  device)
    #print loss
    print(round(loss,2))

