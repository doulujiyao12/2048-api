import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import time
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset




NUM_EPOCHS = 50
BATCH_SIZE = 64
TIME_STEP = 4
INPUT_SIZE = 4
LR = 0.001
test_size = 4
trainfilertoread = 'DATA.csv'
testfiletoread = 'Test.csv'


class DealDataset(Dataset):

    def __init__(self, root,  transform=None):
        super().__init__()
        Data0 = pd.read_csv(root).values
        print('4\n')
        self.board = Data0[:, 0:-1]
        self.direc = Data0[:, -1]
        self.len = len(Data0)
        self.transform = transform
        self.idx = 0

    def __getitem__(self, index):

        board = self.board[index].reshape((4, 4))
        board1=board.T
        board2=np.vstack((board,board1))
        board = board2[:, :, np.newaxis]

        board = board/11.0
        direc = self.direc[index]
        if self.transform is not None:
            board = self.transform(board)
            board = board.type(torch.float)#更改过board = board.type(torch.float)
        return board, direc


    def __len__(self):

        return self.len




class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.my_rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=256,
            num_layers=4,
            batch_first=True
        )
        
        self.my_cnn=nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.out = nn.Linear(512, 4)
        self.fc1=nn.Linear(4,256)
        self.fc2=nn.Linear(256,256)

    def forward(self, x):

        r_out, (h_n, h_c) = self.my_rnn(x,None)
        c_in=torch.unsqueeze(x[:,0:4,:],1)
        c_out=self.my_cnn(c_in)
        c_out1=c_out.view(-1,4)
        c_fc=self.fc1(c_out1)
        r_fc=self.fc2(r_out[:,-1,:])
        final_fc=torch.cat((r_fc,c_fc),1)
        out = self.out(final_fc)
        return out



class TrainModel():

    def __init__(self):
        print('2\n')

        self.model = RNN()

    def trainModel(self):
        print('3\n')

        trainDataset = DealDataset(root=trainfilertoread, transform=transforms.Compose(transforms=[transforms.ToTensor()]))
        # testDataset = DealDataset(root=testfiletoread, transform=transforms.Compose(transforms=[transforms.ToTensor()]))
        train_loader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        # test_loader = DataLoader(dataset=testDataset, batch_size=test_size, shuffle=True, num_workers=0)

        print('1\n')

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=LR)
        for epoch in range(NUM_EPOCHS):


            for index, (board, direc) in enumerate(train_loader):
                board, direc = Variable(board), Variable(direc)


                if torch.cuda.is_available():
                    board, direc = board.cuda(), direc.cuda()
                    self.model.cuda()

                board = board.view(-1,8,4)
                out = self.model(board)
                loss = criterion(out, direc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # train_loss += loss.data[0]
                # pred = torch.max(out, 1)[1]
                # train_correct = (pred == direc).sum().item()
                # train_acc += train_correct

                if index % 50 == 0:

                    out = self.model(board)
                    pred = torch.max(out, 1)[1]


                    train_correct = (pred == direc).sum().item()



                    print('Epoch: ', epoch, '| train loss: %.4f' % loss,
                          '| test accuracy: %.4f' % (train_correct/(BATCH_SIZE * 1.0)))
            if epoch%3==0:              
                torch.save(self.model, 'rnn_model_' + str(epoch) + '.pkl')
        torch.save(self.model, 'rnn_model_final.pkl')



def main():
    # trans =
    print('6\n')

    trian1 = TrainModel()
    trian1.trainModel()
    # cnn = CnnNet()
    # print(cnn)

if __name__ == "__main__":
    # execute only if run as a script
    main()





