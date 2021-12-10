import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from options import Options 
from train64 import Train
from dataloader import load_data_train,load_test_data

if __name__ == "__main__":
    opt = Options().parse()
    train_data = load_data_train(opt) 
    test_data, label = load_test_data(opt)
    model = Train(opt, train_data,test_data=test_data, test_label=label)
    model.train()

    #for idx, (data, _) in enumerate(test_data):
    #    print(f"data.shape=P{data.shape}")
    print(f"test....")
    model.Test()
