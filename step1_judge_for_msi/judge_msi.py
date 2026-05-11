import timm
from torch import nn
import judge_msi_utils
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# Only need input:
params = {
    'folder_path': './../Data_preprocess/',  # The path of the folder containing subfolders of .h5 files
    'model_path': '/home/jmayer/ACI_research/Code/mesoscale_structure/step1_judge_for_msi/finetuned_msi_model.pth', # '/home/jmayer/ACI_research/YuanyuanWu-NJU-Cloud-morphology-dataset/Model_judge/64epochs_accuracy0.91456.pth',
    'batch_size': 512,
    'device_num': '1',
    'save_path':'./'
}


def save_batch_to_npz(all_cats, all_certs, subdirectory):
    npz_filename = f'judged_night_{subdirectory}.npz'
    npz_path = os.path.join(params['save_path'],subdirectory, npz_filename)
    if not os.path.exists(os.path.dirname(npz_path)):
        os.makedirs(os.path.dirname(npz_path))
    np.savez(npz_path, cats=all_cats, certs=all_certs)


def judge(judge_loader, model, class_names, device, save_to=None):
    model.eval()
    all_cats = []
    all_certs = []

    with torch.no_grad():
        for i, (input) in enumerate(judge_loader, start=1):
            input = input.to(torch.float32).to(device)
            output = model(input)
            y_pred = torch.softmax(output, dim=1).cpu()
            cert = torch.max(y_pred, dim=1).values.cpu() / torch.sum(y_pred, dim=1).cpu() 
            cat = torch.argmax(y_pred, dim=1).cpu().numpy()  
            print(cat,cert)
            all_cats.extend(cat)
            all_certs.extend(cert)
    
    all_cats = np.array(all_cats)
    all_certs = np.array(all_certs)
    print(all_cats.shape)
    #save_batch_to_npz(all_cats, all_certs, subdirectory)
    return all_cats, all_certs


def judge_MSI(MCOP, MRGR, debugging=False):
    class_names = ['Solid stratus','Closed MCC', 'Open MCC','Disorganized MCC','Clustered Cu', 'Suppressed Cu'] #[0,1,2,3,4,5]
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda:' + params['device_num'] if torch.cuda.is_available() else 'cpu')

    test_dataset = judge_msi_utils.JudgeDataset(MCOP, MRGR, debugging=debugging)
    test_len = judge_msi_utils.JudgeDataset.__len__(test_dataset)
    scene_params = test_dataset.scene_params
    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    model = timm.create_model('resnet50d', pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 6)
    model = model.to(device)
    weights = torch.load(params['model_path'], map_location=device)
    model.load_state_dict(weights)
    model = model.to(device)
    
    cats, cert = judge(test_dataloader, model, class_names, device)

    del test_dataset
    del test_dataloader
    return cats, cert, scene_params

