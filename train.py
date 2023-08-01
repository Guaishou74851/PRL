import torch, os, glob, cv2, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from model import PRL
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser(description='PRL')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=600)
parser.add_argument('--phase_num', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--testset_name', type=str, default='Set11')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--num_feature', type=int, default=8)
parser.add_argument('--ID_num_feature', type=int, default=8)
parser.add_argument('--cs_ratio', type=float, default=0.1)

args = parser.parse_args()

start_epoch, end_epoch = args.start_epoch, args.end_epoch
learning_rate = args.learning_rate
N_p = args.phase_num
B = args.block_size
nf = args.num_feature
ID_nf = args.ID_num_feature
cs_ratio = args.cs_ratio

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# fixed seed for reproduction
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

patch_size = 128  # training patch size
batch_size = 16
iter_num = 800
N = B * B
q = int(np.ceil(N * cs_ratio))

# training set info
print('reading files...')
start_time = time()
training_image_paths = glob.glob(os.path.join(args.data_dir, 'pristine_images') + '/*')
training_image_num = len(training_image_paths)
print('training_image_num', training_image_num, 'read time', time() - start_time)

model = PRL(N_p, B, torch.nn.init.xavier_normal_(torch.Tensor(q, N)), nf, ID_nf)
model = torch.nn.DataParallel(model).to(device)

class MyDataset(Dataset):
    def __init__(self):
        self.len = iter_num * batch_size
        self.real_len_1 = training_image_num - 1

    def __getitem__(self, index):
        while True:
            index = random.randint(0, self.real_len_1)
            path = training_image_paths[index]

            # read from disk
            training_image_ycrcb = cv2.imread(path, 1)
            training_image_ycrcb = cv2.cvtColor(training_image_ycrcb, cv2.COLOR_BGR2YCrCb)
            training_image_y = training_image_ycrcb[:, :, 0]
            training_image_y_tensor = torch.Tensor(training_image_y) / 255.0

            h, w = training_image_y.shape
            max_h, max_w = h - patch_size, w - patch_size
            if max_h < 0 or max_w < 0:
                continue
                
            start_h = random.randint(0, max_h)
            start_w = random.randint(0, max_w)
            
            return training_image_y_tensor[start_h:start_h+patch_size, start_w:start_w+patch_size]

    def __len__(self):
        return self.len

my_loader = DataLoader(dataset=MyDataset(), batch_size=batch_size, num_workers=8, pin_memory=True)

optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': learning_rate}], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 500, 550], gamma=0.1, last_epoch=start_epoch-1)

model_dir = './%s/ratio_%.2f_layer_%d_block_%d_f_%d_IDnf_%d' % (args.model_dir, cs_ratio, N_p, B, nf, ID_nf)
log_path = './%s/ratio_%.2f_layer_%d_block_%d_f_%d_IDnf_%d.txt' % (args.log_dir, cs_ratio, N_p, B, nf, ID_nf)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + '/*')
test_image_num = len(test_image_paths)

def test():
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(test_image_num):
            test_image = cv2.imread(test_image_paths[i], 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0])
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            
            x_input = torch.from_numpy(img_pad)
            x_input = x_input.type(torch.FloatTensor).to(device)

            x_output = model(x_input)
            x_output = x_output.cpu().data.numpy().squeeze()
            x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0

            PSNR = psnr(x_output, img)
            SSIM = ssim(x_output, img, data_range=255)

            # print('[%d/%d] %s, PSNR: %.2f, SSIM: %.4f' % (i, test_image_num, image_path, PSNR, SSIM))

            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)

    return float(np.mean(PSNR_list)), float(np.mean(SSIM_list))

if start_epoch > 0:
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch)))

print('start training...')
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    start_time = time()
    loss_avg, iter_num = 0.0, 0
    for data in tqdm(my_loader):
        x = data.unsqueeze(1).to(device)
        x = H(x, random.randint(0, 7))
        
        loss = (model(x) - x).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iter_num += 1
        loss_avg += loss.item()
        
    scheduler.step()

    loss_avg /= iter_num
    
    log_data = '[%d/%d] Average loss: %f, time cost: %.2fs.' % (epoch_i, end_epoch, loss_avg, time() - start_time)
    print(log_data)
    with open(log_path, 'a') as log_file:
        log_file.write(log_data + '\n')

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), './%s/net_params_%d.pkl' % (model_dir, epoch_i))  # save only the parameters

    if epoch_i == 1 or epoch_i % 10 == 0:
        cur_psnr, cur_ssim = test()
        log_data = 'CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f.' % (cs_ratio, cur_psnr, cur_ssim)
        print(log_data)
        with open(log_path, 'a') as log_file:
            log_file.write(log_data + '\n')
