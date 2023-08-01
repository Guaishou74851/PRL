import os, glob, random, torch, cv2
import numpy as np
from argparse import ArgumentParser
from model import PRL
from utils import *
from skimage.metrics import structural_similarity as ssim

parser = ArgumentParser(description='PRL')
parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--phase_num', type=int, default=5)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--testset_name', type=str, default='Set11')
parser.add_argument('--result_dir', type=str, default='test_out')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--num_feature', type=int, default=8)
parser.add_argument('--ID_num_feature', type=int, default=8)
parser.add_argument('--cs_ratio', type=float, default=0.1)

args = parser.parse_args()
epoch = args.epoch
N_p = args.phase_num
B = args.block_size
nf = args.num_feature
ID_nf = args.ID_num_feature
cs_ratio = args.cs_ratio

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed seed for reproduction
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

N = B * B

# create and initialize CASNet
model = PRL(N_p, B, torch.zeros(int(np.ceil(cs_ratio*N)), N), nf, ID_nf)
model = torch.nn.DataParallel(model).to(device)
model_dir = './%s/ratio_%.2f_layer_%d_block_%d_f_%d_IDnf_%d' % (args.model_dir, cs_ratio, N_p, B, nf, ID_nf)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch)))

# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + '/*')
test_image_num = len(test_image_paths)

def test():
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(test_image_num):
            image_path = test_image_paths[i]
            test_image = cv2.imread(image_path, 1)  # read test data from image file
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

avg_psnr, avg_ssim = test()
print('CS ratio is %.2f, avg PSNR is %.2f, avg SSIM is %.4f.' % (cs_ratio, avg_psnr, avg_ssim))
