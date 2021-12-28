from options.test_options import TestOptions
import numpy as np
from glob import glob
import os
import torch
import skimage.io
from models.networks.generator import SPADEGenerator
import cv2
from data.base_dataset import get_params,get_transform
from PIL import Image
from matplotlib import pyplot as plt
import random
import time
from config import cfg

def load_input_feature(file_name,use_gpu):
    ############### load average features

    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(file_name.replace('png','jpg'))
    input_style_dic = {}
    label_count = []

    style_img_mask_dic = {}

    for i in range(19):
        input_style_dic[str(i)] = {}

        input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
        input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in average_category_folder_list]

        for style_code_path in average_category_list:
            if style_code_path in input_category_list:
                if use_gpu:
                    input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
                else:
                    input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy')))
            else:
                if use_gpu:
                    input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
                else:
                    input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy')))

    obj_dic = input_style_dic
    return obj_dic

def load_average_feature(use_gpu):
    average_style_code_folder = 'styles_test/mean_style_code/mean/'
    input_style_dic = {}
    for i in range(19):
        input_style_dic[str(i)] = {}

        average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
        average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
                                 average_category_folder_list]

        for style_code_path in average_category_list:
            if use_gpu:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
            else:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy')))

    obj_dic = input_style_dic
    return obj_dic


def generate_target_mask(target_path, reference_path):
    rs = []
    target_mask = cv2.imread(target_path)
    reference_mask = cv2.imread(reference_path)

    for mask in [target_mask.copy(), reference_mask.copy()]:
        mask = mask.reshape((1, -1))
        for i, vl in enumerate(mask[0]):
            if vl == 13:
                mask[0][i] = 255
            else:
                mask[0][i] = 0

        mask = mask.reshape((512, 512, 3))
        rs.append(mask)
    target_bmask, reference_bmask = rs[0], rs[1]
    target2 = target_mask.copy()
    target2[np.where(target_bmask == 255)] = 0
    target2[np.where(reference_bmask == 255)] = reference_mask[np.where(reference_bmask == 255)]
    return target2


def preprocess_input(data,use_gpu,opt):
    # move to GPU and change data types
    data['label'] = data['label'].long()
    if use_gpu:
        data['label'] = data['label'].cuda(non_blocking=True)
        data['instance'] = data['instance'].cuda(non_blocking=True)
        data['image'] = data['image'].cuda(non_blocking=True)

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.label_nc + 1 if opt.contain_dontcare_label \
        else opt.label_nc

    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_() if use_gpu else torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if not opt.no_instance:
        inst_map = data['instance']
        instance_edge_map = self.get_edges(inst_map)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    return input_semantics, data['image']


def preprocess_input_2(data):
    data['label'] = data['label'].long()
    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = 19
    input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return input_semantics, data['image']

def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)



def prepare_data(path_src,style_num,label_img,opt,use_gpu):
    fileStyleHair = f'{style_num}.jpg'
    image_path = path_src
    instance_tensor = torch.Tensor([0])

    label = Image.fromarray(label_img)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc  # 'unknown' is opt.label_nc
    label_tensor.unsqueeze_(0)
    obj_dic = load_average_feature(use_gpu)

    image = Image.open(image_path)
    image = image.convert('RGB')
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(image)
    image_tensor.unsqueeze_(0)
    input_dict = {'label': label_tensor,
                  'instance': instance_tensor,
                  'image': image_tensor,
                  'path': image_path,
                  'obj_dic': obj_dic
                  }
    input_semantics, real_image = preprocess_input(input_dict, use_gpu, opt)
    obj_dic = input_dict['obj_dic']
    input_style_hair_code_folder = "styles_test/style_codes/" + fileStyleHair
    if use_gpu:
        style_hair = torch.from_numpy(
            np.load(os.path.join(input_style_hair_code_folder, str(1), 'ACE' + '.npy'))).cuda()
    else:
        style_hair = torch.from_numpy(np.load(os.path.join(input_style_hair_code_folder, str(1), 'ACE' + '.npy')))
    obj_dic[str(1)]["ACE"] = style_hair

    return input_semantics,real_image,obj_dic

def blend_image(foreground,background,alpha):

    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    cv2.imwrite("final_image_blend.jpg", outImage)
def make_mask_bigger(mask):
    label = mask.copy()
    label[np.where(label == 1)] = 255
    # cv2.imshow('label',label)
    label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    a, thresh = cv2.threshold(label_gray, 60, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(1, 1, 1), thickness=5,
                     lineType=cv2.LINE_AA)
    return mask


def main():
    opt = TestOptions().parse()
    opt.status = 'Test_mode'
    time_run_st = time.time()
    use_gpu = False
    if len(opt.gpu_ids) > 0:
        use_gpu = True
        device = torch.device(f'cuda:{opt.gpu_ids[0]}')
    else:
        device = 'cpu'
    fileName = "20075" + ".png"
    fileStyleHair = random.choice(os.listdir("styles_test/style_codes"))
    print(f'fileStyleHair: {fileStyleHair}')
    #fileStyleHair = "28066.jpg"
    #print(f'fileStyleHair: {fileStyleHair}')
    #fileStyleHair = "29986.jpg"
    #mat_img_path = os.path.join(opt.label_dir, os.path.basename(fileName))
    mat_img_path = "new_label_29258.png"
    GT_img_path = os.path.join(opt.image_dir, os.path.basename(fileName)[:-4] + '.jpg')
    image_path = GT_img_path
    instance_tensor = torch.Tensor([0])

    target_path = os.path.join(opt.label_dir, os.path.basename(fileName))
    reference_path = os.path.join(opt.label_dir, os.path.basename(fileStyleHair.replace('.jpg', '.png')))

    # print(f'target path : {target_path} \n refer path: {reference_path}')
    mat_img = cv2.imread(mat_img_path)

    label_img = mat_img[:, :, 0]
    label = Image.fromarray(label_img)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc  # 'unknown' is opt.label_nc
    label_tensor.unsqueeze_(0)
    print(f'label_tensor.shape: {label_tensor.shape}')

    print(f'opt: {opt}')
    # obj_dic = load_average_feature()
    obj_dic = load_average_feature(use_gpu)

    image = Image.open(image_path)
    image = image.convert('RGB')
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(image)
    image_tensor.unsqueeze_(0)

    input_dict = {'label': label_tensor,
                  'instance': instance_tensor,
                  'image': image_tensor,
                  'path': image_path,
                  'obj_dic': obj_dic
                  }

    ckpt = torch.load('checkpoints/CelebA-HQ_pretrained/60_net_G.pth', map_location=device)
    model_G = SPADEGenerator(opt)
    if use_gpu:
        model_G.cuda()
    model_G.load_state_dict(ckpt)
    model_G.eval()
    input_semantics, real_image = preprocess_input(input_dict, use_gpu, opt)
    obj_dic = input_dict['obj_dic']
    input_style_hair_code_folder = "styles_test/style_codes/" + fileStyleHair
    if use_gpu:
        style_hair = torch.from_numpy(
            np.load(os.path.join(input_style_hair_code_folder, str(1), 'ACE' + '.npy'))).cuda()
    else:
        style_hair = torch.from_numpy(np.load(os.path.join(input_style_hair_code_folder, str(1), 'ACE' + '.npy')))

    obj_dic[str(1)]["ACE"] = style_hair
    st = time.time()
    fake_image = model_G(input_semantics, real_image, obj_dic)
    ed = time.time()
    fps = 1 / (ed - st) if ed > st else 0
    rs = tensor2im(fake_image, tile=False)[0]
    rs = rs[..., ::-1]
    print(f'FPS: {fps}')
    print(f'Time excute: {ed - st}')
    # cv2.imshow('rs',rs)
    cv2.imwrite('result.jpg', rs)
    time_run_ed = time.time()
    print(f'Time running etry :  {time_run_ed - time_run_st}')





if __name__ == "__main__":
    main()

