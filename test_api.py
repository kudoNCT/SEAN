import sys
sys.path.append('./hair_segmentation_model/module')
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
from io import BytesIO
from hair_segmentation_model.module import SegmentationNet,check_model
from hair_segmentation_model.module.utils.dlib_test import make_erode_trimap
from preprocess_hair import prepare_data,tensor2im,blend_image,make_mask_bigger
import os
import uvicorn
import cv2
from models.networks.generator import SPADEGenerator
from config import cfg
import torch
import time




app = FastAPI()
def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload_image")
async def upload_image(style_number: int, file: UploadFile = File(...)):
    st_time = time.time()
    image = load_image_into_numpy_array(await file.read())
    image = image[:,:,:3]
    img_cv = image[:,:,::-1]
    cv2.imwrite('source.jpg',img_cv)
    #try:
    mask = segmentic_net.run(img_cv)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(mask,(256,256),cv2.INTER_AREA)

    bigger_mask = mask.copy()
    bigger_mask = make_mask_bigger(bigger_mask)
    #except Exception:
    #    return "Can't detect hair"
    #cv2.imwrite('mask.png',bigger_mask)
    input_semantics,real_image,obj_dic = prepare_data('source.jpg',style_number,bigger_mask,opt,use_gpu)
    fake_image = model_G(input_semantics, real_image, obj_dic)
    rs = tensor2im(fake_image, tile=False)[0]
    rs = rs[..., ::-1]
    #cv2.imwrite('result.jpg',rs)

    background = cv2.resize(img_cv, (256, 256), cv2.INTER_AREA)
    forceground = rs

    alpha = make_erode_trimap(bigger_mask,iteration=3)
    #print(f'alpha.shape: {alpha.shape}' )
    #print(f'np.unique alpha : {np.unique(alpha)}')
    #alpha = cv2.imread('alpha_ben.png')

    blend_image(forceground,background,alpha)
    path = "final_image_blend.jpg"
    ed_time = time.time()
    print(f'time excute: {ed_time-st_time}')
    return FileResponse(path,media_type="image/jpg")

if __name__ == '__main__':

    segmentic_net = SegmentationNet()
    opt = cfg
    use_gpu = True if len(opt.gpu_ids) > 0 else False
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
    ckpt = torch.load('checkpoints/CelebA-HQ_pretrained/54_net_G.pth', map_location=device)
    model_G = SPADEGenerator(opt)
    model_G.cuda() if use_gpu else model_G.cpu()
    model_G.load_state_dict(ckpt)
    model_G.eval()
    uvicorn.run(app,port=8000,host='192.168.129.24')

