import argparse
import cv2
import numpy as np
import glob 
import time
import matmul_inference
import inpaint_inference
import os


INPUT_SIZE = 512  # input image size for Generator
ATTENTION_SIZE = 32 # size of contextual attention
MULTIPLE = 6
NPTYPE = np.float32

def sort(str_lst):
    return [s for s in sorted(str_lst)]

# reconstruct residual from patches
def reconstruct_image_from_patches(residual, MULTIPLE):
    residual = np.reshape(residual, [ATTENTION_SIZE, ATTENTION_SIZE, MULTIPLE, MULTIPLE, 3])
    residual = np.transpose(residual, [0,2,1,3,4])
    return np.reshape(residual, [ATTENTION_SIZE*MULTIPLE, ATTENTION_SIZE*MULTIPLE, 3])

# extract image patches
def extract_image_patches(img, MULTIPLE):
    h, w, c = img.shape
    img = np.reshape(img, [h//MULTIPLE, MULTIPLE, w//MULTIPLE, MULTIPLE, c])
    img = np.transpose(img, [0,2,1,3,4])
    return img

# resize image by averaging neighbors
def resize_ave(img, MULTIPLE):
    img = img.astype(NPTYPE)
    img_patches = extract_image_patches(img, MULTIPLE)
    img = np.mean(img_patches, axis=(2,3))
    return img


def read_imgs_masks(images,masks):
    paths_img = glob.glob(images+'/*.*[gG]')
    paths_mask = glob.glob(masks+'/*.*[gG]')
    paths_img = sort(paths_img)
    paths_mask = sort(paths_mask)
    print('#imgs: ' + str(len(paths_img)))
    print('#imgs: ' + str(len(paths_mask)))
    print(paths_img)
    print(paths_mask)
    return paths_img, paths_mask

class Inpaint_App(object):
    def __init__(self,images,masks,inpaint_model,matmul_model,output_dir):
        self.inpaint_model = inpaint_inference.InpaintInference(inpaint_model)
        self.matmul_model = matmul_inference.MatMulInference(matmul_model)
        self.images = images
        self.masks = masks
        self.output_dir = output_dir
    
    def execute(self):
        paths_img, paths_mask = read_imgs_masks(self.images,self.masks)
        for i in range(len(paths_img)):
            raw_img = cv2.imread(paths_img[i]) 
            raw_mask = cv2.imread(paths_mask[i]) 
            inpainted = self.inpaint(raw_img, raw_mask)

            filename = self.output_dir + '/' + os.path.basename(paths_img[i])
            cv2.imwrite(filename + '_inpainted.jpg', inpainted)

        print('inpaint execution complete')

    #utilize matmul offline model to accelerate matmul
    def matmul_om(self,attention,residual):
        attention_reshape = attention.reshape(1024,1024)
        residual_reshape = residual.reshape(1024,3072*9)
        result = []
        for i in range(9):
            resi = residual_reshape[:,i*3072:(i+1)*3072]
            tmp = self.matmul_model.Inference(attention_reshape,resi)
            result.append(tmp.reshape(1024,3072))
        return np.hstack(result).reshape(ATTENTION_SIZE,ATTENTION_SIZE,3072*9)

    # residual aggregation module
    def residual_aggregate(self,residual, attention):
        residual = extract_image_patches(residual, MULTIPLE * INPUT_SIZE//ATTENTION_SIZE)
        residual = np.reshape(residual, [1, residual.shape[0] * residual.shape[1], -1])
        residual = self.matmul_om(attention, residual)
        residual = reconstruct_image_from_patches(residual, MULTIPLE * INPUT_SIZE//ATTENTION_SIZE)
        return residual
    
    # pre-processing module
    def preprocess(self,raw_img, raw_mask):

        raw_mask = raw_mask.astype(NPTYPE) / 255.
        raw_img = raw_img.astype(NPTYPE)

        # resize raw image & mask to desinated size
        large_img = cv2.resize(raw_img,  (MULTIPLE * INPUT_SIZE, MULTIPLE * INPUT_SIZE), interpolation = cv2. INTER_LINEAR)
        large_mask = cv2.resize(raw_mask, (MULTIPLE * INPUT_SIZE, MULTIPLE * INPUT_SIZE), interpolation = cv2.INTER_NEAREST)
        
        # down-sample large image & mask to 512x512
        small_img = resize_ave(large_img, MULTIPLE)
        small_mask = cv2.resize(raw_mask, (INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_NEAREST)
        
        # set hole region to 1. and backgroun to 0.
        small_mask = 1. - small_mask
        return large_img, large_mask, small_img, small_mask

    # post-processing module
    def post_process(self,raw_img, large_img, large_mask, inpainted_512, img_512, mask_512, attention):

        # compute the raw residual map
        h, w, c = raw_img.shape
        low_base = cv2.resize(inpainted_512.astype(NPTYPE), (INPUT_SIZE * MULTIPLE, INPUT_SIZE * MULTIPLE), interpolation = cv2.INTER_LINEAR) 
        low_large = cv2.resize(img_512.astype(NPTYPE), (INPUT_SIZE * MULTIPLE, INPUT_SIZE * MULTIPLE), interpolation = cv2.INTER_LINEAR)
        residual = (large_img - low_large) * large_mask

        # reconstruct residual map using residual aggregation module
        residual = self.residual_aggregate(residual, attention)

        # compute large inpainted result
        res_large = low_base + residual
        res_large = np.clip(res_large, 0., 255.)

        # resize large inpainted result to raw size
        res_raw = cv2.resize(res_large, (w, h), interpolation = cv2.INTER_LINEAR)
        
        # paste the hole region to the original raw image
        mask = cv2.resize(mask_512.astype(NPTYPE), (w, h), interpolation = cv2.INTER_LINEAR)
        mask = np.expand_dims(mask, axis=2)
        res_raw = res_raw * mask + raw_img * (1. - mask)
        return res_raw.astype(np.uint8)

    def inpaint(self, raw_img, raw_mask):
        print('==========')
        s = time.time()
        # pre-processing
        img_large, mask_large, img_512, mask_512 = self.preprocess(raw_img, raw_mask)

        #input to om model should be NCHW
        img_512_chw = img_512.transpose((2,0,1)).copy()
        mask_512_chw = mask_512[:,:,0:1]
        
        # neural network with Davinci offline model
        inpainted_512, attention, mask_512_new  = self.inpaint_model.Inference(img_512_chw,mask_512_chw)

        # post-processing
        res_raw_size = self.post_process(raw_img, img_large, mask_large, inpainted_512[0], img_512, mask_512_new[0], attention[0])

        print('processing time', time.time() - s)
        return res_raw_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint App")
    parser.add_argument('--images', type=str, default='./samples/testset')
    parser.add_argument('--masks', type=str, default='./samples/maskset')
    parser.add_argument('--inpaint_model', type=str, default='./inpaint.om')
    parser.add_argument('--matmul_model', type=str, default='./matmul.om')
    parser.add_argument('--output_dir', type=str, default='./samples_result')

    args = parser.parse_args()
    inpaint_app = Inpaint_App(args.images,args.masks,args.inpaint_model,args.matmul_model,args.output_dir)
    inpaint_app.execute()


