# Written and organized by Cathy Y. Li

import sys
import argparse
import pickle
import torch
import clip, open_clip
import numpy as np
from PIL import Image
from einops import rearrange
from glob import glob
from time import time
from torchvision import transforms

######################
# Compute Embeddings #
######################
def load_txt_img(pickle_file_path):
    """
    Load the text-image pairs from the given path to pickle file. 
    Input: 
     - [str] pickle_file_path, path to data pickle file. 
             pickle file format: 
                { 'img': [array] image,
                  'text': [str] text }
    Output:
     - [str] text
     - [image] image (512x512)
    """
    data = pickle.load(open(pickle_file_path, "rb"))
    image_transforms_ = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image = Image.fromarray(data['img'])
    image = image_transforms_(image)
    image = image.unsqueeze(0)
    image = torch.clamp((image.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    image = 255. * rearrange(image[0], 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(np.squeeze(image.astype(np.uint8))) # squeeze to deal with gray image
    return data['text'], image


class CLIP(object):
    """
    Model to compute clip embeddings for images and text. 
    """
    def __init__(self, model_name):
        assert model_name in ["openai", 'laion']
        self.model_name = model_name
        self.device = "cuda"
        if model_name == "openai":
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            tokenizer = clip.tokenize
        elif model_name == 'laion':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
            tokenizer = open_clip.get_tokenizer('ViT-H-14')
        model = model.cuda()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def emb(self, images, text):
        """
        Computes the normalized clip embeddings for the given image and text. 
        Input:
         - [list] images, list of length n
         - [list] text, list of length m
        Output:
         - [array] normalized clip embeddings of the images, shape nx512
         - [array] normalized clip embeddings of the text, shape mx512
        """
        images = [self.preprocess(i).unsqueeze(0).to("cuda") for i in images]
        images = torch.concat(images)
        text = self.tokenizer(text, truncate=True).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text)
            img_emb_normalized = image_features / (image_features.norm(dim=-1, keepdim=True))
            txt_emb_normalized = text_features / (text_features.norm(dim=-1, keepdim=True))
        return img_emb_normalized, txt_emb_normalized


class DataSet:
    def __init__(self, glob_data_path, save_path):
        self.batch_size = 512
        self.start, self.batch_count = time(), 0
        self.txt_batch, self.img_batch, self.ids = [], [], []
        self.result = {}
        self.data_path = glob_data_path
        self.save_path = save_path
        print('Data path:', self.data_path, 'data size:', len(glob(self.data_path)))

    def compute(self):
        """
        Compute the normalized clip embeddings for all the texts and images in the dataset. 
        Write the results in self.save_path every 10 batches and when job finishes. 
        """
        for i in glob(self.data_path):
            if i in self.result: # embedding has been computed
                continue
            text, image = load_txt_img(i)
            self.img_batch.append(image)
            self.txt_batch.append(text)
            self.ids.append(i)
            if len(self.txt_batch) == self.batch_size:
                self.batch_count += 1
                self.compute_batch(save = self.batch_count % 10 == 0)
                self.txt_batch, self.img_batch, self.ids = [], [], []
        self.compute_batch(save = True)
        print('Completed. num samples:', len(self.result), (time()-self.start)/60, 'minutes')

    def compute_batch(self, save=False):
        """
        Update self.result with the normalized clip embeddings for self.txt_batch and self.img_batch. 
        If save, write self.results at self.save_path. 
        """
        clip_model = CLIP("openai")
        img_emb, txt_emb = clip_model.emb(self.img_batch, self.txt_batch)
        img_emb, txt_emb = img_emb.cpu().numpy(), txt_emb.cpu().numpy()
        for j,id in enumerate(self.ids):
            self.result[id] = {'img_emb': img_emb[j].reshape(-1), 'txt_emb': txt_emb[j].reshape(-1)}
        if save:
            pickle.dump(self.result, open(self.save_path, 'wb'))
            print(f'saved {self.batch_count} batches, {(time()-self.start)/60} minutes')
    

##############
# Compute AD #
##############
        
def laion_scale_clip(clip_sim):
    """
    Processes the clip embedding similarity for LAION dataset: clips to [0.2, 0.45] and scales [0.2, 0.45] to [0,1].
    Input: 
     - [array] clip_sim, each entry (float) being the clip similarity of a LAION data point's image and text embeddings.
    Output: 
     - [array] clip_sim, with the same length as input, each entry (float) scaled and clipped.
    """
    clip_sim = (clip_sim - 0.2)*4
    clip_sim[clip_sim<0] = 0
    clip_sim[clip_sim>1] = 1
    return clip_sim

def compute_AD(save_emb_path, alpha=0.8):
    """
    Computes the AD of a given dataset.
    Inputs: 
     - [str] save_emb_path, path to normalized clip embeddings pickle file of the dataset. 
             pickle file format: 
                { [str] filename_1: {
                                      'img_emb': [array] clip embedding of the image, 
                                      'txt_emb': [array] clip embedding of the corresponding text
                                    },
                  [str] filename_2: { ... }, 
                  ...,
                  [str] filename_n: { ... }
                }
     - [float] alpha, float in [0,1]. Trade-off of feature and structure AD. Defaults to 0.8. 
    Output: 
     - [float] AD, the alignment difficulty of the dataset.
    """
    assert 0 <= alpha <= 1
    # load image and text embeddings. 
    emb = pickle.load(open(save_emb_path, 'rb'))
    img_clip, txt_clip = [], []
    for key in emb.keys():
        datap = emb[key]
        img_clip.append(datap['img_emb'])
        txt_clip.append(datap['txt_emb'])
    n = len(img_clip)
    print('Loaded embeddings. Dataset size:', n)
    img_clip, txt_clip = np.array(img_clip, dtype=np.float32), np.array(txt_clip, dtype=np.float32)

    # compute feature distance
    clip_sim = laion_scale_clip(np.sum(img_clip * txt_clip, axis=1))
    feature = n - np.sum(clip_sim) # for each image-text pair, it's 1 - clip_similarity
    # compute structure distance
    structure = 0
    for i in range(n-1):
        if (i % 10000) == 0 and i > 0: 
            # print the progress every 10000 data points
            print(f'structure distance computed for {i} data points, sum = {structure}') 
        cos_sim_img = np.dot(img_clip[i], img_clip[i+1:].T)
        cos_sim_txt = np.dot(txt_clip[i], txt_clip[i+1:].T)
        structure += np.sum(np.abs(cos_sim_img - cos_sim_txt))
    # compute AD with feature and structure distance
    Dfea = feature / n
    Dstru = structure * 2 / (n**2)
    ad = alpha * Dfea + (1-alpha) * Dstru
    print(f'feature AD = {Dfea}, structure AD = {Dstru}, AD (alpha = {alpha}) = {ad}')
    return ad

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_glob_path', type=str, help="Glob path of image-text pair pickle files of the dataset. ", default=None)
    parser.add_argument('-save_emb_path', type=str, help="Path to save the normalized CLIP embeddings. ", default=None)
    parser.add_argument('-alpha', type=float, help="Trade-off parameter for feature and structure AD, in [0,1]. ", default=0.8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    dataset = DataSet(args.data_glob_path, args.save_emb_path)
    dataset.compute()
    compute_AD(args.save_emb_path, args.alpha)