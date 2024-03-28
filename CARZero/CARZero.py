import os
import torch
import numpy as np
import copy
import random
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn.functional  as F
from . import builder
from . import utils
from . import constants
from .models.vision_model import PretrainedImageClassifier
from typing import Union, List
from scipy import ndimage
# import matplotlib.pyplot as plt
from tqdm import tqdm
import time


np.random.seed(10)
random.seed(10)


_MODELS = {
    "CARZero_resnet50": "",
    "CARZero_vit_b_16": ".pretrain_model/CARZero_best_model.ckpt",
}



_FEATURE_DIM = {"CARZero_resnet50": 2048, "CARZero_vit_b_16": 768 }


def available_models() -> List[str]:
    """Returns the names of available CARZero models"""
    return list(_MODELS.keys())



def load_CARZero(
    name: str = "CARZero_vit_b_16",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a CARZero model

    Parameters
    ----------
    name : str
        A model name listed by `CARZero.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    CARZero_model : torch.nn.Module
        The CARZero model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            + "Make sure to download the pretrained weights from \n"
            + "    https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh \n"
            + " and copy it to the ./pretrained folder."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    # CARZero_model = builder.build_CARZero_dqn_llm_model(cfg).to(device)
    # CARZero_model = builder.build_CARZero_dqn_wo_self_atten_model(cfg).to(device)
    CARZero_model = builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(cfg).to(device)
    # CARZero_model = builder.build_CARZero_dqn_self_atten_local_model(cfg).to(device)

    model_weights = CARZero_model.state_dict()

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("CARZero.")[-1]
        if new_key in model_weights:
            fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict
    CARZero_model.load_state_dict(ckpt_dict, strict=True)
    model_weights
    return CARZero_model


def load_img_classification_model(
    name: str = "CARZero_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    num_cls: int = 1,
    freeze_encoder: bool = True,
):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `CARZero.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    num_cls: int
        Number of output classes
    freeze_encoder: bool
        Freeze the pretrained image encoder

    Returns
    -------
    img_model : torch.nn.Module
        The CARZero pretrained image classification model
    """

    # load pretrained image encoder
    CARZero_model = load_CARZero(name, device)
    image_encoder = copy.deepcopy(CARZero_model.img_encoder)
    del CARZero_model

    # create image classifier
    feature_dim = _FEATURE_DIM[name]
    img_model = PretrainedImageClassifier(
        image_encoder, num_cls, feature_dim, freeze_encoder
    )

    return img_model



def get_similarities(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both", 'atten']:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        img_emb_l, img_emb_g = CARZero_model.image_encoder_forward(imgs)
        text_emb_l, text_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )
        
        #  # ram extract features
        # with torch.no_grad():
        #     ram_features = CARZero_model.ram_extract_forward(ram_imgs)

        # # ram features
        # text_emb_g = CARZero_model.ram_forward(ram_features, text_emb_g)


    # get similarities
    global_similarities = CARZero_model.get_global_similarities(img_emb_g, text_emb_g)
    # ipdb.set_trace()
    local_similarities, attention_maps = CARZero_model.get_local_similarities(
        img_emb_l, text_emb_l, txts["cap_lens"], return_atten=True
    )
    similarities = (local_similarities + global_similarities) / 2

    # ipdb.set_trace()

    if similarity_type == "global":
        return global_similarities.detach().cpu().numpy()
    elif similarity_type == "local":
        return local_similarities.detach().cpu().numpy()
    elif similarity_type == "both":
        return similarities.detach().cpu().numpy()
    elif similarity_type == 'atten':
        attention_maps = torch.from_numpy(attention_maps.repeat(16, axis=1).repeat(16, axis=2))#Final 
        return attention_maps


def get_query_similarities(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

        # entry query result
        B = label_img_emb_g.size(0)

        query_emb_g = query_emb_g.unsqueeze(1).repeat(1, B, 1)

        label_img_emb_l = label_img_emb_l.view(label_img_emb_l.size(0), label_img_emb_l.size(1), -1) 

        label_img_emb_l = label_img_emb_l.permute(2, 0, 1) #patch_num b dim
        # 
        cls = CARZero_model.fusion_module(query_emb_g, label_img_emb_l).squeeze(-1)

        cls = torch.sigmoid(cls)

        # ipdb.set_trace()

        return cls.detach().cpu().numpy()


    # get similarities
    # global_similarities = CARZero_model.get_global_similarities(img_emb_g, text_emb_g)
    # local_similarities = CARZero_model.get_local_similarities(
    #     img_emb_l, text_emb_l, txts["cap_lens"]
    # )
    # similarities = (local_similarities + global_similarities) / 2

    # if similarity_type == "global":
    #     return global_similarities.detach().cpu().numpy()
    # elif similarity_type == "local":
    #     return local_similarities.detach().cpu().numpy()
    # else:
    #     return similarities.detach().cpu().numpy()


def cos_proj2proj(cnn_code, rnn_code, eps=1e-8, temp3=10.0):
    bs_img = cnn_code.shape[0]
    bs_text = cnn_code.shape[1]
    # ipdb.set_trace()
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    cnn_code = cnn_code.reshape(bs_text * bs_img, 1, -1)
    rnn_code = rnn_code.reshape(bs_text * bs_img, 1, -1)

    cnn_code_norm = cnn_code_norm.reshape(bs_text * bs_img, 1, -1)
    rnn_code_norm = rnn_code_norm.reshape(bs_text * bs_img, 1, -1)


    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores0 = scores0.reshape(bs_img, bs_text).transpose(1, 0)
    return scores0


def get_dqn_cos_similarities(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        CARZero_model.eval()
        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )
        cls_bs = []
        # bs = label_img_emb_g.size(0)
        # for i in range(bs):
        label_img_emb_l_ = label_img_emb_l.view(label_img_emb_l.size(0), label_img_emb_l.size(1), -1) 

        label_img_emb_g_ = label_img_emb_g

        label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) #patch_num b dim

        query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1) 

        query_emb_l_ = query_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, 512, 768]

        # 
        proj_img, atten_i2t = CARZero_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_atten=True)

        #  
        # i2t_cls = i2t_cls.squeeze(-1)  ## use text as query, use image as k, v, so image batch size have not distrubed the result

        # # 
        proj_text, atten_t2i = CARZero_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_atten=True)
        # torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1)
        
        cls = cos_proj2proj(proj_img.transpose(1, 0), proj_text)

        # label_img_emb_g = label_img_emb_g.unsqueeze(1).repeat(1, query_emb_g.size(0), 1) 
        # query_emb_g = query_emb_g.unsqueeze(1).repeat(1, label_img_emb_g.size(0), 1)

        # i2t_cls = cos_proj2proj(proj_img.transpose(1, 0), query_emb_g)
        # t2i_cls = cos_proj2proj(label_img_emb_g, proj_text.transpose(1, 0))

        #     # cls = t2i_g_cls
        # cls = (i2t_cls + t2i_cls.transpose(1, 0)) / 2


        return cls.detach().cpu().numpy()



    # get similarities
    # global_similarities = CARZero_model.get_global_similarities(img_emb_g, text_emb_g)
    # local_similarities = CARZero_model.get_local_similarities(
    #     img_emb_l, text_emb_l, txts["cap_lens"]
    # )
    # similarities = (local_similarities + global_similarities) / 2

    # if similarity_type == "global":
    #     return global_similarities.detach().cpu().numpy()
    # elif similarity_type == "local":
    #     return local_similarities.detach().cpu().numpy()
    # else:
    #     return similarities.detach().cpu().numpy()

 
def get_dqn_similarities(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        CARZero_model.eval()
        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

        cls_bs = []
        it2_SimR_bs = []
        t2i_SimR_bs = []
        bs = label_img_emb_g.size(0)
        for i in range(bs):
            label_img_emb_l_ = label_img_emb_l[i:i+1].view(label_img_emb_l[i:i+1].size(0), label_img_emb_l[i:i+1].size(1), -1) 

            label_img_emb_g_ = label_img_emb_g[i:i+1]

            label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) #patch_num b dim

            query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1) 

            query_emb_l_ = query_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, 512, 768]

            # label_img_emb_l_ = CARZero_model.multi_modal_vision_proj(label_img_emb_l_)
            # label_img_emb_g_ = CARZero_model.multi_modal_vision_proj(label_img_emb_g_)
            # query_emb_l_ = CARZero_model.multi_modal_language_proj(query_emb_l_)
            # query_emb_g = CARZero_model.multi_modal_language_proj(query_emb_g)
            # 
            # i2t_cls, atten_i2t, it2_SimR = CARZero_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_atten=True, return_SimR=True)

            i2t_cls, atten_i2t = CARZero_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_atten=True)

            # torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1)
            #  pos=CARZero_model.img_encoder.model.pos_embed[:, 1:, :].transpose(0,1), 
            #  mode='local',
            #  
            i2t_cls = i2t_cls.squeeze(-1)  ## use text as query, use image as k, v, so image batch size have not distrubed the result

            # # 
            # t2i_cls, atten_t2i, t2i_SimR = CARZero_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_atten=True, return_SimR=True)
            t2i_cls, atten_t2i = CARZero_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_atten=True)
            # torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1)

            # ipdb.set_trace()
            t2i_cls = t2i_cls.squeeze(-1).transpose(1,0) 

            
            # it2_SimR_bs.append(it2_SimR)
            # t2i_SimR_bs.append(t2i_SimR)
      

            # i2t_g_cls = CARZero_model.fusion_module_global(label_img_emb_g[i:i+1], query_emb_g).squeeze(-1) # don't change by different batch size
            # t2i_g_cls = CARZero_model.fusion_module_global(query_emb_g, label_img_emb_g[i:i+1]).squeeze(-1) # change by different batch size, large batch size will get better result
            # t2i_g_cls = t2i_g_cls.transpose(1,0)

            # i2t_g_cls = CARZero_model.fusion_module(label_img_emb_g[i:i+1], query_emb_g, mode='global').squeeze(-1) # don't change by different batch size

            # t2i_g_cls = CARZero_model.fusion_module(query_emb_g, label_img_emb_g[i:i+1], mode='global').squeeze(-1) # change by different batch size, large batch size will get better result
            # t2i_g_cls = t2i_g_cls.transpose(1,0)


            # cls = t2i_g_cls
            cls = (i2t_cls + t2i_cls) / 2

            cls_bs.append(cls)

        # cls = i2t_cls
        # it2_SimR_bs = torch.cat(it2_SimR_bs, dim=0)
        # t2i_SimR_bs = torch.cat(t2i_SimR_bs, dim=1)
        # np.save('/haoranlai/Project/CARzero/padchest_inference/it2_SimR.npy', it2_SimR_bs.detach().cpu().numpy())
        # np.save('/haoranlai/Project/CARzero/padchest_inference/t2i_SimR.npy', t2i_SimR_bs.detach().cpu().numpy())
        cls = torch.cat(cls_bs, dim=0)

        return cls.detach().cpu().numpy()


def get_dqn_similarities_fast(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        CARZero_model.eval()

        start = time.time()
        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

        label_img_emb_l_ = label_img_emb_l.view(label_img_emb_l.size(0), label_img_emb_l.size(1), -1) 

        label_img_emb_g_ = label_img_emb_g

        label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) #patch_num b dim

        query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1) 

        query_emb_l_ = query_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, 512, 768]


        i2t_cls, atten_i2t = CARZero_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_atten=True)
    
        i2t_cls = i2t_cls.squeeze(-1)  ## use text as query, use image as k, v, so image batch size have not distrubed the result

    
        t2i_cls, atten_t2i = CARZero_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_atten=True)

        t2i_cls = t2i_cls.squeeze(-1).transpose(1,0) 

        # cls = t2i_g_cls
        cls = (i2t_cls + t2i_cls) / 2

        step_time = time.time() - start
        
        
        # cls = i2t_cls
        # it2_SimR_bs = torch.cat(it2_SimR_bs, dim=0)
        # t2i_SimR_bs = torch.cat(t2i_SimR_bs, dim=1)
        # np.save('/haoranlai/Project/CARzero/padchest_inference/it2_SimR.npy', it2_SimR_bs.detach().cpu().numpy())
        # np.save('/haoranlai/Project/CARzero/padchest_inference/t2i_SimR.npy', t2i_SimR_bs.detach().cpu().numpy())

        return cls.detach().cpu().numpy(), step_time



    # get similarities
    # global_similarities = CARZero_model.get_global_similarities(img_emb_g, text_emb_g)
    # local_similarities = CARZero_model.get_local_similarities(
    #     img_emb_l, text_emb_l, txts["cap_lens"]
    # )
    # similarities = (local_similarities + global_similarities) / 2

    # if similarity_type == "global":
    #     return global_similarities.detach().cpu().numpy()
    # elif similarity_type == "local":
    #     return local_similarities.detach().cpu().numpy()
    # else:
    #     return similarities.detach().cpu().numpy()


def get_report_detection(CARZero_model, imgs, txts, img_path, index,text_len, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        CARZero_model.eval()
        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, sents = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )
        # ipdb.set_trace()
        cls_bs = []
        it2_SimR_bs = []
        t2i_SimR_bs = []
        bs = label_img_emb_g.size(0)
        for i in range(bs):
            label_img_emb_l_ = label_img_emb_l[i:i+1].view(label_img_emb_l[i:i+1].size(0), label_img_emb_l[i:i+1].size(1), -1) 

            label_img_emb_g_ = label_img_emb_g[i:i+1]

            label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) #patch_num b dim

            used_query =  query_emb_l[i * text_len : (i + 1) * text_len]

            query_emb_l_ = used_query.view(used_query.size(0), used_query.size(1), -1) 
         
            query_emb_l_ = query_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, 512, 768]

            # 
            i2t_cls, atten_i2t, it2_SimR = CARZero_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g[i * text_len : (i + 1) * text_len], return_atten=True, return_SimR=True)
            # torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1)
            #  pos=CARZero_model.img_encoder.model.pos_embed[:, 1:, :].transpose(0,1), 
            #  mode='local',
            #  
            i2t_cls = i2t_cls.squeeze(-1)  ## use text as query, use image as k, v, so image batch size have not distrubed the result
            
            # # 
            t2i_cls, atten_t2i, t2i_SimR = CARZero_model.fusion_module(torch.cat([query_emb_g[i * text_len : (i + 1) * text_len].unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_atten=True, return_SimR=True)

            # ipdb.set_trace()
            t2i_cls = t2i_cls.squeeze(-1).transpose(1,0) 

            # pred_map = conve_word_attention_map(atten_t2i, batch_size=bs)  
            # os.makedirs('/haoranlai/Project/CARzero/ReportDetection/output/' + str(index), exist_ok=True)
            # plot_word_attention(sents, pred_map[:, :, 1:], os.path.join('/haoranlai/Project/CARzero/ReportDetection/output/' + str(index), img_path[0].split('/')[-1].replace('.png', '')))

            # cls = t2i_g_cls
            cls = (i2t_cls + t2i_cls) / 2

            cls_bs.append(cls)

        # cls = i2t_cls
        # it2_SimR_bs = torch.cat(it2_SimR_bs, dim=0)
        # t2i_SimR_bs = torch.cat(t2i_SimR_bs, dim=1)
        # np.save('/haoranlai/Project/CARzero/padchest_inference/it2_SimR.npy', it2_SimR_bs.detach().cpu().numpy())
        # np.save('/haoranlai/Project/CARzero/padchest_inference/t2i_SimR.npy', t2i_SimR_bs.detach().cpu().numpy())
        cls = torch.cat(cls_bs, dim=0)
        return cls.detach().cpu().numpy()


def get_report_detection_similarities(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both", 'atten']:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        img_emb_l, img_emb_g = CARZero_model.image_encoder_forward(imgs)
        text_emb_l, text_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )
        global_similarities_bs = []
        local_similarities_bs = []
        similarities_bs = []
        bs = img_emb_g.size(0)
        for i in range(bs):        
            # get similarities
            global_similarities = CARZero_model.get_global_similarities(img_emb_g[i:i+1], text_emb_g[i * 2 : (i + 1) * 2])
            local_similarities, attention_maps = CARZero_model.get_local_similarities(
                img_emb_l[i:i+1], text_emb_l[i * 2 : (i + 1) * 2], txts["cap_lens"][i * 2 : (i + 1) * 2], return_atten=True
            )
            similarities = (local_similarities + global_similarities) / 2

            global_similarities_bs.append(global_similarities)
            local_similarities_bs.append(local_similarities)
            similarities_bs.append(similarities)


    # ipdb.set_trace()
    global_similarities = torch.cat(global_similarities_bs, dim=0)
    local_similarities = torch.cat(local_similarities_bs, dim=0)
    similarities = torch.cat(similarities_bs, dim=0)
    if similarity_type == "global":
        return global_similarities.detach().cpu().numpy()
    elif similarity_type == "local":
        return local_similarities.detach().cpu().numpy()
    elif similarity_type == "both":
        return similarities.detach().cpu().numpy()
    elif similarity_type == 'atten':
        attention_maps = torch.from_numpy(attention_maps.repeat(16, axis=1).repeat(16, axis=2))#Final 
        return attention_maps


def plot_word_attention(tokens, attention_weights, output_filename):
    batch_size  = attention_weights.size(0)
    attention_weights = attention_weights.cpu().numpy()
    for b in range(batch_size):
        # 创建图表
        sents = tokens[b]
        filtered_tokens = [token for token in sents if "PAD" not in token]
        sequence_length = len(filtered_tokens)
        fig, ax = plt.subplots(figsize=(sequence_length, 2))
        cax = ax.imshow([attention_weights[b, 0, :sequence_length]], aspect="auto", cmap="hot")
        ax.set_yticks([])
        ax.set_xticks(range(sequence_length))
        ax.set_xticklabels(filtered_tokens)
         # 添加颜色条
        plt.colorbar(cax, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{output_filename}{b}.png', dpi=300, format='png', bbox_inches='tight')
        plt.close()

def gaussian_kernel(kernel_size=5, sigma=1.0):
    """生成一个二维高斯核"""
    coords = torch.arange(kernel_size).float()
    coords -= (kernel_size - 1) / 2.0
    g = (1 / (2 * np.pi * sigma ** 2)) * torch.exp(-(coords ** 2 + coords.view(-1, 1) ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def apply_gaussian_blur_tensor(image_tensor, kernel):
    """使用高斯核对图像进行卷积"""
    # image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    blurred_tensor = F.conv2d(image_tensor, kernel.unsqueeze(0).unsqueeze(0), padding=kernel.size(0)//2)
    # return blurred_tensor.squeeze(0).squeeze(0)  # 移除批次和通道维度
    return blurred_tensor

def clip_ws(atten_weights):
    bz = atten_weights.size(0)
    patch_attenn_weight = []
    for i in range(bz):
        atten_weight = atten_weights[i]
        # nonzero = atten_weight.nonzero().squeeze()
        low = torch.quantile(atten_weight, 0.01)
        high = torch.quantile(atten_weight, 0.99)
        atten_weight = atten_weight.clip(low, high)
        patch_attenn_weight.append(atten_weight.clone())
    atten_weights = torch.stack(patch_attenn_weight, dim=0)
    return atten_weights
       
def conve_attention_map(ws, batch_size):
    
    ws = (ws[-4] +ws[-3]+ws[-2]+ws[-1])/4
    # ipdb.set_trace()
    # img_feature = img_feature.mean(dim=1, keepdim=True)
    ws = ws.view(batch_size,ws.shape[1],14,14)
    # ws = clip_ws(ws.detach().cpu())

    # ws = ws * img_feature
    # gaussian_kernel_matrix = gaussian_kernel( kernel_size=3, sigma=1.0).to(ws.device)
    # smmoth_ws = apply_gaussian_blur_tensor(ws, gaussian_kernel_matrix)
    # ipdb.set_trace()
    # ws_ = F.softmax(ws * 100.0, dim=1) ## add softmax
    # pred_map = ws[:,0,:,:].detach().cpu().numpy()
    pred_map = ws.detach().cpu().numpy()
    # v_max = np.percentile(pred_map, 80)
    # v_min = np.percentile(pred_map, 1)
    # # ipdb.set_trace()
    # pred_map = np.clip(pred_map, v_min, v_max)
    # s_pred_map = []
    # for b in range(batch_size):
    #     s_pred_map.append(np.expand_dims(ndimage.gaussian_filter(pred_map[b], sigma=(1.0, 1.0), order=0), axis=0 ))
    # s_pred_map = np.concatenate(s_pred_map, axis=0)
    # pred_map = torch.from_numpy(pred_map.repeat(16, axis=1).repeat(16, axis=2))#Final 
    pred_map_expand = torch.from_numpy(pred_map.repeat(16, axis=2).repeat(16, axis=3))#Final 
    # 如果expanded_map的尺寸是1022x1022，需要在每个方向上增加1个像素
    # if pred_map.shape[2] == 1022:
    #     pred_map = torch.nn.functional.pad(pred_map, (1, 1, 1, 1))
    # pred_map = F.interpolate(ws, size=(224, 224), mode='bilinear', align_corners=False)[:,0,:,:].detach().cpu()
    return pred_map_expand, pred_map
    
def conve_word_attention_map(ws, batch_size):
    ws = (ws[-4] +ws[-3]+ws[-2]+ws[-1])/4
    # ws = ws.view(batch_size,ws.shape[1],14,14)

    # pred_map = ws.detach().cpu().numpy()
    # pred_map_expand = torch.from_numpy(pred_map.repeat(16, axis=2).repeat(16, axis=3))#Final 
    # 如果expanded_map的尺寸是1022x1022，需要在每个方向上增加1个像素
    # if pred_map.shape[2] == 1022:
    #     pred_map = torch.nn.functional.pad(pred_map, (1, 1, 1, 1))
    # pred_map = F.interpolate(ws, size=(224, 224), mode='bilinear', align_corners=False)[:,0,:,:].detach().cpu()
    return ws


def get_grounding_map(CARZero_model, imgs, txts, similarity_type="both"):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    txts:
        processed text using CARZero_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use CARZero_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use CARZero_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        CARZero_model.eval()
        label_img_emb_l, label_img_emb_g = CARZero_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = CARZero_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )
        # cls_bs = []
        bs = label_img_emb_g.size(0)
        # for i in range(bs):

        label_img_emb_l_ = label_img_emb_l.view(label_img_emb_l.size(0), label_img_emb_l.size(1), -1)  #patch_num  dim

        label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1)

        query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1) 

        query_emb_l_ = query_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, 512, 768]

        # label_img_emb_l_ = CARZero_model.multi_modal_vision_proj(label_img_emb_l_)
        # label_img_emb_g = CARZero_model.multi_modal_vision_proj(label_img_emb_g)
        # query_emb_l_ = CARZero_model.multi_modal_language_proj(query_emb_l_)
        # query_emb_g = CARZero_model.multi_modal_language_proj(query_emb_g)

        # 
        i2t_cls, atten_i2t = CARZero_model.fusion_module(torch.cat([label_img_emb_g.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_atten=True)
        
        # 
        atten_i2t = [atten[:,:, 1:] for atten in atten_i2t]
        #  , pos=CARZero_model.img_encoder.model.pos_embed[:, 1:, :].transpose(0,1)
        #  mode='local',
        #  
        i2t_cls = i2t_cls.squeeze(-1)  ## use text as query, use image as k, v, so image batch size have not distrubed the result

        # # 
        t2i_cls, atten_t2i = CARZero_model.fusion_module(query_emb_l_, label_img_emb_g, return_atten=True)

        t2i_cls = t2i_cls.squeeze(-1).transpose(1,0)   
        
        # ipdb.set_trace()
        pred_map, pred_map_min = conve_attention_map(atten_i2t, batch_size=bs)    

            # # cls = t2i_g_cls
            # cls = (i2t_cls + t2i_cls) / 2

        # cls_bs.append(pred_map)

         # # cls = i2t_cls
        
        # cls = torch.cat(cls_bs, dim=0)

        return pred_map, pred_map_min


    # get similarities
    # global_similarities = CARZero_model.get_global_similarities(img_emb_g, text_emb_g)
    # local_similarities = CARZero_model.get_local_similarities(
    #     img_emb_l, text_emb_l, txts["cap_lens"]
    # )
    # similarities = (local_similarities + global_similarities) / 2

    # if similarity_type == "global":
    #     return global_similarities.detach().cpu().numpy()
    # elif similarity_type == "local":
    #     return local_similarities.detach().cpu().numpy()
    # else:
    #     return similarities.detach().cpu().numpy()



def text_embedding(CARZero_model, cls_txt_mapping):
    with torch.no_grad():
        text_emb_l, text_emb_g, _ = CARZero_model.text_encoder_forward(
            cls_txt_mapping["caption_ids"], cls_txt_mapping["attention_mask"], cls_txt_mapping["token_type_ids"]
        )
    return text_emb_l, text_emb_g


def image_embedding(CARZero_model, imgs):
        # get global and local image features
    with torch.no_grad():
        img_emb_l, img_emb_g = CARZero_model.image_encoder_forward(imgs)
    return img_emb_l, img_emb_g


def zero_shot_classification(CARZero_model, imgs, cls_txt_mapping):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    class_similarities = []
    for cls_name, cls_txt in cls_txt_mapping.items():
        similarities = get_similarities(
            CARZero_model, imgs, cls_txt, similarity_type="both"
        )
        cls_similarity = similarities.max(axis=1)  # average between class prompts
        class_similarities.append(cls_similarity)
    class_similarities = np.stack(class_similarities, axis=1)

    # normalize across class
    if class_similarities.shape[0] > 1:
        class_similarities = utils.normalize(class_similarities)
    class_similarities = pd.DataFrame(
        class_similarities, columns=cls_txt_mapping.keys()
    )
    return class_similarities


def zero_shot_fast_classification(CARZero_model, imgs, cls_txt_mapping):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    cap_lens = []
    for cls_name, txts in cls_txt_mapping.items():
        # ipdb.set_trace()
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])
        cap_lens += txts["cap_lens"]
    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids, "cap_lens": cap_lens}

    # for cls_name, cls_txt in cls_txt_mapping.items():
    # class_similarities = get_similarities(
    #         CARZero_model, imgs, text_batch, similarity_type="both"
    #     )
    
    class_similarities = get_similarities(
            CARZero_model, imgs, text_batch, similarity_type="both"
        )
    
    # cls_similarity = similarities.max(axis=1)  # average between class prompts
    # class_similarities.append(similarities)
    # class_similarities = np.concatenate(class_similarities, axis=0)
    # ipdb.set_trace()
    # normalize across class
    if class_similarities.shape[0] > 1:
        class_similarities = utils.normalize(class_similarities, method='identity')
    class_similarities = pd.DataFrame(
        class_similarities, columns=cls_txt_mapping.keys()
    )
    return class_similarities


def query_shot_classification(CARZero_model, imgs, cls_txt_mapping):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    # class_similarities = []
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    for cls_name, txts in cls_txt_mapping.items():
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])

    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}

    cls_similarity = get_query_similarities(
        CARZero_model, imgs, text_batch, similarity_type="both"
    )
        # cls_similarity = similarities.max(axis=1)  # average between class prompts
    #     class_similarities.append(cls_similarity)

    # class_similarities = np.concatenate(class_similarities, axis=1)

    # ipdb.set_trace()

    # # normalize across class
    # if class_similarities.shape[0] > 1:
    #     class_similarities = utils.normalize(class_similarities)
    class_similarities = pd.DataFrame(
        cls_similarity, columns=cls_txt_mapping.keys()
    )
    return class_similarities


def dqn_shot_classification(CARZero_model, imgs, cls_txt_mapping):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    # class_similarities = []
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    for cls_name, txts in cls_txt_mapping.items():
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])

    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}

    cls_similarity = get_dqn_similarities(
        CARZero_model, imgs, text_batch, similarity_type="both"
    )
    # cls_similarity, step_time = get_dqn_similarities_fast(
    #     CARZero_model, imgs, text_batch, similarity_type="both"
    # )

    # used_time = pd.DataFrame(data={'bs': [imgs.size(0)], 'time': [step_time]})
    # cls_similarity = get_dqn_cos_similarities(
    #     CARZero_model, imgs, text_batch, similarity_type="both"
    # )
        # cls_similarity = similarities.max(axis=1)  # average between class prompts
    #     class_similarities.append(cls_similarity)

    # class_similarities = np.concatenate(class_similarities, axis=1)

    # ipdb.set_trace()

    # # normalize across class
    # if class_similarities.shape[0] > 1:
    #     class_similarities = utils.normalize(class_similarities)
    class_similarities = pd.DataFrame(
        cls_similarity, columns=cls_txt_mapping.keys()
    )
    return class_similarities


def retriveal_shot_classification(CARZero_model, imgs, cls_txt_mapping, text_bs=256):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    # class_similarities = []
    total = len(cls_txt_mapping)
    # split into batches
    cls_txt_mapping_ = list(cls_txt_mapping.items())
    cls_txt_mapping_ = [cls_txt_mapping_[i : i + text_bs] for i in range(0, total, text_bs)]
    class_similarities = []
    for cls_txt_mapping_batch in tqdm(cls_txt_mapping_):
        caption_ids = []
        attention_mask = []
        token_type_ids = []
        for cls_name, txts in cls_txt_mapping_batch:
            caption_ids.append(txts["caption_ids"])
            attention_mask.append(txts["attention_mask"])
            token_type_ids.append(txts["token_type_ids"])
        caption_ids = torch.cat(caption_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}
        cls_similarity = get_dqn_similarities(
            CARZero_model, imgs, text_batch, similarity_type="both")
        class_similarities.append(cls_similarity)
    class_similarities = np.concatenate(class_similarities, axis=1)
    class_similarities = pd.DataFrame(class_similarities, columns=cls_txt_mapping.keys())
    return class_similarities


def report_detection_classification(CARZero_model, imgs, img_path, index, cls_txt_mapping):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    # class_similarities = []\
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    cap_lens = []
    sents = []
    for get_txts in cls_txt_mapping:
        for cls_name, txts in get_txts.items():
            caption_ids.append(txts["caption_ids"])
            attention_mask.append(txts["attention_mask"])
            token_type_ids.append(txts["token_type_ids"])
            cap_lens += txts["cap_lens"]
            sents.append(txts['sents'])
    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids, "sents": sents, "cap_lens": cap_lens}
    cls_similarity = get_report_detection(
        CARZero_model, imgs, text_batch, img_path, index, len(list(cls_txt_mapping[0].keys())), similarity_type="both"
    )

    # cls_similarity = get_report_detection_similarities( CARZero_model, imgs, text_batch, similarity_type="both")

    # class_similarities = np.concatenate(class_similarities, axis=1)

    # ipdb.set_trace()

    # # normalize across class
    # if class_similarities.shape[0] > 1:
    #     class_similarities = utils.normalize(class_similarities)
    if len(list(cls_txt_mapping[0].keys())) == 2:
        columns_name = ['correct', 'incorrect']
    elif len(list(cls_txt_mapping[0].keys())) == 1:
        columns_name = ['report']
    class_similarities = pd.DataFrame(
        cls_similarity, columns=columns_name
    )
    return class_similarities


def grounding_shot_classification(CARZero_model, imgs, cls_txt_mapping):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    # class_similarities = []
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    for cls_name, txts in cls_txt_mapping.items():
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])

    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}

    pred_map, pred_map_min = get_grounding_map(
        CARZero_model, imgs, text_batch, similarity_type="both"
    )



    # class_similarities = pd.DataFrame(
    #     cls_similarity, columns=cls_txt_mapping.keys()
    # )
    return pred_map, pred_map_min


def CARZero_grounding_fast_classification(CARZero_model, imgs, cls_txt_mapping):
    """Load a CARZero pretrained classification model

    Parameters
    ----------
    CARZero_model : str
        CARZero model, load via CARZero.load_models()
    imgs:
        processed images using CARZero_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    cap_lens = []
    for cls_name, txts in cls_txt_mapping.items():
        # ipdb.set_trace()
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])
        cap_lens += txts["cap_lens"]
    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids, "cap_lens": cap_lens}

    # for cls_name, cls_txt in cls_txt_mapping.items():
    # class_similarities = get_similarities(
    #         CARZero_model, imgs, text_batch, similarity_type="both"
    #     )
    
    class_similarities = get_similarities(
            CARZero_model, imgs, text_batch, similarity_type="atten"
        )
    
    # cls_similarity = similarities.max(axis=1)  # average between class prompts
    # class_similarities.append(similarities)
    # class_similarities = np.concatenate(class_similarities, axis=0)
    # ipdb.set_trace()
    # normalize across class
    if class_similarities.shape[0] > 1:
        class_similarities = utils.normalize(class_similarities, method='identity')
    # class_similarities = pd.DataFrame(
    #     class_similarities, columns=cls_txt_mapping.keys()
    # )
    return class_similarities





