from . import text_model
from . import bert_model
from . import vision_model
from . import dqn
from . import dqn_wo_self_atten
from . import dqn_wo_self_atten_mlp
from . import cnn_backbones
from . import fusion_module
from . import mrm_pretrain_model
from . import CARZero_model_dqn_wo_self_atten
from . import CARZero_model_dqn_wo_self_atten_gl
from . import CARZero_model_dqn_wo_self_atten_global
from . import CARZero_model_dqn_wo_self_atten_gl_mlp

IMAGE_MODELS = {
    "pretrain_llm_dqn_wo_self_atten": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_global": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_gl": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl": vision_model.ImageEncoder,
}
