from .pretrain_model import PretrainModel
from .pretrain_model_dqn_wo_self_atten import PretrainDQNWOSAModel
from .pretrain_model_dqn_wo_self_atten_gl import PretrainDQNWOSAGLModel 
from .pretrain_model_dqn_wo_self_atten_global import PretrainDQNWOSAGModel
from .pretrain_model_dqn_wo_self_atten_mlp_gl import PretrainDQNWOSAMLPGLModel


LIGHTNING_MODULES = {
    "pretrain": PretrainModel,
    "pretrain_llm": PretrainModel,
    "pretrain_llm_dqn_wo_self_atten": PretrainDQNWOSAModel,
    "pretrain_llm_dqn_wo_self_atten_gl": PretrainDQNWOSAGLModel,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl": PretrainDQNWOSAMLPGLModel,
    "pretrain_llm_dqn_wo_self_atten_global": PretrainDQNWOSAGModel
}
