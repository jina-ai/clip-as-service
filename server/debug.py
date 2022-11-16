from transformers import AutoConfig
from clip_server.model.mclip_model import MultilingualCLIPModel, MCLIPConfig
from clip_server.model.openclip_model import OpenCLIPModel

config = MCLIPConfig(modelBase='M-CLIP/XLM-Roberta-Large-Vit-B-32')
name1 = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
name2 = 'M-CLIP/XLM-Roberta-Base-Vit-B-32'
name3 = 'laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k'
name4 = 'roberta-base'
model = MultilingualCLIPModel(name=name4)

# cfg = AutoConfig.from_pretrained('roberta-base')
