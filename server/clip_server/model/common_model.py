import open_clip
import torch

class CLIPModel:
    def __init__(self, model_name, device, jit):
        self._model = self._vision_preprocessor = self._tokenizer = None
        self._model_name = model_name
        self._device = device
        self._jit = jit

    @classmethod
    def load(cls, model_name, device, jit):
        device = torch.device(device)
        if '::' in model_name:
            name, pretrained = model_name.split('::')
            from clip_server.model.model_mclip import MultilingualCLIPModel
            customized_models = {
                "M-CLIP": MultilingualCLIPModel
                # "cn-clip": ...
            }
            if pretrained in customized_models:
                model_class = customized_models[pretrained]
                model, tokenizer, vision_preprocessor = model_class.load(model_name, device, jit)
            else:
                # we use open_clip loader as default
                model = cls(model_name, device, jit)
                model._model, _, vision_preprocessor = open_clip.create_model_and_transforms(name,
                                                                                         pretrained=pretrained,
                                                                                         device=device, jit=jit)
                tokenizer = open_clip.tokenize
            # model._model.eval().requires_grad_(False).to(device)  # mark
        else:
            raise ValueError(f'''
        Now, CLIP-as-service depends on `open-clip` which supports more CLIP variants and pretrained weights. 
        The new names is now a string in the format of `<model_name>::<pretrained_weights_name>`, e.g. 
        `ViT-B-32::openai` or `ViT-B-32::laion2b_e16`. 
        ''')
        return model, tokenizer, vision_preprocessor

    def encode_text(self, batch_data):
        return self._model.encode_text(batch_data)

    def encode_image(self, batch_data):
        return self._model.encode_image(batch_data)
