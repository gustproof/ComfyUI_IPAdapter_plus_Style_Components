import numpy as np
import torch
import pickle
import comfy.model_management as model_management
import folder_paths

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    ToPILImage,
    Normalize,
    InterpolationMode,
)

N_COMPONENTS = 100


class SCompBase:
    CATEGORY = "style_components"
    FUNCTION = "node_entry"


class SCompPCA(SCompBase):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"path": (folder_paths.get_filename_list("ipadapter"),)}}

    RETURN_TYPES = ("SCOMP_PCA",)

    def node_entry(self, path):
        path = folder_paths.get_full_path("ipadapter", path)
        with open(path, "rb") as f:
            return (pickle.load(f),)


class SCompComp2Style(SCompBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"style_components": ("SCOMP",), "style_pca": ("SCOMP_PCA",)}
        }

    RETURN_TYPES = ("EMBEDS",)

    def node_entry(self, style_components, style_pca):
        pca, std = style_pca
        return (style_components * std[:N_COMPONENTS] @ pca.components_ + pca.mean_,)


class SCompRandComps(SCompBase):
    RETURN_TYPES = ("SCOMP",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": (
                    "INT:seed",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "display": "number",
                    },
                )
            }
        }

    def node_entry(self, seed):
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            res = torch.randn((1, N_COMPONENTS))
            print("Random style components:", res[0])
            return res


class SCompText2Comps(SCompBase):
    RETURN_TYPES = ("SCOMP",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "comps_text": (
                    "STRING",
                    {"default": str([0] * N_COMPONENTS), "multiline": True},
                )
            }
        }

    def node_entry(self, comps_text):
        vals = [0] * N_COMPONENTS
        for i, w in enumerate(comps_text.replace(",", " ").split()):
            w = "".join(c for c in w if c in "1234567890.-eE")
            vals[i] = float(w)
        return torch.tensor(vals).reshape((1, N_COMPONENTS))


class SCompCompsScaler(SCompBase):
    RETURN_TYPES = ("SCOMP",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "comps": ("SCOMP",),
                "scaling_factor": (
                    "FLOAT",
                    dict(min=-3, max=3, step=0.05, default=1),
                ),
            }
        }

    def node_entry(self, comps, scaling_factor):
        return (comps * scaling_factor,)


class SCompSliders2Comps(SCompBase):
    RETURN_TYPES = ("SCOMP",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                f"comp_{i}": (
                    "FLOAT",
                    dict(min=-10, max=10, step=0.1),
                )
                for i in range(N_COMPONENTS)
            }
        }

    def node_entry(self, *args, **kwargs):
        return torch.tensor([*args, *kwargs.values()]).reshape((1, N_COMPONENTS))


class SCompStyleExtractorLoader(SCompBase):
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(path=(folder_paths.get_filename_list("ipadapter"),)))

    RETURN_TYPES = ("SCOMP_EXTRACTOR",)

    def node_entry(self, path):
        path = folder_paths.get_full_path("ipadapter", path)
        return (torch.load(path).to(model_management.get_torch_device()).eval(),)


class SCompStyleExtractor(SCompBase):
    @classmethod
    def INPUT_TYPES(s):
        return dict(
            required=dict(style_extractor=("SCOMP_EXTRACTOR",), image=("IMAGE",))
        )

    RETURN_TYPES = ("EMBEDS",)

    def node_entry(self, style_extractor, image):
        tf = Compose(
            [
                ToPILImage(),
                Resize(
                    size=336,
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                ToTensor(),
                Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
            ]
        )
        with torch.no_grad():
            image = tf(image[0].permute((2, 0, 1))).unsqueeze(0)
            return (style_extractor(image.to(model_management.get_torch_device())),)


classnames = dict(
    SCompComp2Style="SCompComp2Style",
    SCompPCA="SCompPCA",
    SCompRandComps="SCompRandComps",
    SCompSliders2Comps="SCompSliders2Comps",
    SCompText2Comps="SCompText2Comps",
    SCompStyleExtractorLoader="SCompStyleExtractorLoader",
    SCompStyleExtractor="SCompStyleExtractor",
    SCompCompsScaler="SCompCompsScaler",
)

SCOMP_NODE_DISPLAY_NAME_MAPPINGS = classnames
SCOMP_NODE_CLASS_MAPPINGS = {k: eval(k) for k in classnames}
