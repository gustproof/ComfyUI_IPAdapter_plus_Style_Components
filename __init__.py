"""
 ██▓ ██▓███   ▄▄▄      ▓█████▄  ▄▄▄       ██▓███  ▄▄▄█████▓▓█████  ██▀███
▓██▒▓██░  ██▒▒████▄    ▒██▀ ██▌▒████▄    ▓██░  ██▒▓  ██▒ ▓▒▓█   ▀ ▓██ ▒ ██▒
▒██▒▓██░ ██▓▒▒██  ▀█▄  ░██   █▌▒██  ▀█▄  ▓██░ ██▓▒▒ ▓██░ ▒░▒███   ▓██ ░▄█ ▒
░██░▒██▄█▓▒ ▒░██▄▄▄▄██ ░▓█▄   ▌░██▄▄▄▄██ ▒██▄█▓▒ ▒░ ▓██▓ ░ ▒▓█  ▄ ▒██▀▀█▄
░██░▒██▒ ░  ░ ▓█   ▓██▒░▒████▓  ▓█   ▓██▒▒██▒ ░  ░  ▒██▒ ░ ░▒████▒░██▓ ▒██▒
░▓  ▒▓▒░ ░  ░ ▒▒   ▓▒█░ ▒▒▓  ▒  ▒▒   ▓▒█░▒▓▒░ ░  ░  ▒ ░░   ░░ ▒░ ░░ ▒▓ ░▒▓░
 ▒ ░░▒ ░       ▒   ▒▒ ░ ░ ▒  ▒   ▒   ▒▒ ░░▒ ░         ░     ░ ░  ░  ░▒ ░ ▒░
 ▒ ░░░         ░   ▒    ░ ░  ░   ░   ▒   ░░         ░         ░     ░░   ░
 ░                 ░  ░   ░          ░  ░                     ░  ░   ░
                        ░
             ·-—+ IPAdapter Plus Extension for ComfyUI +—-· ·
             Brought to you by Matteo "Matt3o/Cubiq" Spinelli
             https://github.com/cubiq/ComfyUI_IPAdapter_plus/
"""

from .IPAdapterPlus import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .StyleComponents import SCOMP_NODE_CLASS_MAPPINGS, SCOMP_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS |= SCOMP_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS |= SCOMP_NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
