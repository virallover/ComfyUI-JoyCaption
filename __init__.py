from .JC import JC, JC_adv, JC_ExtraOptions

NODE_CLASS_MAPPINGS = {
    "JC": JC,
    "JC_adv": JC_adv,
    "JC_ExtraOptions": JC_ExtraOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JC": "JoyCaption",
    "JC_adv": "JoyCaption (Advanced)",
    "JC_ExtraOptions": "JoyCaption (Extra Options)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
