import os
from vlmeval.config import supported_VLM

class OpenVLLM:
    def __init__(self, model_name: str = None):
        """
        Initialize the OpenVLLM object by instantiating the underlying model 
        from the supported_VLM dictionary.
        
        Args:
            model_name (str, optional): Name of the model to use. If not provided, 
                                        it is read from the 'model_name' environment variable.
        """
        if model_name is None:
            model_name = os.environ.get("model_name")
            if model_name is None:
                raise ValueError("Model name must be provided either as an argument or in the environment variable 'model_name'.")
        self.model_name = model_name

        if model_name == 'InstructBLIP-7B':
            from .vlm_open.instructblip_7b import InstructBLIP
            self.model = InstructBLIP()
        elif model_name == 'InstructBLIP-13B':
            from .vlm_open.instructblip_13b import InstructBLIP
            self.model = InstructBLIP()
        else:
            if model_name not in supported_VLM:
                raise ValueError(
                    f"Model '{model_name}' is not supported. Supported models are: {list(supported_VLM.keys())}."
                )
            model_cls = supported_VLM[model_name]
            self.model = model_cls()
    
    def generate(self, system: str, prompt: str, image_paths: list = None) -> str:
        """
        Generate an answer using the underlying model.
        
        Args:
            system (str): A system-level instruction or context message. This will be 
                          prepended to the prompt if provided.
            prompt (str): The text prompt or question to be answered.
            image_paths (list, optional): A list of image file paths. Defaults to None.
            
        Returns:
            str: The generated answer from the model.
        """
        # Prepend the system message to the prompt if one is provided.
        if system:
            full_prompt = f"{system}\n{prompt}"
        else:
            full_prompt = prompt
        
        # Call the generate method of the underlying model.
        # Here, we package the image_paths and the prompt into a list to mimic the original call:
        #   ret = model.generate([image_path, prompt])
        ret = self.model.generate(image_paths + [full_prompt])
        return ret
