# Import the supported models dictionary
from vlmeval.config import supported_VLM

def get_model_answer(model_name: str, prompt: str, image_path: str = None):
    """
    Instantiate a model from supported_VLM, build a message, call its generate function, and return the answer.
    
    Args:
        model_name (str): The key for the supported model (e.g., "Eagle", "Kosmos2", etc.).
        prompt (str): The text prompt/question.
        image_path (str, optional): The path to the image file to include. Defaults to None.
        dataset (str, optional): A string to denote the dataset type, if needed (used by some models). Defaults to None.
    
    Returns:
        str: The generated answer from the model.
    """
    # Get the model class from the supported models dictionary
    model_cls = supported_VLM[model_name]
    # Instantiate the model object
    model = model_cls()
    ret = model.generate([image_path, prompt])
    return ret

if __name__ == "__main__":
    ms = ['flamingov2','Pixtral-12B','llava-internlm2-7b','llava-v1.5-7b-xtuner','llava-v1.5-13b-xtuner','Yi_VL_6B','Yi_VL_34B','MiniGPT-4-v2','instructblip_7b','instructblip_13b','Janus-Pro-7B','MiniCPM-o-2_6','cogvlm2-llama3-chat-19B','Ovis1.6-Gemma2-27B','Ovis2-16B','Aria','ola','deepseek_vl2','Llama-3.2-90B-Vision-Instruct','InternVL2_5-26B-MPO','InternVL2_5-78B-MPO']
    # ms = ['DeepSeek-VL2']
    # print(supported_VLM.keys())
    for model_name in ms:
        try:
            # Example usage:
            # model_name = "InternVL2-1B"  # Make sure this model name is in your supported_VLM dictionary
            prompt = "Describe the image"
            image_path = "../data/test.jpg"  # Use an existing image file path; omit or pass None if not needed
            
            answer = get_model_answer(model_name, prompt, image_path=image_path)
            print(model_name, "Generated Answer:", answer) 
        except Exception as e:
            # raise
            print(f"Error: {model_name} {e}")

