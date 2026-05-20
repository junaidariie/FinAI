from langchain_community.chat_models import ChatLlamaCpp
from huggingface_hub import hf_hub_download
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_llm_instance = None

def get_model():
    try:
        global _llm_instance

        if _llm_instance is None:
            model_path = hf_hub_download(
                repo_id="junaid17/qwen2.5-finance-assistant-gguf",
                filename="qwen2.5-finance-assistant-q4_k_m.gguf",
            )

            logger.info(f"Loading model from: {model_path}")

            _llm_instance = ChatLlamaCpp(
                model_path=model_path,
                temperature=0.5,
                max_tokens=2048,
                n_ctx=4096,
                n_batch=512,
                n_threads=max(4, os.cpu_count() // 2),
                n_gpu_layers=0,
                verbose=False,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()] 
            )

            logger.info("Model loaded successfully!")
    except Exception as e:
        logger.exception(f"Error while loading the model, {str(e)}")

    return _llm_instance
