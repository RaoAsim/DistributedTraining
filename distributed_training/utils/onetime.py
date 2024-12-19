import time
import logging
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console (captured by PM2)
    ]
)

# Initialize local epoch variable
local_epoch = None


def model_download(global_epoch):
    """Download the model for the specified epoch revision."""
    logging.info(f"downloading started")
    model = AutoModelForCausalLM.from_pretrained(
        "distributed/optimized-gpt2-1b",
        revision=str(global_epoch),
        trust_remote_code=True,
    )
    logging.info(f"Model downloaded for epoch {global_epoch}")
    return model

def main():
    """Main function to check and update the model if the epoch has changed."""
    global local_epoch
    
  
    try:
            global_epoch = 361
            if global_epoch is not None and global_epoch != local_epoch:
                # Update the model if there's a new global epoch
                model = model_download(global_epoch)
                local_epoch = global_epoch
                logging.info(f"Updated local epoch to {local_epoch}")
            else:
                logging.info("No update required.")
    except Exception as e:
            logging.error(f"An error occurred: {e}")
        
        # Wait for 5 minutes before the next check
        

if __name__ == "__main__":
    main()
