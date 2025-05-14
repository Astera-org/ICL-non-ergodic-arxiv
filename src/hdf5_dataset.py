import h5py
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from pathlib import Path

from .logging_config import get_logger

log = get_logger(__name__)

class HDF5SequentialDataset(Dataset):
    """
    A PyTorch Dataset to load all token chunks sequentially from an HDF5 file.

    The HDF5 file is expected to contain:
    - 'token_chunks': Dataset of [N_total_chunks, chunk_size]
    - Attributes on the HDF5 file or 'token_chunks' dataset (optional but good practice):
        - 'total_chunks': Total number of chunks
        - 'chunk_size': The length of each token chunk (e.g., 100)
    """
    def __init__(self, cfg: DictConfig, split: str = "train"):
        """
        Initializes the sequential HDF5 dataset loader.

        Args:
            cfg: Hydra configuration object. Expected to have:
                 cfg.dataset.path: Path to the HDF5 file.
                 cfg.dataset.token_chunk_size: Expected chunk size (e.g. 100)
            split (str): The dataset split to load (e.g., "train", "validation", "test").
                         Currently, this implementation assumes a single HDF5 file contains
                         all data, and the split argument is for future extension or if
                         different HDF5 files per split are used. For now, it mainly
                         influences logging.
        """
        self.hdf5_path_str = cfg.dataset.get("path", None)
        if not self.hdf5_path_str:
            raise ValueError("cfg.dataset.path must be specified for HDF5SequentialDataset.")
        
        self.hdf5_path = Path(self.hdf5_path_str)
        self.split = split
        self.token_chunk_size = cfg.dataset.get("token_chunk_size", 100) # Default to 100 if not in cfg

        if not self.hdf5_path.exists():
            log.error(f"HDF5 file not found for split '{self.split}' at: {self.hdf5_path}")
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        try:
            with h5py.File(self.hdf5_path, 'r') as hf:
                if 'token_chunks' not in hf:
                    log.error(f"'token_chunks' dataset not found in HDF5 file: {self.hdf5_path}")
                    raise ValueError(f"'token_chunks' dataset not found in {self.hdf5_path}")
                
                self.num_sequences = hf['token_chunks'].shape[0]
                actual_chunk_size = hf['token_chunks'].shape[1]

                # Verify chunk size
                if actual_chunk_size != self.token_chunk_size:
                    log.warning(
                        f"HDF5 file '{self.hdf5_path}' has chunk size {actual_chunk_size}, "
                        f"but cfg.dataset.token_chunk_size is {self.token_chunk_size}. "
                        f"Using actual chunk size from HDF5: {actual_chunk_size}."
                    )
                    self.token_chunk_size = actual_chunk_size
                
                log.info(f"Successfully opened HDF5 file for split '{self.split}': {self.hdf5_path}")
                log.info(f"Found {self.num_sequences} sequences (chunks) of length {self.token_chunk_size}.")

        except Exception as e:
            log.error(f"Error initializing HDF5SequentialDataset for split '{self.split}' with file {self.hdf5_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if not (0 <= idx < self.num_sequences):
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_sequences} sequences.")

        try:
            # Open HDF5 file in each __getitem__ call. This is generally recommended for
            # use with PyTorch DataLoader when num_workers > 0, as h5py file handles
            # are not generally safe to be shared across processes.
            # OS-level caching should mitigate performance concerns.
            with h5py.File(self.hdf5_path, 'r') as hf:
                token_chunk = hf['token_chunks'][idx, :]
            
            return {'input_ids': torch.tensor(token_chunk, dtype=torch.long)}
        
        except Exception as e:
            log.error(f"Error loading item at index {idx} from HDF5 file {self.hdf5_path} for split '{self.split}': {e}", exc_info=True)
            # Depending on the error, might want to return a dummy item or raise a more specific error.
            raise

if __name__ == '__main__':
    # Example Usage (requires a dummy config and HDF5 file)
    from omegaconf import OmegaConf
    from .logging_config import setup_logging # Assuming relative import works if run as module

    # Dummy config
    dummy_cfg = OmegaConf.create({
        'dataset': {
            'path': 'data/custom_tokenized_data_chunked_len100.hdf5', # Ensure this file exists
            'token_chunk_size': 100,
        },
        'logging': { # Add logging config
            'level': 'INFO',
            'log_file': 'hdf5_sequential_dataset_test.log'
        }
    })

    # Setup logging using the config
    setup_logging(cfg=dummy_cfg) # Pass the whole cfg object

    log.info("--- Testing HDF5SequentialDataset ---")

    # Check if the dummy HDF5 file exists before trying to load
    hdf5_test_path = Path(dummy_cfg.dataset.path)
    if not hdf5_test_path.exists():
        log.error(f"Test HDF5 file '{hdf5_test_path}' not found. Cannot run example usage.")
        log.info("Please create a dummy HDF5 file or point to an existing one.")
        # Example: Create a small dummy HDF5 file
        # with h5py.File(hdf5_test_path, 'w') as hf:
        #     hf.create_dataset('token_chunks', data=np.random.randint(0, 30000, size=(10, 100), dtype=np.int32))
        # log.info(f"Created a dummy HDF5 file at '{hdf5_test_path}' for testing.")
    else:
        try:
            train_dataset = HDF5SequentialDataset(cfg=dummy_cfg, split="train")
            log.info(f"Length of train_dataset: {len(train_dataset)}")

            if len(train_dataset) > 0:
                sample_item = train_dataset[0]
                log.info(f"First item (train_dataset[0]): {sample_item}")
                assert 'input_ids' in sample_item
                assert sample_item['input_ids'].shape == torch.Size([dummy_cfg.dataset.token_chunk_size])
                assert sample_item['input_ids'].dtype == torch.long
                log.info("Sample item structure and type are correct.")

                # Test with DataLoader
                from torch.utils.data import DataLoader
                train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
                
                for i, batch in enumerate(train_dataloader):
                    log.info(f"Batch {i+1}:")
                    log.info(f"  input_ids shape: {batch['input_ids'].shape}")
                    assert batch['input_ids'].shape[0] <= 2 # Batch size
                    assert batch['input_ids'].shape[1] == dummy_cfg.dataset.token_chunk_size # Sequence length
                    if i >= 2: # Test a few batches
                        break
                log.info("DataLoader test successful.")

        except Exception as e:
            log.error(f"Error during example usage: {e}", exc_info=True) 