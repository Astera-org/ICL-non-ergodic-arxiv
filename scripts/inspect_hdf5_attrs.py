import h5py
import sys

def inspect_attributes(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Attributes for HDF5 file: {file_path}")
            for key, value in f.attrs.items():
                print(f"  '{key}': {value}")
            
            if 'selected_categories_in_file' in f.attrs:
                print("\nSpecifically, 'selected_categories_in_file':")
                print(f"  {list(f.attrs['selected_categories_in_file'])}")
            else:
                print("\n'selected_categories_in_file' attribute not found.")

    except Exception as e:
        print(f"Error inspecting HDF5 file {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        hdf5_file_path = sys.argv[1]
        inspect_attributes(hdf5_file_path)
    else:
        print("Usage: python inspect_hdf5_attrs.py <path_to_hdf5_file>")
        # Default to inspecting the main training data file if no arg is given
        default_path = "data/custom_tokenized_data_chunked_len100.hdf5"
        print(f"No path provided, attempting to inspect default: {default_path}")
        inspect_attributes(default_path) 