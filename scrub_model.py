import h5py
import json

def scrub_h5(path):
    print(f"Scrubbing {path}...")
    with h5py.File(path, 'r+') as f:
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            
            def remove_key(obj):
                if isinstance(obj, dict):
                    if 'quantization_config' in obj:
                        print("  Found and removed quantization_config")
                        del obj['quantization_config']
                    for key in obj:
                        remove_key(obj[key])
                elif isinstance(obj, list):
                    for item in obj:
                        remove_key(item)

            remove_key(config)
            f.attrs['model_config'] = json.dumps(config).encode('utf-8')
            print("  Save complete.")
        else:
            print("  'model_config' not found in attributes.")

scrub_h5('app/models/lstm_model/lstm_model.h5')
