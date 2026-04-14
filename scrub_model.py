import h5py
import json
import zipfile
import os
import shutil

def recursive_remove(obj):
    if isinstance(obj, dict):
        key_found = False
        if 'quantization_config' in obj:
            print("  Found and removed quantization_config")
            del obj['quantization_config']
            key_found = True
        for key in list(obj.keys()):
            if recursive_remove(obj[key]):
                key_found = True
        return key_found
    elif isinstance(obj, list):
        any_found = False
        for item in obj:
            if recursive_remove(item):
                any_found = True
        return any_found
    return False

def scrub_model(path):
    print(f"Scrubbing {path}...")
    if not os.path.exists(path):
        print(f"  Error: File {path} not found.")
        return

    if zipfile.is_zipfile(path):
        print("  Detected ZIP archive (.keras format)")
        tmp_zip = path + ".tmp.zip"
        found = False
        with zipfile.ZipFile(path, 'r') as zin:
            with zipfile.ZipFile(tmp_zip, 'w') as zout:
                for item in zin.infolist():
                    buffer = zin.read(item.filename)
                    if item.filename == 'config.json':
                        config = json.loads(buffer.decode('utf-8'))
                        if recursive_remove(config):
                            found = True
                            buffer = json.dumps(config).encode('utf-8')
                    zout.writestr(item, buffer)
        if found:
            shutil.move(tmp_zip, path)
            print("  Save complete (modified).")
        else:
            os.remove(tmp_zip)
            print("  No quantization_config found.")
    else:
        print("  Detected HDF5 format (.h5)")
        with h5py.File(path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config = json.loads(config_str)
                
                if recursive_remove(config):
                    f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                    print("  Save complete (modified).")
                else:
                    print("  No quantization_config found.")
            else:
                print("  'model_config' not found in attributes.")

if __name__ == "__main__":
    # Scrub both just in case
    scrub_model('app/models/lstm_model/lstm_model.h5')
    scrub_model('app/models/lstm_model/lstm_model.keras')
