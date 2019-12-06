import os

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_csv(df, out_path, out_filename, export_index=None):
    create_directory(out_path)
    out_file = f'{out_path}{out_filename}'
    df.to_csv(out_file, sep=';', index=export_index, header=True)
    
    print(f'Generated: {out_file}')