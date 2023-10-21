import soundata
data_path = '/vast/sk8974/experiments/dsynth/data/util_data/3d_marco/'
dataset = soundata.initialize('marco', data_home=data_path)
dataset.download(force_overwrite=True, cleanup=True)