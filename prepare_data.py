#!/path/to/python2.7.6/python
"""
Summary:  Prepare data. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: - 
"""
import os
import soundfile
from scipy.io import wavfile
import numpy as np
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import cPickle
import h5py
from sklearn import preprocessing

import prepare_data as pp_data
import config as cfg
import librosa


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


# zxy
# wav-format trans
def convert_16bit_to_float(audio):
    max_abs = abs(max(audio))
    min_abs = abs(min(audio))
    if max_abs > min_abs:
        deep = max_abs
    else:
        deep = min_abs
    if deep > 1 and deep < 32768:
        audio = audio / float(32768)
    return audio


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    # (fs,audio) = wavfile.read(path)
    # audio = convert_16bit_to_float(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


###
def create_mixture_csv(args):
    """Create csv containing mixture information.
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]

    Args:
      workspace: str, path of workspace.
      speech_dir: str, path of speech data.
      noise_dir: str, path of noise data.
      data_type: str, 'train' | 'test'.
      magnification: int, only used when data_type='train', number of noise
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger
          than the species of noises.
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    # magnification = args.magnification
    speech_size = args.speech_size
    fs = cfg.sample_rate

    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]
    noise_names = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]

    rs = np.random.RandomState(0)
    out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    print out_csv_path
    pp_data.create_folder(os.path.dirname(out_csv_path))

    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\t%s\t%s\t%s\n" % ("speech_name", "noise_name", "noise_onset", "noise_offset"))

    speech_names = rs.choice(speech_names, size=speech_size, replace=False)
    for speech_na in speech_names:
        # Read speech.
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, fs)  # zxy
        len_speech = len(speech_audio)

        # For training data, mix each speech with randomly picked #magnification noises.
        """
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
        # For test data, mix each speech with all noises. 
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception("data_type must be train | test!")
        """
        selected_noise_names = noise_names
        # Mix one speech with different noises many times.
        for noise_na in selected_noise_names:
            noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path, fs)  # zxy
            len_noise = len(noise_audio)

            if len_noise <= len_speech:
                noise_onset = 0
                nosie_offset = len_speech
            # If noise longer than speech then randomly select a segment of noise.
            else:
                noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
                nosie_offset = noise_onset + len_speech

            if cnt % 100 == 0:
                print (cnt)

            cnt += 1
            f.write("%s\t%s\t%d\t%d\n" % (speech_na, noise_na, noise_onset, nosie_offset))
    f.close()
    print(out_csv_path)
    print("Create %s mixture csv finished!" % data_type)


###
def calculate_mixture_features(args):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the
    features to disk.

    Args:
      workspace: str, path of workspace.
      speech_dir: str, path of speech data.
      noise_dir: str, path of noise data.
      data_type: str, 'train' | 'test'.
      snr: float, signal to noise ratio to be mixed.
    """

    print(args)
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    snr = args.snr
    """
    print('123')
    workspace = 'workspace'
    speech_dir = 'mini_data/test_speech'
    noise_dir = 'mini_data/test_noise'
    data_type = 'test'
    snr = 20
    """
    fs = cfg.sample_rate

    # Open mixture csv.
    mixture_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    with open(mixture_csv_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    t1 = time.time()
    cnt = 0
    for i1 in xrange(1, len(lis)):
        [speech_na, noise_na, noise_onset, noise_offset] = lis[i1]
        noise_onset = int(noise_onset)
        noise_offset = int(noise_offset)

        # Read speech audio.
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)

        # Read noise audio.
        noise_path = os.path.join(noise_dir, noise_na)
        (noise_audio, _) = read_audio(noise_path, target_fs=fs)

        # Repeat noise to the same length as speech.
        if len(noise_audio) < len(speech_audio):
            n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
            noise_audio_ex = np.tile(noise_audio, n_repeat)
            noise_audio = noise_audio_ex[0: len(speech_audio)]

        # Truncate noise to the same length as speech.
        else:
            noise_audio = noise_audio[noise_onset: noise_offset]

        # Scale speech to given snr.
        scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
        speech_audio *= scaler

        # Get normalized mixture, speech, noise.
        (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)

        # Write out mixed audio.
        out_bare_na = os.path.join("%s.%s" %
                                   (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0]))
        """
        out_audio_path = os.path.join(workspace, "mixed_audios", "spectrogram", 
            data_type, "%ddb" % int(snr), "%s.wav" % out_bare_na)
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, mixed_audio, fs)
        """
        # Extract spectrogram.
        mixed_complx_x = calc_sp(mixed_audio, mode='complex')
        speech_x = calc_sp(speech_audio, mode='magnitude')
        noise_x = calc_sp(noise_audio, mode='magnitude')

        # Write out features.
        out_feat_path = os.path.join(workspace, "features", "spectrogram",
                                     data_type, "%ddb" % int(snr), "%s.p" % out_bare_na)
        create_folder(os.path.dirname(out_feat_path))
        data = [mixed_complx_x, speech_x, noise_x, alpha, out_bare_na]
        cPickle.dump(data, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        # Print.
        if cnt % 100 == 0:
            print(cnt)
        cnt += 1

    print("Extracting feature time: %s" % (time.time() - t1))


def rms(y):
    """Root mean square.
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr.

    Args:
      s: ndarray, source1.
      n: ndarray, source2.
      snr: float, SNR.
      method: 'rms'.

    Outputs:
      float, scaler.
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio = 10. ** (float(snr) / 20.)  # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor


def additive_mixing(s, n):
    """Mix normalized source1 and source2.

    Args:
      s: ndarray, source1.
      n: ndarray, source2.

    Returns:
      mix_audio: ndarray, mixed audio.
      s: ndarray, pad or truncated and scalered source1.
      n: ndarray, scaled source2.
      alpha: float, normalize coefficient.
    """
    mixed_audio = s + n

    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha


def calc_sp(audio, mode):
    """Calculate spectrogram.

    Args:
      audio: 1darray.
      mode: string, 'magnitude' | 'complex'

    Returns:
      spectrogram: 2darray, (n_time, n_freq).
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
        audio,
        window=ham_win,
        nperseg=n_window,
        noverlap=n_overlap,
        detrend=False,
        return_onesided=True,
        mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x


###
# def pack_features(workspace,data_type,snr,n_concat,n_hop):
def pack_features(args):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file.

    Args:
      workspace: str, path of workspace.
      data_type: str, 'train' | 'test'.
      snr: float, signal to noise ratio to be mixed.
      n_concat: int, number of frames to be concatenated.
      n_hop: int, hop frames.
    """

    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    n_concat = args.n_concat
    n_hop = args.n_hop
    speech_size = args.speech_size
    """
    print('123')
    workspace = 'workspace'
    data_type = 'test'
    snr = 20
    n_concat = 7
    n_hop = 3
    speech_size = 2000
    """
    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)
    n_all = []  # zxy

    cnt = 0
    t1 = time.time()
    num = 0

    # Load all features.
    feat_dir = os.path.join(workspace, "features", "spectrogram", data_type, "%ddb" % int(snr))
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    create_folder(os.path.dirname(out_path))
    names = os.listdir(feat_dir)

    # rs = np.random.RandomState(0)
    # names = rs.choice(names, size=speech_size, replace=False)

    for na in names:
        # Load feature.
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with boarder values.
        n_pad = (n_concat - 1) / 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)
        noise_x = pad_with_border(noise_x, n_pad)  # zxy

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        x_all.append(mixed_x_3d)

        # Cut target spectrogram and take the center frame of each 3D segment.
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, (n_concat - 1) / 2, :]
        y_all.append(y)

        # Cut target spectrogram and take the center frame of each 3D segment. #zxy
        noise_x_3d = mat_2d_to_3d(noise_x, agg_num=n_concat, hop=n_hop)
        n = noise_x_3d[:, (n_concat - 1) / 2, :]
        n_all.append(n)

        # Print.
        if cnt % 100 == 0:
            print(cnt)

        # if cnt == 3: break
        if cnt % 5000 == 0:
            x_all = np.concatenate(x_all, axis=0)  # (n_segs, n_concat, n_freq)
            y_all = np.concatenate(y_all, axis=0)  # (n_segs, n_freq)
            n_all = np.concatenate(n_all, axis=0)  # (n_segs, n_freq)#zxy

            x_all = log_sp(x_all).astype(np.float32)
            y_all = log_sp(y_all).astype(np.float32)
            n_all = log_sp(n_all).astype(np.float32)  # zxy

            (aa, bb, cc) = x_all.shape
            # print(x_all.shape)

            if cnt == 0:
                with h5py.File(out_path, 'w') as hf:
                    hf.create_dataset('x', (aa, bb, cc), maxshape=(None, bb, cc))
                    hf.create_dataset('y', (aa, cc), maxshape=(None, cc))
                    hf.create_dataset('n', (aa, cc), maxshape=(None, cc))
            # if cnt != 0:
            # print("before add cnt=%d"%cnt)
            # data_size(workspace,data_type,snr)
            h5f = h5py.File(out_path, 'a')
            h5f['x'].resize([(num + aa), bb, cc])
            h5f['y'].resize([(num + aa), cc])
            h5f['n'].resize([(num + aa), cc])
            h5f['x'][num:num + aa] = x_all
            h5f['y'][num:num + aa] = y_all
            h5f['n'][num:num + aa] = n_all
            # save_h5py(x_all,y_all,n_all,out_path,cnt/10000)
            # print("after add cnt=%d"%cnt)
            # data_size(workspace,data_type,snr)
            num += aa
            x_all = []
            y_all = []
            n_all = []

        cnt += 1

    ##############################
    # print(num)
    """    
    if(len(x_all)!=0):        
        x_all = np.concatenate(x_all, axis=0)   # (n_segs, n_concat, n_freq)
        y_all = np.concatenate(y_all, axis=0)   # (n_segs, n_freq)
        n_all = np.concatenate(n_all, axis=0)   # (n_segs, n_freq)#zxy

        x_all = log_sp(x_all).astype(np.float32)
        y_all = log_sp(y_all).astype(np.float32)
        n_all = log_sp(n_all).astype(np.float32) #zxy

        print(x_all.shape)

        h5f = h5py.File(out_path, 'a')
        h5f['x'][(cnt-cnt%10000):(cnt+1)] = x_all
        h5f['y'][(cnt-cnt%10000):(cnt+1)] = y_all
        h5f['n'][(cnt-cnt%10000):(cnt+1)] = n_all

    # Write out data to .h5 file. 

    out_path = os.path.join(workspace,"packed_features","spectrogram",data_type,"%ddb"%int(snr),"data_%dth.h5"%(cnt/10000+1))
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        hf.create_dataset('n', data=n_all)
    """
    if (len(x_all) != 0):
        x_all = np.concatenate(x_all, axis=0)  # (n_segs, n_concat, n_freq)
        y_all = np.concatenate(y_all, axis=0)  # (n_segs, n_freq)
        n_all = np.concatenate(n_all, axis=0)  # (n_segs, n_freq)#zxy

        x_all = log_sp(x_all).astype(np.float32)
        y_all = log_sp(y_all).astype(np.float32)
        n_all = log_sp(n_all).astype(np.float32)  # zxy

        (aa, bb, cc) = x_all.shape
        # print(x_all.shape)

        h5f = h5py.File(out_path, 'a')
        h5f['x'].resize([(num + aa), bb, cc])
        h5f['y'].resize([(num + aa), cc])
        h5f['n'].resize([(num + aa), cc])
        h5f['x'][num:num + aa] = x_all
        h5f['y'][num:num + aa] = y_all
        h5f['n'][num:num + aa] = n_all

        num += aa

    # print("num=%d"%num)
    # print("after add cnt=%d"%cnt)
    data_size(workspace, data_type, snr)
    print("Write out to %s" % out_path)
    print("Pack features finished! %s s" % (time.time() - t1,))


def data_size(workspace, data_type, snr):
    hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        x = np.array(x)
    print("x.shape=")
    print(x.shape)


def save_h5py(x_all, y_all, n_all, out_path, i):
    print("save_h5py %dth" % i)
    x_all = np.concatenate(x_all, axis=0)  # (n_segs, n_concat, n_freq)
    y_all = np.concatenate(y_all, axis=0)  # (n_segs, n_freq)
    n_all = np.concatenate(n_all, axis=0)  # (n_segs, n_freq)#zxy

    x_all = log_sp(x_all).astype(np.float32)
    y_all = log_sp(y_all).astype(np.float32)
    n_all = log_sp(n_all).astype(np.float32)  # zxy

    # Write out data to .h5 file.
    out_path = os.path.join(out_path, "data_%dth.h5" % i)
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        hf.create_dataset('n', data=n_all)

    print("Write out to %s" % out_path)


def log_sp(x):
    return np.log(x + 1e-08)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments.
    """
    # Pad to at least one block.
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1: i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value.
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)


###
def compute_scaler(args):
    """Compute and write out scaler of data.
    """

    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    """

    workspace = "workspace"
    data_type = "test"
    snr = 20
    """
    # Load data.
    t1 = time.time()
    hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        x = np.array(x)  # (n_segs, n_concat, n_freq)
    print(x.shape)
    # Compute scaler.
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    print(scaler.mean_)
    print(scaler.scale_)

    # Write out scaler.
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))

    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %s s" % (time.time() - t1,))


def scale_on_2d(x2d, scaler):
    """Scale 2D array data.
    """
    return scaler.transform(x2d)


def scale_on_3d(x3d, scaler):
    """Scale 3D array data.
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))
    return x3d


def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data.
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]


###
def load_hdf5(hdf5_path):
    """Load hdf5 data.
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        n = hf.get('n')  # zxy
        x = np.array(x)  # (n_segs, n_concat, n_freq)
        y = np.array(y)  # (n_segs, n_freq)
        n = np.array(n)  # (n_segs, n_freq) #zxy
    return x, y, n


def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)
    parser_create_mixture_csv.add_argument('--speech_size', type=int, default=1)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)

    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    parser_pack_features.add_argument('--speech_size', type=int, default=1)

    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    parser_compute_scaler.add_argument('--snr', type=float, required=True)

    args = parser.parse_args()
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Error!")
