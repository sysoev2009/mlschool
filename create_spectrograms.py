import os, sys, re, random, pathlib, math
import numpy as np
import matplotlib.pyplot as mpl
from scipy.io.wavfile import read
import numpy.fft as nfft
from scipy import signal
from PIL import Image

mpl.switch_backend('agg')
mpl.figure(figsize=(12,5))
max_samlpes_len = 150000
max_hz = 2500

def save(samples, path):
	f, t, Sxx = signal.spectrogram(samples, 16000)
	j = 0
	while j < len(f):
		if f[j] > max_hz:
			f[j] = max_hz
		j += 1
	
	mpl.pcolormesh(t, f, Sxx)
	mpl.savefig(path)
	spectrogram = Image.open(path)
	cropped_spectrogram = spectrogram.crop((152, 62, 1079, 445))
	cropped_spectrogram.save(path)
	mpl.clf()

f = open('data_v_7_stc/meta/meta.txt')
line = f.readline()
filename = []
label = []
i = 0
while i < 11306:
	line = f.readline()
	line = line.strip()
	line = re.sub(r'\s+', ' ', line)
	line = line.split(' ')
	filename.append(line[0])
	label.append(line[4])
	i += 1
f.close()

folders = list(set(label))
pathlib.Path('train').mkdir(parents=True, exist_ok=True)
for folder in folders:
	pathlib.Path('train/'+folder).mkdir(parents=True, exist_ok=True)

i = 0
while i < len(filename):
	n = 0
	rate, samples_from_file = read('data_v_7_stc/audio/' + filename[i])
	samples = samples_from_file
	add_samples = np.array(np.zeros(500))
	samples = np.append(add_samples, samples)
		
	if len(samples) > max_samlpes_len:
		j = 0
		new_samples = np.array(np.zeros(max_samlpes_len))
		while j < len(new_samples):
			new_samples[j] = samples[j]
			j += 1
		samples = new_samples
			
	if len(samples) < max_samlpes_len:
		add_samples = np.zeros(max_samlpes_len - len(samples))
		samples = np.append(samples, add_samples)
		
	path = 'train/'+label[i]+'/'+filename[i].split('.')[0]+'_'+str(n)+'.png'
	save(samples, path)

	n += 1
	samples = samples_from_file
	add_samples = np.array(np.zeros(int(random.uniform(0, 30000))))
	samples = np.append(add_samples, samples)
	
	if len(samples) > max_samlpes_len:
		j = 0
		new_samples = np.array(np.zeros(max_samlpes_len))
		while j < len(new_samples):
			new_samples[j] = samples[j]
			j += 1
		samples = new_samples
			
	if len(samples) < max_samlpes_len:
		add_samples = np.zeros(max_samlpes_len - len(samples))
		samples = np.append(samples, add_samples)
		
	path = 'train/'+label[i]+'/'+filename[i].split('.')[0]+'_'+str(n)+'.png'
	save(samples, path)
	
	sys.stdout.write("\rCreating train spectrograms %d/%d" % (i+1, len(filename)))
	sys.stdout.flush()
	i += 1
print()

pathlib.Path('test').mkdir(parents=True, exist_ok=True)
pathlib.Path('validation').mkdir(parents=True, exist_ok=True)
for folder in folders:
	pathlib.Path('validation/'+folder).mkdir(parents=True, exist_ok=True)
	
filename = os.listdir('data_v_7_stc/test')
i = 0
while i < len(filename):
	label = filename[i].split('_')[0]
	if label == 'knocking':
		label = 'knocking_door'
	if label == 'unknown':
		label = 'test'

	rate, samples = read('data_v_7_stc/test/' + filename[i])
	add_samples = np.array(np.zeros(500))
	samples = np.append(add_samples, samples)
		
	if len(samples) > max_samlpes_len:
		j = 0
		new_samples = np.array(np.zeros(max_samlpes_len))
		while j < len(new_samples):
			new_samples[j] = samples[j]
			j += 1
		samples = new_samples
		
	if len(samples) < max_samlpes_len:
		add_samples = np.zeros(max_samlpes_len - len(samples))
		samples = np.append(samples, add_samples)
		
	f, t, Sxx = signal.spectrogram(samples, 16000)
	j = 0
	while j < len(f):
		if f[j] > max_hz:
			f[j] = max_hz
		j += 1
	
	if label == 'test':
		path = label+'/'+filename[i].split('.')[0]+'.png';
	else:
		path = 'validation/'+label+'/'+filename[i].split('.')[0]+'.png';
	save(samples, path)

	sys.stdout.write("\rCreating validation and test spectrograms %d/%d" % (i+1, len(filename)))
	sys.stdout.flush()
	i += 1
print()