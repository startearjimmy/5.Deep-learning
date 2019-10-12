import _pickle as cPickle
import yaml
import pickle
import numpy as np

data_file ='dataset/train'
with open("SeqGAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
with open(data_file, 'rb') as f:
            # load pickle data
            data = cPickle.load(f)
            data = np.array(data)
            print('shape: ',data.shape)
            print('max: ',np.max(data))
            print('min: ',np.min(data))
            print('mean: ',np.mean(data))
            print('std: ',np.std(data))
            for line in data:
                parse_line = [int(x) for x in line]
                #print(len(line))    
                if len(parse_line) != config['SEQ_LENGTH']:
                    #self.token_stream.append(parse_line)
                    print(parse_line)

with open('./dataset/chords', 'rb') as fp:
    chord_ref = pickle.load(fp)
print(len(chord_ref))
print(chord_ref)
with open('./dataset/octaves', 'rb') as fp:
    octave_ref = pickle.load(fp)
print(len(octave_ref))
print(octave_ref)
for i in octave_ref:
    if(len(i)==3): print(i)
