
from numpy import *
from tensorflow import keras
import FlowCal
import h5py
import json
import warnings
from os import getenv
from os.path import join, isfile, basename
import sys



model_dir = join(getenv('PWD'), 'cnn_models')
file_c8   = 'cnn_model_12x12_c8.h5'
file_c3   = 'cnn_model_12x12_c3.h5'
file_c2   = 'cnn_model_12x12_c2.h5'



def print_usage():
    print("USAGE:")
    print("   python {0:} job_ID, tube_A.fcs tube_B.fcs output.json".format(basename(sys.argv[0])))
    return



class probability(object):

    def __init__(self, model):
        self.classes  = model.get_classes()
        self.git_hash = model.get_hash()
        self.prob     = None
        return

    def __call__(self, name):
        return self.get_probability(name)

    def get_hash(self):
        return self.git_hash

    def get_probability(self, name):
        if name not in self.classes:
            raise ValueError("Error: '{0:}' not found".format(name))
        idx = self.classes.index(name)
        return float(self.prob[idx])

    def get_classes(self, saas_input=True):
        map_classes = [['CLL + MBL', 'CM'], \
                       ['CLL + MBL + MCL + PL', 'a'], \
                       ['LPL + MZL + FL + HCL', 'b'], \
                       ['CLL + FL + HCL + LPL + MBL + MCL + MZL + PL', 'pathogen']]
        if saas_input:
            classes = self.classes.copy()
            for search, replace in map_classes:
                try:
                    idx          = classes.index(search)
                    classes[idx] = replace
                except ValueError:
                    pass
            return classes
        else:
            return self.classes

    def predict_class(self):
        idx = argmax(self.prob)
        return self.classes[idx]

    def predict_class_probability(self, name):
        idx = self.classes.index(name)
        return self.prob[idx]

    def store(self, probabilities):
        self.prob = probabilities[0,:]
        return



class keras_model(object):

    def __init__(self, file_dir, file_model):
        self.filename   = join(file_dir, file_model)
        self.git_hash   = None
        self.model      = None
        self.classes    = []
        return

    def get_hash(self):
        return self.git_hash

    def get_classes(self):
        return self.classes

    def load_keras_model(self):
        print("Loading model '{0:}'".format(basename(self.filename)))
        self.model = keras.models.load_model(self.filename)
        with h5py.File(self.filename, "r") as fhandle:
            self.git_hash = fhandle['git_hash'][()]
            self.classes  = list(fhandle['classes'][()])
        self.classes = [entry.decode("utf-8") for entry in self.classes]
        return

    def predict(self, data):
        prob = probability(self)
        prob.store(self.model.predict(data.features))
        return prob



class dataset(object):

    def __init__(self, file_tube_a, file_tube_b, nr_bins=12, \
                 select_2d=['t1_02x09', 't1_03x09', 't1_04x09', 't1_05x09', 't1_06x07', \
                            't1_06x10', 't1_09x10', 't2_02x03', 't2_05x06', 't2_06x09', \
                            't2_07x09', 't2_09x10']):
        self.file_id_ok  = False
        self.file_tube_a = file_tube_a
        self.file_tube_b = file_tube_b
        self.nr_bins     = nr_bins
        self.select_2d   = select_2d
        self.data        = {}
        self.nr_features = self.get_number_of_entries()
        self.features    = zeros([1, self.nr_features])
        self.grid_xy     = linspace(0.0, 1.0, self.nr_bins + 1, dtype='float32')
        return

    def get_number_of_entries(self):
        nr_elem = len(self.select_2d) * self.nr_bins**2
        return nr_elem

    def load_single_fcs_file(self, filename, tube):
        if not isfile(filename):
            print("Error: FCS file for tube '{0:}' not found".format(tube))
            raise FileNotFoundError
        if tube.lower() == 'a':
            key_t = 'tube_a'
            key_m = 'marker_a'
        elif tube.lower() == 'b':
            key_t = 'tube_b'
            key_m = 'marker_b'
        else:
            raise Exception("Error: Invalid argument for 'tube'")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp = FlowCal.io.FCSFile(filename)
        self.data[key_t] = tmp.data.astype('float32')
        self.data[key_m] = tmp.text
        return

    def verify_and_match_tubes(self):
        file_id       = '$FIL'
        markers_tube1 = {'$P3S': 'FMC7-FITC', \
                         '$P4S': 'CD10-PE', \
                         '$P5S': 'IgM-ECD', \
                         '$P6S': 'CD79b-PC5.5', \
                         '$P7S': 'CD20-PC7', \
                         '$P8S': 'CD23-APC', \
                         '$P10S': 'CD19-APCA750', \
                         '$P11S': 'CD5-PacBlue'}
        markers_tube2 = {'$P3S': 'Kappa-FITC', \
                         '$P4S': 'Lambda-PE', \
                         '$P7S': 'CD11c-PC7', \
                         '$P8S': 'CD103-APC'}

        file_id_a       = self.data['marker_a'][file_id].split(' ')[0]
        file_id_b       = self.data['marker_b'][file_id].split(' ')[0]
        self.file_id_ok = file_id_a == file_id_b
        if not self.file_id_ok:
            print("WARNING: Mismatch of tube 1/2 file IDs")

        a_is_1 = True
        b_is_1 = True
        for key, entry in markers_tube1.items():
            a_is_1 = a_is_1 and self.data['marker_a'][key] == entry
            b_is_1 = b_is_1 and self.data['marker_b'][key] == entry
        a_is_2 = True
        b_is_2 = True
        for key, entry in markers_tube2.items():
            a_is_2 = a_is_2 and self.data['marker_a'][key] == entry
            b_is_2 = b_is_2 and self.data['marker_b'][key] == entry
        if not ((a_is_1 ^ b_is_1) and (a_is_2 ^ b_is_2) and (a_is_1 != a_is_2)):
            raise Exception("Error: Invalid marker combination in input files")

        if a_is_1:
            self.data['tube1'] = self.data['tube_a']
            self.data['tube2'] = self.data['tube_b']
            self.file_tube_1   = self.file_tube_a
            self.file_tube_2   = self.file_tube_b
        else:
            self.data['tube1'] = self.data['tube_b']
            self.data['tube2'] = self.data['tube_a']
            self.file_tube_1   = self.file_tube_b
            self.file_tube_2   = self.file_tube_a
        return

    def normalize_data(self):
        self.data['tube1'] *= 1.0 / 1023.0
        self.data['tube2'] *= 1.0 / 1023.0
        return

    def load_fcs_files(self):
        self.load_single_fcs_file(self.file_tube_a, 'a')
        self.load_single_fcs_file(self.file_tube_b, 'b')
        self.verify_and_match_tubes()
        self.normalize_data()
        return

    def select_channels_2d(self):
        for entry in self.select_2d:
            tube     = entry.split('_')[0].replace('t', 'tube')
            ch1, ch2 = entry.split('_')[1].split('x')
            ch1      = int(ch1)
            ch2      = int(ch2)
            yield tube, ch1, ch2

    def compute_features(self):
        idx2 = 0
        for tube, idx_x, idx_y in self.select_channels_2d():
            samples_x = self.data[tube][:,idx_x]
            samples_y = self.data[tube][:,idx_y]
            h, ex, ey = histogram2d(samples_x, samples_y, bins=(self.grid_xy, self.grid_xy), \
                                    density=True)
            h         = log(h + 1.0)
            self.features[0,idx2:idx2+self.nr_bins**2] = h.flat
            idx2     += self.nr_bins**2
        assert(idx2 == self.nr_features)
        return



def save_predictions(prob_c8, prob_c3, prob_c2, data, job_id, \
                     file_dir=join(getenv('HOME'), 'data/storage/files/mll/predictions'), \
                     file_json='classification.json'):
    fname  = join(file_dir, file_json)
    output = {"anfrageId": job_id, \
              "files": [data.file_tube_1, data.file_tube_2], \
              "inputConsistent": data.file_id_ok, \
              "prediction": {"model": [prob_c8.get_hash(), \
                                       prob_c3.get_hash(), \
                                       prob_c2.get_hash()], \
                             "achtKlassenOrder": prob_c8.get_classes(), \
                             "dreiKlassenOrder": prob_c3.get_classes(), \
                             "zweiKlassenOrder": prob_c2.get_classes(), \
                             "achtKlassen": {"normal":   prob_c8('normal'), \
                                             "CM":       prob_c8('CLL + MBL'), \
                                             "MCL":      prob_c8('MCL'), \
                                             "PL":       prob_c8('PL'), \
                                             "LPL":      prob_c8('LPL'), \
                                             "MZL":      prob_c8('MZL'), \
                                             "FL":       prob_c8('FL'), \
                                             "HCL":      prob_c8('HCL')}, \
                             "dreiKlassen": {"normal":   prob_c3('normal'), \
                                             "a":        prob_c3('CLL + MBL + MCL + PL'), \
                                             "b":        prob_c3('LPL + MZL + FL + HCL')}, \
                             "zweiKlassen": {"normal":   prob_c2('normal'), \
                                             "pathogen": prob_c2('CLL + FL + HCL + LPL + MBL + MCL + MZL + PL')}}}
    with open(fname, 'w') as fhandle:
        json.dump(output, fhandle)
    print("Output written to '{0:}'".format(basename(fname)))
    return



if __name__ == '__main__':

    if len(sys.argv) != 5:
        print_usage()
        raise Exception("Error: Invalid aruments")
    job_id      = sys.argv[1]
    file_tube_a = sys.argv[2]
    file_tube_b = sys.argv[3]
    file_json   = sys.argv[4]

    data = dataset(file_tube_a, file_tube_b)
    data.load_fcs_files()
    data.compute_features()

    model_c8 = keras_model(model_dir, file_c8)
    model_c8.load_keras_model()
    prob_c8  = model_c8.predict(data)

    model_c3 = keras_model(model_dir, file_c3)
    model_c3.load_keras_model()
    prob_c3  = model_c3.predict(data)

    model_c2 = keras_model(model_dir, file_c2)
    model_c2.load_keras_model()
    prob_c2  = model_c2.predict(data)

    save_predictions(prob_c8, prob_c3, prob_c2, data, job_id, \
                     file_dir='', file_json=file_json)
