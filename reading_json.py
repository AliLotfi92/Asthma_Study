import json
import os
import numpy as np
features = ['city']

files_name = os.listdir('synthetic_patients')
print('number of total samples:', len(files_name))
dest_path = 'synthetic_patients'


class generate_dataset():
    def __init__(self):
        self.dataset = []

    def add_feature(self, feature_name):
        self.feature_extraction = []
        self.feature_extraction.append(feature_name)

    def add_value_to_feature(self, true_value):
        self.feature_extraction.append(true_value)

    def dataset_asemble(self):
        self.dataset.append(self.feature_extraction)

    def get_dataset(self):
        return self.dataset

def get_all_features(dict):
    return dict.keys()

gen_dataset = generate_dataset()

def reading_dataset(feature_name):
    gen_dataset.add_feature(feature_name)
    for name in files_name:
        files_directory = os.path.join(dest_path, name)
        with open(files_directory) as f:
            data = json.load(f)
            all_feature_name = get_all_features(data)
            #print(all_feature_name)
            gen_dataset.add_value_to_feature(data[feature_name])


def reading_complex_dataset(feature_name, sub_feature_name):
    gen_dataset.add_feature(sub_feature_name)

    for name in files_name:
        files_directory = os.path.join(dest_path, name)

        with open(files_directory) as f:
            data = json.load(f)
            #print(get_all_features(data))
            complex_data = data[feature_name]
            value_list = []

            for i in range(len(complex_data)):
                B = complex_data[i]
                if B['description'] == sub_feature_name:
                    value_list.append(float(B['value']))

            gen_dataset.add_value_to_feature(np.average(value_list))



def reading_complex_dataset_str(feature_name, sub_feature_name):
    gen_dataset.add_feature(sub_feature_name)

    for name in files_name:
        files_directory = os.path.join(dest_path, name)

        with open(files_directory) as f:
            data = json.load(f)
            #print(get_all_features(data))
            complex_data = data[feature_name]
            value_list = []

            for i in range(len(complex_data)):
                B = complex_data[i]
                if B['description'] == sub_feature_name:
                    value_list.append(float(1.0) if B['value'] == 'Former smoker' else float(0.0))


            gen_dataset.add_value_to_feature(np.average(value_list))

def reading_complex_dataset_allergies(feature_name):
    gen_dataset.add_feature(feature_name)

    for name in files_name:
        files_directory = os.path.join(dest_path, name)

        with open(files_directory) as f:
            data = json.load(f)
            complex_data = data[feature_name]
            value = 0.0 if not complex_data else 1.0
            gen_dataset.add_value_to_feature(value)

reading_complex_dataset_allergies('allergies')
gen_dataset.dataset_asemble()

reading_complex_dataset_str('observations', 'Tobacco smoking status NHIS')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Pain severity - 0-10 verbal numeric rating [Score] - Reported')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Body Weight')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Systolic Blood Pressure')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Heart rate')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Diastolic Blood Pressure')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Body Height')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Respiratory rate')
gen_dataset.dataset_asemble()

reading_complex_dataset('observations', 'Body Mass Index')
gen_dataset.dataset_asemble()


reading_dataset('deathdate')
gen_dataset.dataset_asemble()

reading_dataset('marital')
gen_dataset.dataset_asemble()

reading_dataset('birthdate')
gen_dataset.dataset_asemble()

reading_dataset('ethnicity')
gen_dataset.dataset_asemble()

reading_dataset('gender')
gen_dataset.dataset_asemble()

reading_dataset('race')
gen_dataset.dataset_asemble()

dataset = np.asarray(gen_dataset.get_dataset())

np.save('here', dataset)
exit()





       # A = data['observations']
       # for i in range(len(A)):
       #     B = A[i]
            #exit()
            #print(get_all_features(B))
            #print(B['description'])
       #     if B['description'] == 'Respiratory rate':
       #         print(B['value'])
