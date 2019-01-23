from flowcat import utils, classify, configuration
from flowcat.dataset import case
import fcsparser
import toml
from pprint import PrettyPrinter
p = PrettyPrinter()
import os
from keras.models import load_model
import tempfile

def load_toml(path):
    with open(str(path), "r") as f:
        data = toml.load(f)
    print(f'-------------------- loading {path}')
    p.pprint(data)
    return data


def fcs_columns_matches_tube(columns, expected):
    #print(f'is {columns} a superset of {expected}?')
    for e in expected:
        if e not in columns:
            #print(f'no match, due to column {e}!')
            return False
    #print(f'yes, all columns match!')
    return True

def detect_tube(filename, som1, som2):
    meta, data = fcsparser.parse(filename, encoding="latin-1")
    columns = data.columns
    for tube, som in [(1, som1), (2, som2)]:
        if fcs_columns_matches_tube(columns, som['channels']):
            return tube
    raise Exception(
        f'unable to detect tube 1 or 2 for file {filename} with channels {columns}.')
            


def inference(filenames, modeldir):
    som1 = load_toml(f'{modeldir}/dataset/SOM_1.toml')
    som2 = load_toml(f'{modeldir}/dataset/SOM_2.toml')

    filepaths = []
    for filename in filenames:
        tube = detect_tube(filename, som1, som2)
        filepaths.append({ "fcs": { "path": filename },
                           "date": "2019-01-01",
                           "tube": tube })

    print('filepaths:')
    p.pprint(filepaths)

    tdict = {
        "id": '87432894739472384932749823',
        "date": "2019-01-01",
        "filepaths": filepaths,
    }
    cs = case.Case(tdict, path='.')

    if True: #with somnodes
        model, transform, dataseqGroups, filters = \
            classify.load_model(modeldir, also_return_somnodes=True)
        print('transform', transform)
        transformed, allsomnodes = transform([cs])
        somnodes = allsomnodes[0]
    else:
        model, transform, dataseqGroups, filters = \
            classify.load_model(modeldir)
        print('transform', transform)
        transformed = transform([cs])

    groups = load_toml(f'{modeldir}/dataset/config.toml')['groups']
    print(groups)
    output = model.predict(transformed) #(a, b)
    print(output)
    predictions = {group: float(p) for group,p in zip(groups, output[0])}

    #return predictions, du

    predicted_class_index = output.argmax()
    predicted_class = groups[predicted_class_index]

    print(predicted_class_index, predicted_class)


    import flowcat.visual.plotting as plo

    # gradients = plo.calc_saliency(dataset, case, c_model, classes=classes)

    layer_idx = -1

    # modify model for saliency usage
    import keras
    from vis.visualization import visualize_saliency
    from vis.utils import utils

    model.layers[layer_idx].activation = keras.activations.linear
    model = utils.apply_modifications(model)

    input_indices = [0, 1]
    grad_groups = [predicted_class]
    print('grad_groups', grad_groups)

    ts = [transformed[0][0], transformed[1][0]]
    gradients = [visualize_saliency(model,
                                    layer_idx,
                                    groups.index(group),
                                    seed_input=ts,
                                    input_indices=input_indices,
                                    maximization=False)
                 for group in grad_groups]


    # wtf?
    # all_fcsdata = [ case.get_tube(tube).data.data
    #                 for tube in [1, 2]]

    du = {'filepaths': filepaths,
          # 'all_fcsdata': all_fcsdata,
          'groups': groups,
          'predictions': predictions,
          'predicted_class_index': predicted_class_index,
          'predicted_class': predicted_class,
          'somnodes': somnodes,
          'grad_groups': grad_groups,
          'gradients': gradients,
          'ts': ts,
          'som_channels': [som1['channels'],
                           som2['channels']]}
    

    ### # regroup gradients into 2D array (nodes,channels) for tube1 and tube2
    ### gradients = [[grad[0].reshape(1156, 11) for grad in gradients],
    ###              [grad[1].reshape(1156, 11) for grad in gradients]]

    # def save_function(idx, scatterplot):
    #     filename = f"scatter-{groups[idx]}-tube-{tube}-scatterplots.png"
    #     plo.save_figure(scatterplot, filename)
    #     print('wrote scatterplot', filename)

    # grads = gradients[tube - 1]
    # channels = [som1, som2][tube-1]['channels']
    # plo.plot_tube(
    #     cs, tube, grads, groups,
    #     somnodemapping=somnodes[tube],
    #     channels=channels,
    #     save=save_function)

    return predictions, du, predicted_class


filenames = [
    "some-facs-files/CLL/81e00e637713c5a4819253e6d5b68defa7735d5e-4 CLL 9F 01 N07 001.LMD",
    "some-facs-files/CLL/81e00e637713c5a4819253e6d5b68defa7735d5e-4 CLL 9F 02 N09 001.LMD"

    # 'some-facs-files/AML/81d8a3d14f1a4e186ecacaf4d22a8758d8b2b8b2-3 CLL 9F 01 001.LMD',
    # 'some-facs-files/AML/81d8a3d14f1a4e186ecacaf4d22a8758d8b2b8b2-3 CLL 9F 02 001.LMD'

    # 'some-facs-files/LPL/814ae549ff444396017ace503d4314cfdcb7deda-2 CLL 9F 01 N07 001.LMD',
    # 'some-facs-files/LPL/814ae549ff444396017ace503d4314cfdcb7deda-2 CLL 9F 02 N17 001.LMD'
    
    # 'some-facs-files/FL/81ade3605a67eac5bc798d63e96f9bf70286664d-2 CLL 9F 02 N03 001.LMD',
    # 'some-facs-files/FL/81ade3605a67eac5bc798d63e96f9bf70286664d-2 CLL 9F 01 N06 001.LMD',
    #'some-facs-files/FL/81ade3605a67eac5bc798d63e96f9bf70286664d-2 CLL 9F 03 N06 001.LMD',
]

def format_predictions(anfrageId, fileA, fileB, modelName,
                       predictions,
                       predicted_class,
                       scatterPlotImageUrl):
    da = sum(predictions[k] for k in ['CM', 'MCL', 'PL'])
    db = sum(predictions[k] for k in ['LPL', 'MZL', 'FL', 'HCL'])

    return {
        'anfrageId': anfrageId,
        'files': [fileA, fileB],
        'prediction': {
            'model': modelName,
            'achtKlassenOrder': [
                'normal', 'CM', 'MCL', 'PL', 'LPL', 'MZL', 'FL', 'HCL'],
            'achtKlassen': predictions,

            'dreiKlassenOrder': ['normal', 'a', 'b'],
            'dreiKlassen': {
                'normal': predictions['normal'],
                'a': da,
                'b': db,
            },

            'zweiKlassenOrder': ['normal', 'pathogen'],
            'zweiKlassen': {
                'normal': predictions['normal'],
                'pathogen': da+db
            },
            'scatterPlots': {
                predicted_class: {
                    'imageURL': scatterPlotImageUrl
                }
            }
        }
    }



if __name__ == "__main__":
    modeldir = './testjan19'
    import sys
    args = sys.argv[1:]
    if len(args) == 1 and args[0] == 'test':
        anfrageId, a, b, output_filename = '777', filenames[0], filenames[1], '-'
    else:
        if len(args) != 4:
            print('expected arguments: anfrageId tube1-fcs-file tube2-fcs-file output-file-name')
            exit(1)
        anfrageId, a, b, output_filename = args
    predictions, du, predicted_class = inference([a, b], modeldir)
    p.pprint(predictions)

    anfrageDir = os.path.dirname(a)

    scatterPlotImageUrl = f'/anfrage/{anfrageId}/scatter-{predicted_class}.png'

    result = format_predictions(anfrageId, a, b, modeldir, predictions,
                                predicted_class, scatterPlotImageUrl)

    import json
    if output_filename == '-':
        print(json.dumps(result))
    else:
        with open(output_filename, 'w') as out:
            json.dump(result, out)

    import pickle
    filename = f'{anfrageDir}/du.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(du, f)
        print(f'wrote {filename}')
    
    os.system(f'python3 ./visualize.py "{filename}"')


# saliency - https://arxiv.org/pdf/1312.6034.pdf

# todo:
#   * welche prob ist welche klasse?
#   * ich muss bei den files rausfinden welche tube sie sind.
#       oder wir fordern einfach vom uploader dass er sie richtig eingibt.
#       Ich koennte mir die Liste an markern anschauen...
#   * auf aws zum laufen bringen
#   * saliency


''' installing:

sudo yum install git
sudo pip-3.6 install -r requirements.txt 
sudo pip-3.6 uninstall numpy
sudo pip-3.6 install numpy==1.15

cd flowCat/
python3 inference.py test

'''
