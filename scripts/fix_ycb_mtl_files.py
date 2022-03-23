"""
Context:
Loading object models from YCB dataset, using google16k/textured.obj.

Open3D has issues loading the textures of these models, which is due to a white space in the `textured.mtl` file.
This white space comes right after the filename of the texture image, and Open3D does not recognise "png " as a valid
image file type.
This script goes through the mtl files and removes the white space, so that Open3D can properly load the textures.
"""
import os
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-y', '--ycb_base_path', type=str, default='/home/rudorfem/datasets/ycb-objects',
                        help='path ycb objects folder')
    parser.add_argument('-m', '--model_type', type=str, default='google_16k', help='model type, i.e. sub-folder name')
    return parser.parse_args()


def remove_whitespace(material_file):
    with open(material_file, 'r') as infile:
        data = infile.read()

    # check number of occurrences
    n = len(re.findall(r'\.png ', data))
    if n == 0:
        # nothing to be done here
        return

    if n > 1:
        print(f'{n} occurrences of ".png " found in {material_file}, please check manually.')
        return

    data = data.replace('.png ', '.png')
    with open(material_file, 'w') as outfile:
        outfile.write(data)


def traverse_ycb_objects(ycb_base_path, model_type):
    object_names = sorted([x for x in os.listdir(ycb_base_path) if os.path.isdir(os.path.join(ycb_base_path, x))])
    for name in object_names:
        print(name)
        material_file = os.path.join(ycb_base_path, name, model_type, 'textured.mtl')
        if not os.path.isfile(material_file):
            print(f'WARNING: found shape {name} but it does not have a {model_type}/textured.mtl file. Skipping.')
            continue
        remove_whitespace(material_file)


if __name__ == '__main__':
    opt = parse_args()
    traverse_ycb_objects(opt.ycb_base_path, opt.model_type)
