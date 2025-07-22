import os
import re
import xml.etree.ElementTree as ET
import json


# Creates the <body> tag, that encapsulates the whole file
def insert_body_tag(filename):
    with open(filename, 'r+', encoding='utf8') as f:
        content = f.read()
        if not content.split('>')[0] == '<body':
            f.seek(0, 0)
            f.write('<body>\n' + content.replace('&', 'and') + '\n</body>')
        f.close()


# Makes the first letter of the name upper case
def make_first_cap(name):
    return name[0].upper() + name[1:]


# Returns the list of attributes taking as input the values written as 'Prem1|Prem2'
def make_value_list(values):
    result = []
    for val in values.split('|'):
        result.append(val)
    return result


# Checks if an XML element has sub-elements
def has_children(element):
    return len(list(element))


# Main conversion from XML to JSON
# plain_text_presence is a boolean variable that indicates whether we want the plain text in the JSON file or not
def convert_to_json_old(xml_files_path, base_id=0, plain_text_presence=True, language='english', change_name=True):
    if not os.path.isdir('./demosthenes_dataset_json'):
        os.mkdir('./demosthenes_dataset_json')

    arr = os.listdir(xml_files_path)

    count = 0
    for sentence in arr:
        print(sentence)

        to_open = xml_files_path + '/' + sentence

        # Inserting the tag <body> that encapsulates the whole file
        insert_body_tag(to_open)

        # DOCUMENT
        # Name
        file_name = sentence.split('.')[0]

        # ID
        file_id = str(base_id + count)
        print(str(count) + '\t' + file_id)
        count += 1

        tree = ET.parse(to_open)
        root = tree.getroot()

        # Plain Text
        plain_text = ''
        for i in root.itertext():
            plain_text = plain_text + i
        plain_text = re.sub(r'\\([^rnt])', r'/\1', plain_text)

        # ANNOTATIONS
        annotations = []
        for child in root.iter():
            if not child.tag == 'body':
                tag = {}
                # Document
                tag['document'] = file_id
                # Name
                tag['name'] = child.tag
                # _id
                if 'ID' in child.attrib.keys():
                    tag['_id'] = child.attrib['ID']

                # Attributes
                attributes = {}
                for attr in child.attrib:
                    if '|' in child.attrib[attr] and (not child.attrib[attr] == 'L|F'):
                        attributes[attr] = make_value_list(child.attrib[attr])
                    else:
                        attributes[attr] = child.attrib[attr]
                tag['attributes'] = attributes
                # Position
                tag_text = ''
                for i in child.itertext():
                    tag_text = tag_text + i
                tag['start'] = plain_text.find(tag_text)
                tag['end'] = tag['start'] + len(tag_text)

                annotations.append(tag)

        # Creating the internal file name
        if change_name:
            internal_file_name = language + '_' + str(file_id)
        else:
            internal_file_name = file_name

        # Writing the JSON file
        json_name = './demosthenes_dataset_json/' + internal_file_name + '.json'
        json_file = open(json_name, 'w', encoding='utf8')
        # Document
        json_file.write('{"document":{"_id":"')
        json_file.write(file_id)
        json_file.write('","name":"')
        json_file.write(file_name)
        if plain_text_presence:
            json_file.write('","plainText":"')
            json_file.write(plain_text.replace('"', "'").replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r'))
        json_file.write('"},')
        # Types
        # json_file.write(types)
        # Annotations
        json_file.write('"annotations":[')
        first_annotation = True
        for annotation in annotations:
            if not first_annotation:
                json_file.write(',')
            json_file.write(str(annotation).replace("'", '"'))
            if first_annotation:
                first_annotation = False
        json_file.write(']}')


# convert_to_json_old('./demosthenes_dataset', base_id=1000, plain_text_presence=True, language='english')

# Revised convert_to_json using json module for robust output
def convert_to_json(xml_files_path, base_id=0, plain_text_presence=True, language='english', change_name=True):
    out_dir = os.path.join(os.path.dirname(__file__), 'demosthenes_dataset_json')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(xml_files_path) if f.lower().endswith('.xml'))
    for count, filename in enumerate(files):
        file_id = str(base_id + count)
        file_stem = os.path.splitext(filename)[0]

        path = os.path.join(xml_files_path, filename)
        insert_body_tag(path)

        tree = ET.parse(path)
        root = tree.getroot()

        plain_text = ''.join(root.itertext())
        plain_text = re.sub(r'\\([^rnt])', r'/\\1', plain_text)

        annotations = []
        for child in root.iter():
            if child.tag == 'body':
                continue
            tag_text = ''.join(child.itertext())
            start = plain_text.find(tag_text)
            end = start + len(tag_text)
            attrs = {}
            for key, val in child.attrib.items():
                if '|' in val and val != 'L|F':
                    attrs[key] = make_value_list(val)
                else:
                    attrs[key] = val
            ann = {
                'document': file_id,
                'name': child.tag,
                '_id': child.attrib.get('ID'),
                'attributes': attrs,
                'start': start,
                'end': end
            }
            annotations.append(ann)

        doc = {'_id': file_id, 'name': file_stem}
        if plain_text_presence:
            doc['plainText'] = plain_text
        data = {'document': doc, 'annotations': annotations}

        out_name = f"{language}_{file_id}" if change_name else file_stem
        out_path = os.path.join(out_dir, out_name + '.json')
        with open(out_path, 'w', encoding='utf8') as jf:
            json.dump(data, jf, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    convert_to_json('./demosthenes_dataset', base_id=1000, plain_text_presence=True, language='english')
