import torchvision.datasets as dset
import torchvision.transforms as tfs
import json

path2data = '/scratch/jialin/round-10/projects/object_detection/coco_clean_dataset/train'
path2json = '/scratch/jialin/round-10/projects/object_detection/cat_label/_annotations.coco.json'
path2changedjson = '/scratch/jialin/round-10/projects/object_detection/cat_label/_annotations_changed.coco.json'
path2label = '/scratch/jialin/round-10/projects/object_detection/cat_label/coco-labels-paper.txt'

def get_class_name_to_id_dict(path2label):
    '''
    returns {coco_category_name [str]: coco_category_id [int]}
    '''
    cat_txt = open(path2label, 'r')
    data = cat_txt.read()
    cat_lst = data.split('\n')
    cat_txt.close()
    cat_dict = {v : k+1 for k, v in enumerate(cat_lst)}
    return cat_dict

def filter_by_id(ids, max_example=50):
    '''
    input: List[int] - list of category ids from coco dataset, int - number of examples as output (may be fewer than max_example if there are no matching images)
    output: List[tuple] - list of tuple with tensor[0] - tensor of the image, tensor[1] - annotations
    '''
    ret = []
    for img, anns in coco_train:
        for ann in anns:
            if ann['category_id'] in ids and len(ret) < max_example:
                ret.append((img, anns))
            if len(ret) >= max_example:
                return ret
    return ret

def update_json():
    cat_dict = get_class_name_to_id_dict(path2label)
    with open(path2json, 'r') as json_file:
        unchanged_anns = json.load(json_file)
    dataset_cat_id_to_example_cat_id_dict = {'no-match':[]}
    for cat in unchanged_anns['categories']:
        if cat['name'] in cat_dict:
            dataset_cat_id_to_example_cat_id_dict[cat['id']] = cat_dict[cat['name']]
        else:
            dataset_cat_id_to_example_cat_id_dict['no-match'].append(cat['name'])
    for cat in unchanged_anns['categories']:
        if cat['id'] == 0:
            # coco-object, no match
            continue
        dataset_cat_id = cat['id']
        cat['id'] = dataset_cat_id_to_example_cat_id_dict[dataset_cat_id]
    for ann in unchanged_anns['annotations']:
        dataset_cat_id = ann['category_id']
        ann['category_id'] = dataset_cat_id_to_example_cat_id_dict[dataset_cat_id]

    with open(path2changedjson, 'w') as json_file:
        json.dump(unchanged_anns, json_file, indent=4)

coco_train = dset.CocoDetection(root=path2data, annFile=path2changedjson, transform=tfs.ToTensor())