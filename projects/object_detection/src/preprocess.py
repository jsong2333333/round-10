import os
import numpy as np
import torch
import cv2
import torchvision
import json

PREPROCESS_FILEPATH = '/scratch/jialin/round-10/projects/object_detection/models/id-000000' 
TEST_OUTPUT = '/scratch/jialin/round-10/projects/object_detection/test_output'

def convert():
    cat_txt = open('/scratch/jialin/round-10/projects/object_detection/cat_label/coco-labels-paper.txt', 'r')
    data = cat_txt.read()
    cat_lst = data.split('\n')
    cat_txt.close()
    cat_dict = {str(k+1) : v for k, v in enumerate(cat_lst)}
    return cat_dict

def get_output_from_example_images(model, EXAMPLE_PATH, device):
    example_images = [os.path.join(EXAMPLE_PATH, img) for img in os.listdir(EXAMPLE_PATH) if img.endswith('.jpg')]
    example_images.sort()

    target = {}
    images, targets, ids = [], [], []
    for img in example_images:
        image_id = os.path.basename(img)
        image_id = int(image_id.replace('.jpg',''))
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            img = torch.as_tensor(img).permute((2, 0, 1))
            img = torchvision.transforms.functional.convert_image_dtype(img, torch.float)
            target['boxes'] = torch.as_tensor(np.zeros((0, 4)))
            target['labels'] = torch.as_tensor(np.zeros((0))).type(torch.int64)
            target['image_id'] = torch.as_tensor(image_id)
        images.append(img)
        targets.append(target)
        ids.append(image_id)
    with torch.no_grad():
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images, targets)
        if isinstance(outputs, tuple):
            outputs = outputs[1]
    return {'image_id': ids, 'outputs': outputs}


def get_output_images(OUTPUT_DIR, EXAMPLE_PATH, example_outputs, number_of_detection=3):
    ids = example_outputs['image_id']
    outputs = example_outputs['outputs']

    example_images = [os.path.join(EXAMPLE_PATH, img) for img in os.listdir(EXAMPLE_PATH) if img.endswith('.jpg')]
    example_images.sort()

    for ind, out in enumerate(outputs):
    # print(out)
    # print(out['boxes'].size())
    # print(out['scores'])
    # print(out['labels'])
        img = cv2.imread(example_images[ind], cv2.IMREAD_UNCHANGED)
        for i in range(number_of_detection):
            x1, y1, x2, y2 = out['boxes'][i]
            label = out['labels'][i]
            cat_name = cat_dict[str(label.item())]
            # print(box)
            # print(label)
            img = cv2.rectangle(img, (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item())), (255, 0, 0), 2)
            img = cv2.putText(img, cat_name, (int(x1.item()), int(y1.item()) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(OUTPUT_DIR+'/'+str(ids[ind])+'.jpg', img)

PRESENT_OUTPUT = True
cat_dict = convert()
for model_ind in range(5):
    path_ind = ''
    POISONED_FLAG = False
    if model_ind < 10:
        path_ind = '0' + str(model_ind)
    else:
        path_ind = str(model_ind)
    PATH_WITH_ID = PREPROCESS_FILEPATH + path_ind
    MODEL_PATH = os.path.join(PATH_WITH_ID, 'model.pt')
    CLEAN_EXAMPLE_PATH = os.path.join(PATH_WITH_ID,'clean-example-data')
    POISONED_EXAMPLE_PATH = os.path.join(PATH_WITH_ID,'poisoned-example-data')
    if os.path.isdir(POISONED_EXAMPLE_PATH):
        POISONED_FLAG = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    test_model = torch.load(MODEL_PATH)
    test_model.to(device)
    test_model.eval()

    # get outputs from example images
    clean_example_outputs = get_output_from_example_images(test_model, CLEAN_EXAMPLE_PATH, device)
    if POISONED_FLAG:
        poisoned_example_outputs = get_output_from_example_images(test_model, POISONED_EXAMPLE_PATH, device)
    
    if PRESENT_OUTPUT:
        clean_output_path = os.path.join(TEST_OUTPUT, path_ind, 'clean-example-outputs')
        if not os.path.exists(clean_output_path):
            os.makedirs(clean_output_path)
        get_output_images(clean_output_path, CLEAN_EXAMPLE_PATH, clean_example_outputs, 1)
        if POISONED_FLAG:
            poisoned_output_path = os.path.join(TEST_OUTPUT, path_ind, 'poisoned-example-outputs')
            if not os.path.exists(poisoned_output_path):
                os.makedirs(poisoned_output_path)
            get_output_images(poisoned_output_path, POISONED_EXAMPLE_PATH, poisoned_example_outputs, 1)
    