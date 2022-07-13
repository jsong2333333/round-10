import os
import numpy as np
import torch
import cv2
import torchvision
import json

PREPROCESS_FILEPATH = '/scratch/jialin/round-10/projects/object_detection/models/id-000000' 
# PREPROCESS_FILEPATH = '/scratch/data/TrojAI/round10test/id-000000'
TEST_OUTPUT = '/scratch/jialin/round-10/projects/object_detection/test_output'


def prepare_boxes(anns, image_id):
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for answer in anns:
            boxes.append(answer['bbox'])
            class_ids.append(answer['category_id'])

        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
        # convert [x,y,w,h] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))

    degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    if degenerate_boxes.any():
        boxes = boxes[degenerate_boxes == 0, :]
        class_ids = class_ids[degenerate_boxes == 0]
    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target


def get_class_id_to_name_dict():
    '''
    returns {coco_category_id [int] : coco_category_name [str]}
    '''
    cat_txt = open('/scratch/jialin/round-10/projects/object_detection/cat_label/coco-labels-paper.txt', 'r')
    data = cat_txt.read()
    cat_lst = data.split('\n')
    cat_txt.close()
    cat_dict = {str(k+1) : v for k, v in enumerate(cat_lst)}
    return cat_dict

def get_output_from_example_images(model, EXAMPLE_PATH, device):
    images_filepaths = [os.path.join(EXAMPLE_PATH, img) for img in os.listdir(EXAMPLE_PATH) if img.endswith('.jpg')]
    images_filepaths.sort()

    target = {}
    images, targets, ids = [], [], []
    for img_filepath in images_filepaths:
        image_id = os.path.basename(img_filepath)
        image_id = int(image_id.replace('.jpg',''))
        img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(img_filepath.replace('.jpg', '.json')) as json_file:
            annotations = json.load(json_file)

        target = prepare_boxes(annotations, image_id)

        # with torch.no_grad():
        img = torch.as_tensor(img).permute((2, 0, 1))
        img = torchvision.transforms.functional.convert_image_dtype(img, torch.float)

        images.append(img)
        targets.append(target)
        ids.append(image_id)
    
    # with torch.no_grad():
    images = list(image.to(device) for image in images)
    images = [img.requires_grad_() for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    outputs = model(images, targets)

    return {'image_id': ids, 'outputs': outputs, 'images': images}


def get_output_images(OUTPUT_DIR, EXAMPLE_PATH, example_outputs, number_of_detection=3):
    ids = example_outputs['image_id']
    outputs = example_outputs['outputs']
    if isinstance(outputs, tuple):
        outputs = outputs[1]

    example_images = [os.path.join(EXAMPLE_PATH, img) for img in os.listdir(EXAMPLE_PATH) if img.endswith('.jpg')]
    example_images.sort()

    for ind, out in enumerate(outputs):
        img = cv2.imread(example_images[ind], cv2.IMREAD_UNCHANGED)
        for i in range(number_of_detection):
            x1, y1, x2, y2 = out['boxes'][i]
            label = out['labels'][i]
            cat_name = cat_dict[str(label.item())]
            img = cv2.rectangle(img, (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item())), (255, 0, 0), 2)
            img = cv2.putText(img, cat_name, (int(x1.item()), int(y1.item()) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(OUTPUT_DIR+'/'+str(ids[ind])+'.jpg', img)


def save_images_with_gradient(OUTPUT_DIR, example_outputs):
    ids = example_outputs['image_id']
    outputs = example_outputs['outputs']
    images = example_outputs['images']
    outputs[0]['classification'].backward()
    for ind, img in enumerate(images):
        grad_img = img.grad.cpu().permute(1, 2, 0).numpy()
        grad_img -= grad_img.min()
        grad_img /= grad_img.max()
        grad_img *= 255
        grad_img = grad_img.astype(int)

        cv2.imwrite(OUTPUT_DIR+'/'+str(ids[ind])+'.jpg', grad_img)

PRESENT_OUTPUT = True
cat_dict = get_class_id_to_name_dict()
for model_ind in range(1):
    # path_ind = ''
    # POISONED_FLAG = False
    # if model_ind < 10:
    #     path_ind = '0' + str(model_ind)
    # else:
    #     path_ind = str(model_ind)
    path_ind = '09'
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
            get_output_images(poisoned_output_path, POISONED_EXAMPLE_PATH, poisoned_example_outputs, 5)

    # grad_output_path_clean = os.path.join(TEST_OUTPUT, path_ind, 'grad-example-outputs', 'clean')
    # if not os.path.exists(grad_output_path_clean):
    #     os.makedirs(grad_output_path_clean)
    # save_images_with_gradient(grad_output_path_clean, clean_example_outputs)
    # if POISONED_FLAG:
    #     grad_output_path_poisoned = os.path.join(TEST_OUTPUT, path_ind, 'grad-example-outputs', 'poisoned')
    #     if not os.path.exists(grad_output_path_poisoned):
    #         os.makedirs(grad_output_path_poisoned)
    #     save_images_with_gradient(grad_output_path_poisoned, poisoned_example_outputs)