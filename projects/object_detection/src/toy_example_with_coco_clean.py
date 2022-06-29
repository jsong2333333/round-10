import os
import numpy as np
import torch
import cv2
import torchvision
import json
import load_dataset
# import PIL.Image
import polygon_trigger


FILEPATH = '/scratch/jialin/round-10/projects/object_detection/models/id-00000018' 
TRIAL_OUTPUT = '/scratch/jialin/round-10/projects/object_detection/test_output/toy_clean/unconstrained'
TOY_OUTPUT = '/scratch/jialin/round-10/projects/object_detection/test_output/toy_clean'

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


def save_grad_and_output_images(img_output, grad_output, ind):
    def process_image(output_data_from_model):
        img = output_data_from_model.cpu().detach().permute(1, 2, 0).numpy()
        img -= img.min()
        img /= img.max()
        img *= 255
        img = img.astype(int)
        return img

    grad_image = process_image(grad_output)
    image = process_image(img_output)
    grad_output_filepath = os.path.join(TRIAL_OUTPUT, 'grad')
    img_output_filepath = os.path.join(TRIAL_OUTPUT, 'image_output')
    cv2.imwrite(grad_output_filepath+'/'+str(ind)+'.jpg', grad_image)
    cv2.imwrite(img_output_filepath+'/'+str(ind)+'.jpg', image)


cat_dict = get_class_id_to_name_dict()

PATH_WITH_ID = FILEPATH
MODEL_PATH = os.path.join(PATH_WITH_ID, 'model.pt')
# TRIGGER_PATH = os.path.join(PATH_WITH_ID, 'trigger_0.png')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load coco dataset - load a list of tuples [(tensor of image, annotations)]
clean_images = load_dataset.filter_by_id([19])
# trigger = np.ones(shape=(50, 50, 3))
# trigger = cv2.imread(TRIGGER_PATH, cv2.IMREAD_UNCHANGED)
# trigger = cv2.cvtColor(trigger, cv2.COLOR_BGR2RGB)
# trigger = cv2.resize(trigger, (50, 50))
# trigger[trigger > 0] = 1
# trigger = torch.as_tensor(trigger).permute((2, 0, 1))
# tri_w, tri_h = trigger.shape[1], trigger.shape[2]

# load image and target
img, tgt = clean_images[10]
# img[:,10:10+tri_w, 10:10+tri_h] = trigger
img = torchvision.transforms.functional.convert_image_dtype(img, torch.float)
img = img.to(device)
img = img.requires_grad_()
tgt = prepare_boxes(tgt, tgt[0]['image_id'])
tgt = {k:v.to(device) for k, v in tgt.items()}

# load model
test_model = torch.load(MODEL_PATH)
test_model.to(device)
test_model.eval()

# initialize
delta = torch.zeros_like(img, dtype=torch.float, requires_grad=True).to(device)
# p_trigger = polygon_trigger.PolygonTrigger()
# p_trigger

total_steps, EPSILON = 20, 0.2
LOSS, DELTA_NORM = [], []

for step in range(1, total_steps+1):
    img = img + delta
    img = img - img.min()
    img = img / img.max()
    img = img.requires_grad_()
    if torch.all(img >= 0.) and torch.all(img <= 1.):
        print('step: ', step)
        img.retain_grad()
        output = test_model([img], [tgt])
        LOSS.append(output[0]['classification'])
        output[0]['classification'].backward()

        img_grad = img.grad
        save_grad_and_output_images(img, img_grad, step)

        delta = delta + img_grad*EPSILON*2.5/step/torch.linalg.norm(torch.linalg.norm(img_grad, dim=1, ord=2), ord=2)
        DELTA_NORM.append(torch.linalg.norm(torch.linalg.norm(delta, dim=1, ord=2), ord=2))

        test_model.zero_grad()
        delta = delta.detach()
        img = img.detach()
    else:
        break

print('LOSS: ', LOSS)
print('DELTA_NORM: ', DELTA_NORM)