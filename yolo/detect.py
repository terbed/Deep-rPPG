from yolo.utils import *
import torchvision.transforms as transforms
import torch
tr = torch


def babybox(model, img):
    """
        Returns bounding box of baby: x_1, y_1, x_2, y_2
    """

    # -----------------
    # parameters
    # -----------------
    conf_thres = 0.8
    nms_thres = 0.4
    classes = ["baby", "nurse_hand", "parent_hand", "nursing_bottle"]

    # ----------------------------
    # Construct input for network
    # ----------------------------
    inp = transforms.ToTensor()(img)
    inp = F.interpolate(inp.unsqueeze(0), size=416, mode="nearest")
    inp = inp.type(tr.FloatTensor)

    # -----------------
    # Run network
    # -----------------
    with tr.set_grad_enabled(False):
        outputs = model(inp)
        outputs = non_max_suppression(outputs, conf_thres, nms_thres)[0]

    # --------------------------
    # Extract baby bounding box
    # --------------------------
    # rescale boxes for image size
    detections = rescale_boxes(outputs, 416, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)

    if detections is not None:
        x_1 = y_1 = x_2 = y_2 = 0
        prev_conf = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            # Crop baby
            if classes[int(cls_pred)] == 'baby':
                if prev_conf < cls_conf:
                    prev_conf = cls_conf
                    x_1, y_1, x_2, y_2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())

        return x_1, y_1, x_2, y_2

    else:
        print('NO OBJECT WAS FOUND!!!')
        return None
