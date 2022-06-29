import cv2

from task import TaskParser, ModelInit


def predict(task, image, model_initial):
    # 0. create task
    print("{0}\tCreate Task\t{0}".format("-" * 30))
    task_parser = TaskParser(task)

    # 1. create model
    print("{0}\tCreate Net\t{0}".format("-" * 30))
    model = task_parser.create_model(model_initial, skip_detect_layer=False)

    # 2. create detector
    print("{0}\tCreate Detector\t{0}".format("-" * 30))
    detector = task_parser.create_detector(model, object_thresh=0.5, nms_thresh=0.5)

    # 3. run detection
    print("{0}\tRun Detection\t{0}".format("-" * 30))
    boxes, labels, probs = detector.detect_from_file(image, cls_threshold=0.5)
    print(boxes, labels, probs)

    # 4. draw result
    image = cv2.imread(image)
    for box, label, probs in zip(boxes, labels, probs):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(image, pt1, pt2, (255, 255, 0), 1)
        class_name = task_parser.model_cfg.labels[label]
        cv2.putText(image, "{0} {1:.2f}%".format(class_name, probs * 100), pt1, cv2.FONT_ITALIC, 1, (0, 0, 255), 1,
                    lineType=cv2.LINE_AA)

    cv2.imshow("Result", image)
    cv2.waitKey(5000)


if __name__ == '__main__':
    # predict(r'config\coco.json', r"F:\PASCALVOC\VOC2007\JPEGImages\000012.jpg", ModelInit.original)

    # predict(r'config\pascalVoc.json', r"F:\PASCALVOC\VOC2007_Val\JPEGImages\006926.jpg",
    #         ModelInit.pretrain)

    # "F:\Raccoon\images\raccoon-170.jpg" 199, 193, 184, 170
    predict(r'config\raccoon.json', r"F:\Raccoon\images\raccoon-2.jpg",
            ModelInit.pretrain)
