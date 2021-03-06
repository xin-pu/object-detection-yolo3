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
    detector = task_parser.create_detector(model,
                                           object_thresh=0.9,
                                           class_thresh=0.8,  # 使用交叉熵，
                                           nms_thresh=0.5)

    # 3. run detection
    print("{0}\tRun Detection\t{0}".format("-" * 30))
    boxes = detector.detect_from_file(image)
    if len(boxes) == 0:
        print("don't find object.")
        return

        # 4. draw result
    image = cv2.imread(image)
    for box in boxes:
        min_max = box.as_min_max_rect()
        pt1 = (int(min_max[0]), int(min_max[1]))
        pt2 = (int(min_max[2]), int(min_max[3]))
        cv2.rectangle(image, pt1, pt2, (255, 255, 0), 1)
        class_name = task_parser.model_cfg.labels[int(box.label_index)]
        cv2.putText(image, "{0} {1:.2f}%".format(class_name, box.object_ness * 100), pt1, cv2.FONT_ITALIC, 1,
                    (0, 0, 255), 1,
                    lineType=cv2.LINE_AA)

    cv2.imshow("Result", image)
    cv2.waitKey(5000)


if __name__ == '__main__':
    # predict(r'config\pascalVoc.json', r"F:\PASCALVOC\VOC2007_Val\JPEGImages\000178.jpg",
    #         ModelInit.pretrain)

    # "F:\Raccoon\images\raccoon-170.jpg" 199, 193, 184, 170, 148, 156
    predict(r'config\raccoon.json', r"F:\Raccoon\images\raccoon-148.jpg",
            ModelInit.pretrain)

    # predict(r'config\module.json', r"F:\Module Object Detection\Val\images\000004.png",
    #         ModelInit.pretrain)
