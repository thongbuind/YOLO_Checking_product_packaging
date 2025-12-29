import json
import threading

from camera.CamThread import CamThread
from camera.CamInfo import SlotInfo, CamInfo

def bootstrap(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    image_size = config["image_size"]
    classes = config["classes"]

    slot_expected_items = {
        i: config[f"slot_{i}"] for i in range(1, 11)
    }

    slots = {
        i: SlotInfo(expected_item=slot_expected_items[i])
        for i in range(1, 11)
    }

    slots_list_for_cam_12 = {i: slots[i] for i in range(1, 6)}
    slots_list_for_cam_34 = {i: slots[i] for i in range(6, 11)}

    cameras = {
        "cam_1": CamInfo(slot_will_be_checked=[1, 2, 3], slots_list=slots_list_for_cam_12),
        "cam_2": CamInfo(slot_will_be_checked=[1, 2, 3, 4, 5], slots_list=slots_list_for_cam_12),
        "cam_3": CamInfo(slot_will_be_checked=[6, 7, 8], slots_list=slots_list_for_cam_34),
        "cam_4": CamInfo(slot_will_be_checked=[6, 7, 8, 9, 10], slots_list=slots_list_for_cam_34)
    }

    cam_configs = [
        ("cam_1", 1),
        ("cam_2", 1),
        ("cam_3", 1),
        ("cam_4", 1),
    ]

    cam_threads = {
        name: CamThread(name, source, mode="webcam")
        for name, source in cam_configs
    }

    device = 'mps:0'

    return image_size, classes, cameras, cam_threads, device
