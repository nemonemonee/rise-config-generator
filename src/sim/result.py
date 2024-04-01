import numpy as np
from lxml import etree


def get_center_of_mass(result_path):
    with open(result_path, "r") as file:
        result = file.read()
    """Note: Unit is voxels"""
    doc = etree.fromstring(bytes(result, encoding="utf-8"))
    start_com = np.array(
        [
            float(doc.xpath("/report/detail/start_center_of_mass/x")[0].text),
            float(doc.xpath("/report/detail/start_center_of_mass/y")[0].text),
            float(doc.xpath("/report/detail/start_center_of_mass/z")[0].text),
        ]
    )
    end_com = np.array(
        [
            float(doc.xpath("/report/detail/end_center_of_mass/x")[0].text),
            float(doc.xpath("/report/detail/end_center_of_mass/y")[0].text),
            float(doc.xpath("/report/detail/end_center_of_mass/z")[0].text),
        ]
    )
    return start_com, end_com
